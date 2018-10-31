import torch
import torch.nn as nn

from torch.nn import init
import matplotlib.pyplot as plt

from guided_filter_pytorch.guided_filter import FastGuidedFilter


def weights_init_identity(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n_out, n_in, h, w = m.weight.data.size()
        # Last Layer
        if n_out < n_in:
            init.xavier_uniform(m.weight.data)
            return

        # Except Last Layer
        m.weight.data.zero_()
        ch, cw = h // 2, w // 2
        for i in range(n_in):
            m.weight.data[i, i, ch, cw] = 1.0

    elif classname.find('BatchNorm2d') != -1:
        init.constant(m.weight.data, 1.0)
        init.constant(m.bias.data, 0.0)


class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.bn = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)


def build_lr_net(norm=AdaptiveNorm, layer=5):
    layers = [
        nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),
    ]

    for l in range(1, layer):
        layers += [nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=2 ** l, dilation=2 ** l, bias=False),
                   norm(24),
                   nn.LeakyReLU(0.2, inplace=True)]

    layers += [
        nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(24, 3, kernel_size=1, stride=1, padding=0, dilation=1)
    ]

    net = nn.Sequential(*layers)

    net.apply(weights_init_identity)

    return net


class DeepGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-8):
        super(DeepGuidedFilter, self).__init__()
        self.lr = build_lr_net()
        self.gf = FastGuidedFilter(radius, eps)

    def forward(self, x_lr, x_hr):
        maps = self.lr(x_lr)
        return self.gf(x_lr, maps, x_hr).clamp(0, 1)

    def init_lr(self, path):
        self.lr.load_state_dict(torch.load(path))


class DeepGuidedFilterAdvanced(DeepGuidedFilter):
    def __init__(self, radius=1, eps=1e-4):
        super(DeepGuidedFilterAdvanced, self).__init__(radius, eps)

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, 15, 1, bias=False),
            AdaptiveNorm(15),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(15, 3, 1)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x_lr, x_hr):
        return self.gf(self.guided_map(x_lr), self.lr(x_lr), self.guided_map(x_hr))


def shrinkage(x, kappa):
    a = torch.add(x, -kappa)
    b = torch.add(-x, -kappa)
    c = torch.gt(a, 0).float()
    d = torch.gt(b, 0).float()
    result = torch.mul(a, c) - torch.mul(b, d)
    return result


def sparse_bls(A1, H1, lam, itrs):
    AA = torch.mm(torch.t(A1), A1)
    m = A1.shape[1]
    n = H1.shape[1]
    x = torch.zeros(m, n).cuda()
    wk = x
    ok = x
    uk = x
    L1 = torch.eye(m).cuda() / (AA + torch.eye(m).cuda())
    L2 = torch.mm(torch.mm(L1, torch.t(A1)), H1)
    for i in range(itrs):
        tempc = ok - uk
        ck = L2 + torch.mm(L1, tempc)
        ok = shrinkage(ck + uk, lam)
        uk = uk + (ck - ok)
        wk = ok
    return wk


def product(sequence):
    p = 1
    for e in sequence:
        p *= e
    return p

class BLS(nn.Module):
    def __init__(self, N1, N2, N3, s, input=210, output=18432):
        super(BLS, self).__init__()
        # define the structure of BLS
        self.predict = nn.Linear(input, output)
        self.N1, self.N2, self.N3 = N1, N2, N3
        self.We = []
        self.Wh = []
        self.ps = []
        self.generated = False
        self.s = s
        self.T3 = []
        self.out_shape = []
        self.in_size = input

    def generate_nodes(self, x):

        self.out_shape = x.size()
        self.predict = nn.Linear(self.in_size, product(x.size()))

        H = x.view(1, -1)  # flatten the input data

        # zscore H
        H = (H - torch.mean(H)) / torch.std(H)

        #todo
        # param
        # @N1: number of feature nodes of each window
        # @N2: number of windows
        # @N3: number of enhanced nodes
        # (1) generate feature nodes
        # 1. normalize H (the input is already normalized)
        # 2. add bias to H to obtain H1
        H1 = torch.cat([H, torch.tensor([[1.]]).cuda()], dim=1)
        self.We = torch.empty(self.N2, H1.shape[1], self.N1).cuda()
        y = torch.empty(H1.shape[0], self.N2 * self.N1).cuda()
        self.ps = []
        # 3. generate feature nodes for each window (iteration index i <= N2)
        for i in range(self.N2):
            # 3.1 randomly generate weights matrix we(i) subject to Gaussian distribution
            we = torch.randn(H1.shape[1], self.N1).cuda()

            # 3.2 generate feature nodes A(i) = H1 * we
            A1 = torch.mm(H1, we)

            # 3.3 normalize A(i)

            A1_mean = torch.mean(A1)
            A1_min = torch.min(A1)
            A1_max = torch.max(A1)
            A1 = (A1 - A1_mean) / (A1_max - A1_min)      # min-max normalization  (-1 to 1)

            # 3.4 optimize a function to obtain W such that H1 * W = A(i)
            # todo here are three hyper-parameters

            self.We[i] = torch.t(sparse_bls(A1, H1, 1e-3, 50))
            # 3.5 finally get the feature node y(i) of this window by normal(H * W)
            T1 = torch.mm(H1, self.We[i])
            T1_min = torch.min(T1)
            T1_max = torch.max(T1)
            T1 = T1.view(1, -1)
            T1 = (T1 - T1_min) / (T1_max - T1_min)  # min-max normalization (0 to 1)
            y[:, self.N1*i:self.N1*(i + 1)] = T1
            self.ps.append( (T1_min, T1_max) )

        # 4. obtain the feature matrix y

        # (2) generate enhanced nodes (add nonlinear property to the model)
        # 1. normalize and add bias y to obtain H2
        H2 = torch.cat([y, torch.tensor([[1.]]).cuda()], dim=1)

        # 2. generate a random and normalized weight matrix wh
        if self.N1 * self.N2 >= self.N3:
            Wh = init.orthogonal(2 * torch.rand(self.N2 * self.N1 + 1, self.N3 ) - 1)
        else:
            Wh = torch.t(init.orthogonal(2 * torch.t(torch.rand(self.N2 * self.N1 + 1, self.N3)) - 1))
        self.Wh = Wh.cuda()
        T2 = torch.mm(H2, self.Wh)
        l2 = torch.max(T2)
        l2 = self.s / l2
        T2 = T2 * l2
        # 3. activate enhanced nodes
        tanh = nn.Tanh()
        T2 = tanh(T2)

        # (3) generate the final input
        self.T3 = torch.cat((y, T2), dim=1)
        self.generated = True

    def forward(self, x):
        if not self.generated:
            self.generate_nodes(x)
        else:
            H = x.view(1, -1)  # flatten the input data

            # zscore H
            H = (H - torch.mean(H)) / torch.std(H)

            H1 = torch.cat([H, torch.tensor([[1.]]).cuda()], dim=1)
            y = torch.empty(H1.shape[0], self.N2 * self.N1).cuda()
            for i in range(self.N2):
                T1 = torch.mm(H1, self.We[i])
                T1_min = self.ps[i][0]
                T1_max = self.ps[i][1]
                T1 = T1.view(1, -1)
                T1 = (T1 - T1_min) / (T1_max - T1_min)  # min-max normalization (0 to 1)
                y[:, self.N1 * i:self.N1 * (i + 1)] = T1
            H2 = torch.cat([y, torch.tensor([[1.]]).cuda()], dim=1)
            T2 = torch.mm(H2, self.Wh)
            l2 = torch.max(T2)
            l2 = self.s / l2
            T2 = T2 * l2
            # 3. activate enhanced nodes
            tanh = nn.Tanh()
            T2 = tanh(T2)

            # (3) generate the final input
            self.T3 = torch.cat((y, T2), dim=1)
        return self.predict(self.T3.cpu())


class BroadGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-8,):
        super(BroadGuidedFilter, self).__init__()
        self.bl_lr = BLS(10, 10, 110, .8)
        self.bl_hr = BLS(10, 10, 110, .8, output=1181184)
        self.gf = FastGuidedFilter(radius, eps)


    def forward(self, x_lr, x_hr):
        #img = x_lr.cpu()[0]
        #plt.imshow(img[0] * 255)

        #print("trian_bgf_forward==>")
        #map_lr = self.bl_lr(x_lr).view(1, 3, 64, 96)
        #map_hr = self.bl_hr(x_hr).view(1, 3, 512, 769)
        #output = self.gf(x_lr, map_lr, map_hr).clamp(0, 1)
        #return output

        maps = self.bl_lr(x_lr).view(x_lr.size()).cuda()
        output = self.gf(x_lr, maps, x_hr).clamp(0, 1)
        return output

class BroadGuidedFilter_lr(nn.Module):
    def __init__(self, radius=1, eps=1e-8,):
        super(BroadGuidedFilter, self).__init__()
        self.bl = BLS(10, 10, 11000, .8)
        self.gf = FastGuidedFilter(radius, eps)


    def forward(self, x_lr, x_hr):
        #img = x_lr.cpu()[0]
        #plt.imshow(img[0] * 255)
        print("trian_bgf_forward==>")
        map_lr = self.bl_lr(x_lr).view(1, 3, 64, 96)
        return self.gf(map_lr, x_lr, map_hr).clamp(0, 1)