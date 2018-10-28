import torch
import torch.nn as nn

from torch.nn import init
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import linear_model

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



def build_bl_net(norm=AdaptiveNorm, layer=5):
    layers = [
        nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),
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


def sparse_bls(H1, A1, lam, itrs, lr):
    W = torch.rand(H1.shape[1], A1.shape[0], requires_grad=True).cuda()
    for i in range(itrs):
        y = torch.mm(H1, W)
        y.backwards()
        loss = torch.norm(y - A1, 2) ** 2 + lam * torch.norm(W, 2)
        loss.backwards()
        grad = W.grad
        W += lr * grad
    return W



class BLS(nn.Module):
    def __init__(self, N1, N2, N3, input=1024, output=512):
        super(BLS, self).__init__()
        # define the structure of BLS
        self.predict = nn.Linear(input, output)
        self.N1, self.N2, self.N3 = N1, N2, N3
        self.We = []
        self.y = []

    def forward(self, x):
        #a_flat = input.view(input.numel())
        #m = nn.Sigmoid()
        #feature_nodes = m(self.feature_mapping(a_flat))
        #enhance_nodes = m(self.feature_enhancement(feature_nodes))
        #output = self.predict(torch.cat((feature_nodes, enhance_nodes)))
        #output = output.view(input.shape)
        #return output

        H = x.view(1, -1)

        #todo
        # param
        # @N1: number of feature nodes of each window
        # @N2: number of windows
        # @N3: number of enhanced nodes
        # (1) generate feature nodes
        # 1. normalize H (the input is already normalized)
        # 2. add bias to H to obtain H1
        H1 = torch.cat([H, torch.tensor([[1.]]).cuda()], dim=1)
        self.We = torch.empty([self.N2, H1.shape[1], self.N1]).cuda()
        self.y = torch.empty([self.N2, self.N1])
        # 3. generate feature nodes for each window (iteration index i <= N2)
        for i in range(self.N2):
            # 3.1 randomly generate weights matrix we(i) subject to Gaussian distribution
            self.We[i] = torch.randn(H1.shape[1], self.N1)
            A1 = torch.mm(H1, self.We[i])
            A1 = F.normalize(A1)
            W = sparse_bls(H1, A1, 1e-3, 100, 0.01)
            print(A1)
            # 3.2 generate feature nodes A(i) = H1 * we(i)
            # 3.3 normalize A(i)
            # 3.4 optimize a function to obtain W such that H1 * W = A(i)
            # 3.5 finally get the feature node T(i) of this window by normal(H * W)

        # 4. obtain the feature matrix y
        # (2) generate enhanced nodes (add nonlinear property to the model)
        # 1. normalize and add bias y to obtain H2
        # 2. generate a random and normalized weight matrix wh
        # 3. activate enhanced nodes
        # (3) generate the final input

        return self.predit(H1)

class BroadGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-8,):
        super(BroadGuidedFilter, self).__init__()
        self.bl = BLS(10, 10, 10)
        self.gf = FastGuidedFilter(radius, eps)


    def forward(self, x_lr, x_hr):
        #img = x_lr.cpu()[0]
        #plt.imshow(img[0] * 255)
        maps = self.bl(x_lr)
        return self.gf(x_lr, maps, x_hr).clamp(0, 1)

