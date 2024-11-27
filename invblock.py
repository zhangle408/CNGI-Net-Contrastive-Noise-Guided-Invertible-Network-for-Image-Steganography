from math import exp
import torch
import torch.nn as nn
from denseblock import Dense
import config_image_s1 as c


class INV_block_addition(nn.Module):
    def __init__(self, subnet_constructor=Dense, clamp=c.clamp, harr=True, in_1=3, in_2=3):
        super().__init__()

        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
        self.clamp = clamp

        # ρ
        # self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        # self.s2 = subnet_constructor(self.split_len2, self.split_len1)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        if not rev:

            t2 = self.f(x2)
            y1 = x1 + t2
            t1 = self.y(y1)
            y2 = x2 + t1

        else:  # names of x and y are swapped!

            t1 = self.y(x1)
            y2 = (x2 - t1)
            t2 = self.f(y2)
            y1 = (x1 - t2)

        return torch.cat((y1, y2), 1)


class INV_block_affine(nn.Module):
    def __init__(self, subnet_constructor=Dense, clamp=c.clamp, harr=True, in_1=6, in_2=3, imp_map=True):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
        self.clamp = clamp
        if imp_map:
            self.imp = 12
        else:
            self.imp = 0

        # ρ
        self.r = subnet_constructor(self.split_len1 + self.imp, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1 + self.imp, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1 + self.imp)
        # ψ
        self.p = subnet_constructor(self.split_len2, self.split_len1 + self.imp)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):

        x1, x2 = (x.narrow(1, 0, self.split_len1 + self.imp),
                  x.narrow(1, self.split_len1 + self.imp, self.split_len2))

        if not rev:

            t2 = self.f(x2)
            s2 = self.p(x2)
            y1 = self.e(s2) * x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1

        else:  # names of x and y are swapped!

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            s2 = self.p(y2)
            y1 = (x1 - t2) / self.e(s2)

        return torch.cat((y1, y2), 1)



class NGI_block(nn.Module):
    def __init__(self, subnet_constructor=Dense, clamp=c.clamp, harr=True, in_1=3, in_2=3, imp_map=True):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
            self.split_len3 = 3 * 4
        self.clamp = clamp
        if imp_map:
            self.imp = 12
        else:
            self.imp = 0

        # ρ
        self.r = subnet_constructor(self.split_len1 + self.imp, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1 + self.imp, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1 + self.imp)
        # ψ
        self.p = subnet_constructor(self.split_len2, self.split_len1 + self.imp)
        # lamda1
        self.l1 = subnet_constructor(self.split_len1, self.split_len3)
        # lamda2
        self.l2 = subnet_constructor(self.split_len2, self.split_len3)
        # gamma
        self.g = subnet_constructor(self.split_len3, self.split_len1)

        self.sigma = subnet_constructor(self.split_len3, self.split_len2)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):

        x1, x2, x3 = (x.narrow(1, 0, self.split_len1 + self.imp),
                  x.narrow(1, self.split_len1 + self.imp, self.split_len2),
                      x.narrow(1, self.split_len2, self.split_len3))

        if not rev:

            s2 = self.p(x2)
            y1 = self.e(s2) * x1 + self.f(x2) + self.g(x3)
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1+ self.sigma(x3)
            y3 = x3+self.l1(x1)+self.l2(x2)

        else:  # names of x and y are swapped!

            y3 = x3 - self.l1(x1) - self.l2(x2)
            s1, t1 = self.r(x1), self.y(x1)
            t3 = self.g(y3)
            y2 = (x2 - t1 -self.sigma(y3)) / self.e(s1)
            t2 = self.f(y2)
            s2 = self.p(y2)
            y1 = (x1 - t2 - t3) / self.e(s2)


        return torch.cat((y1, y2, y3), 1)
