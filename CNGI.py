from model import *
from invblock import NGI_block

class NGI(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self):
        super(NGI, self).__init__()

        self.inv1 = NGI_block(imp_map=False)
        self.inv2 = NGI_block(imp_map=False)
        self.inv3 = NGI_block(imp_map=False)
        self.inv4 = NGI_block(imp_map=False)
        self.inv5 = NGI_block(imp_map=False)
        self.inv6 = NGI_block(imp_map=False)
        self.inv7 = NGI_block(imp_map=False)
        self.inv8 = NGI_block(imp_map=False)


    def forward(self, x, rev=False):

        if not rev:
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
            out = self.inv5(out)
            out = self.inv6(out)
            out = self.inv7(out)
            out = self.inv8(out)



        else:


            out = self.inv8(x, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv5(out, rev=True)
            out = self.inv4(out, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)

        return out


