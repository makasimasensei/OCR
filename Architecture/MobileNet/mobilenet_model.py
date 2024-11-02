from Architecture.FPN.FPN_ASF import FeaturePyramid_ASF
from Architecture.MobileNet.MobileV3Large import *
from Architecture.FPN.FPN import *


class My_MobileNet(nn.Module):
    def __init__(self, args):
        super(My_MobileNet, self).__init__()
        self.step1 = MobileNetV3()
        assert (args.FPN + args.FPN_ASF) > 1, \
            "only one FPN model can be chosen"
        assert (args.FPN + args.FPN_ASF) < 1, \
            "choose at least one FPN model"
        if args.FPN is True:
            self.step2 = FeaturePyramid()
        elif args.FPN_ASF is True:
            self.step2 = FeaturePyramid_ASF()

    def forward(self, x):
        result = self.step1(x)
        x = self.step2(result)
        return x
