import argparse
import cv2

from MyNet.backbone import *
from Preprocessing.image_preprocessing import Normalize_image
from PostProcess.postprint import post_print


# from graduation_project.resnet.model import My_ResNet


class Detection:
    def __init__(self, comboBoxText1, comboBoxText2):
        if comboBoxText2 == "WBFFPN":
            self.mynet = BM_b0().to('cuda')
        checkpoints = "E:\\anaconda\\envs\\pytorch\\graduation_project\\checkpoints\\model_weights_689.pth"
        self.checkpoint = torch.load(checkpoints)
        self.norm = Normalize_image()

    def __call__(self, img_path):
        self.mynet.load_state_dict(self.checkpoint['dic'])
        img = cv2.imread(img_path)
        image = self.norm(img)
        input_x = image['img_norm'].to('cuda')
        output_mynet = self.mynet(input_x)
        x = output_mynet[0]

        return x


def main():
    pic = "C:/Users/14485/Pictures/Screenshots/gt_0.jpg"
    comboBoxText1 = "WBFFPN"
    comboBoxText2 = "WBFFPN"

    demo_mode = Detection(comboBoxText1, comboBoxText2)
    out = demo_mode(pic)
    post_print(out, pic, 1)


if __name__ == '__main__':
    main()
