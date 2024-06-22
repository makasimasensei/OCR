import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyclipper
from shapely import Polygon

from graduation_project.Architecture.mobilenet_model import My_MobileNet
from graduation_project.FasterNet.fasternet_model import My_FasterNet
from graduation_project.LoadData.load_data import MyDataloader
from graduation_project.Loss.loss import DBLoss
from graduation_project.SegFormer.backbone import *
from graduation_project.PostProcess.Score import score_fun
from graduation_project.resnet.model import My_ResNet


class Eval:
    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = 0
        self.input_path = args.input_path
        self.train_label_path = os.path.join(self.input_path, "train.txt")
        self.eval_label_path = os.path.join(self.input_path, "eval.txt")
        self.db_loss = DBLoss().to(self.device)
        self.checkpoint = torch.load(args.checkpoints)
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.show_map = args.show_map
        self.precision = 0
        self.recall = 0
        self.f1_score = 0

        if (args.MobileNetV3 + args.FasterNet + args.SegFormer + args.ResNet) > 1:
            raise Exception("MobileNetV3, FasterNet, SegFormer, only one model can be chosen")
        elif (args.MobileNetV3 + args.FasterNet + args.SegFormer + args.ResNet) < 1:
            raise Exception("MobileNetV3, FasterNet, SegFormer, choose at least one model")

        if args.SegFormer is True:
            if (
                    args.mit_b0 + args.mit_b1 + args.mit_b2 + args.mit_b3 + args.mit_b4 + args.mit_b5 + args.bm_b0 + args.bm_b1 + args.bm_b2 + args.bm_b3 + args.bm_b4 + args.bm_b5) > 1:
                raise Exception("only one mit model can be chosen")
            elif (
                    args.mit_b0 + args.mit_b1 + args.mit_b2 + args.mit_b3 + args.mit_b4 + args.mit_b5 + args.bm_b0 + args.bm_b1 + args.bm_b2 + args.bm_b3 + args.bm_b4 + args.bm_b5) < 1:
                raise Exception("make sure at least one mit model is chosen")
            if args.mit_b0 is True:
                self.mynet = MiT_b0().to(self.device)
            elif args.mit_b1 is True:
                self.mynet = MiT_b1().to(self.device)
            elif args.mit_b2 is True:
                self.mynet = MiT_b2().to(self.device)
            elif args.mit_b3 is True:
                self.mynet = MiT_b3().to(self.device)
            elif args.mit_b4 is True:
                self.mynet = MiT_b4().to(self.device)
            elif args.mit_b5 is True:
                self.mynet = MiT_b5().to(self.device)
            elif args.bm_b0 is True:
                self.mynet = BM_b0().to(self.device)
            elif args.bm_b1 is True:
                self.mynet = BM_b1().to(self.device)
            elif args.bm_b2 is True:
                self.mynet = BM_b2().to(self.device)
            elif args.bm_b3 is True:
                self.mynet = BM_b3().to(self.device)
            elif args.bm_b4 is True:
                self.mynet = BM_b4().to(self.device)
            elif args.bm_b5 is True:
                self.mynet = BM_b5().to(self.device)

        if args.MobileNetV3 is True:
            self.mynet = My_MobileNet(args).to(self.device)

        if args.ResNet is True:
            self.mynet = My_ResNet().to(self.device)

        if args.FasterNet is True:
            self.mynet = My_FasterNet(args).to(self.device)

    def __call__(self):
        self.mynet.load_state_dict(self.checkpoint['dic'])
        local_eval_dataloader = (MyDataloader(
            self.input_path, self.train_label_path, self.eval_label_path).
                                 eval_dataloader_fun(batch_size=self.batch_size, num_workers=self.num_workers))
        with torch.no_grad():
            for data in local_eval_dataloader:
                input_x = data['img_norm'].to('cuda')
                output_mynet = self.mynet(input_x).to('cuda')

                if self.show_map is True:
                    plt.subplot(2, 2, 1)
                    plt.imshow(data['image'][0])
                    plt.subplot(2, 2, 2)
                    img1 = output_mynet[0, 0, :, :]
                    plt.imshow(img1.cpu().detach().numpy())
                    plt.subplot(2, 2, 3)
                    img2 = output_mynet[0, 1, :, :]
                    plt.imshow(img2.cpu().detach().numpy())
                    plt.subplot(2, 2, 4)
                    img3 = output_mynet[0, 2, :, :]
                    plt.imshow(img3.cpu().detach().numpy())
                    plt.show()

                # post_print(output_mynet[0], data['img_path'])

                loss_all = self.db_loss(data, output_mynet)
                self.loss += loss_all

                p, r, f = score_fun(data, output_mynet)
                self.precision += p
                self.recall += r
                self.f1_score += f

            print('-----测试集的损失函数是：-----{}'.format(self.loss / len(local_eval_dataloader) / self.batch_size))
            print('-----Precision是：-----{}'.format(self.precision / len(local_eval_dataloader)))
            print('-----Recall是：-----{}'.format(self.recall / len(local_eval_dataloader)))
            print('-----F1 Score是：-----{}'.format(self.f1_score / len(local_eval_dataloader)))


def post_print(x, input_pic):
    img = cv2.imread(input_pic[0])
    h, w, _ = img.shape

    original = torch.permute(x, (1, 2, 0))
    original = original.cpu().detach().numpy()
    original = cv2.resize(original, (w, h))

    image = x[0].cpu().detach().numpy()
    _, binary_image = cv2.threshold(image, 0.3, 1, cv2.THRESH_BINARY)
    binary_image = cv2.resize(binary_image, (w, h))
    binary_image = np.uint8(binary_image)

    # 使用cv2.connectedComponentsWithStats获取连通组件信息
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)

    # 输出每个连通组件的面积（注意：第一个标签是背景，所以从1开始）
    areas = stats[1:, cv2.CC_STAT_AREA]

    # 找到面积为150的组件的标签
    small_area_labels = np.where(areas <= 50)[0] + 1  # 加1是因为标签从1开始

    # 将面积为150的像素置零
    for label in small_area_labels:
        binary_image[labels == label] = 0

    # 定义结构元素（这里使用一个较小的矩形结构元素，你可以根据实际情况调整）
    kernel = np.ones((7, 7), np.uint8)

    # 执行膨胀操作，迭代两次（你可以根据需要调整结构元素的大小和迭代次数）
    dilate_image = cv2.dilate(binary_image, kernel, iterations=1)

    # 使用 Canny 边缘检测算法
    edges = cv2.Canny(dilate_image, 0, 1)  # 调整参数以适应不同图像

    # 寻找边缘点的坐标
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fig, ax = plt.subplots()
    ax.imshow(img)

    # 在图像上绘制多边形
    for i, contour in enumerate(contours):
        # 将轮廓坐标转换为 (x, y) 格式
        contour_points = contour.squeeze()

        # 寻找包围多边形的最小矩形
        rect = cv2.minAreaRect(contour_points)

        # 将最小矩形的信息转换为矩形的四个顶点坐标
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        poly = Polygon(box)
        distance = poly.area * 1 / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        try:
            boxes = expanded.squeeze(axis=0)
        except:
            continue
        coordinates = [[point[0], point[1]] for point in boxes]
        x, y = zip(*coordinates)

        # 闭合轮廓
        x = np.append(x, x[0])
        y = np.append(y, y[0])

        # 绘制轮廓
        ax.plot(x, y, color='red')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    # 测试模型架构
    parser.add_argument('--MobileNetV3', action='store_true',
                        help='backbone using MobileNet. Paper: https://arxiv.org/pdf/1905.02244.pdf')
    parser.add_argument('--ResNet', action='store_true',
                        help='backbone using ResNet. Paper: https://arxiv.org/abs/1512.03385')
    parser.add_argument('--FasterNet', action='store_true',
                        help='backbone using FasterNet. Paper: https://arxiv.org/pdf/2303.03667.pdf')
    parser.add_argument('--FPN', action='store_true',
                        help='Feature Pyramid Network. Paper: https://arxiv.org/pdf/1612.03144.pdf')
    parser.add_argument('--FPN_ASF', action='store_true',
                        help='FPN with ASF. Paper: https://arxiv.org/pdf/2202.10304.pdf')

    parser.add_argument('--SegFormer', action='store_true',
                        help='Model is SegFormer. Paper: https://arxiv.org/abs/2105.15203')
    parser.add_argument('--mit_b0', action='store_true', help='Model type')
    parser.add_argument('--mit_b1', action='store_true', help='Model type')
    parser.add_argument('--mit_b2', action='store_true', help='Model type')
    parser.add_argument('--mit_b3', action='store_true', help='Model type')
    parser.add_argument('--mit_b4', action='store_true', help='Model type')
    parser.add_argument('--mit_b5', action='store_true', help='Model type')

    parser.add_argument('--bm_b0', action='store_true', help='Model type')
    parser.add_argument('--bm_b1', action='store_true', help='Model type')
    parser.add_argument('--bm_b2', action='store_true', help='Model type')
    parser.add_argument('--bm_b3', action='store_true', help='Model type')
    parser.add_argument('--bm_b4', action='store_true', help='Model type')
    parser.add_argument('--bm_b5', action='store_true', help='Model type')

    # 测试参数
    parser.add_argument('--input_path', default="E:/anaconda/envs/pytorch/graduation_project/demo", type=str,
                        help='输入')
    parser.add_argument('--checkpoints', default=None, type=str, help='保存的参数模型')
    parser.add_argument('--batch_size', default=4, type=int, help='批次大小')
    parser.add_argument('--num_workers', default=4, type=int, help='工作进程数')
    parser.add_argument('--show_map', action='store_true', help='显示阈值图，概率图，近似二值化图')
    args = parser.parse_args()

    eval_mode = Eval(args)
    eval_mode()


if __name__ == '__main__':
    main()
