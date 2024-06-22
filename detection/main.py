from trainer import *
from eval import *


def main():
    parser = argparse.ArgumentParser()
    # 训练模型架构
    parser.add_argument('--MobileNetV3', action='store_true',
                        help='backbone using MobileNet. Paper: https://arxiv.org/pdf/1905.02244')
    parser.add_argument('--ResNet', action='store_true',
                        help='backbone using ResNet. Paper: https://arxiv.org/abs/1512.03385')
    parser.add_argument('--FasterNet', action='store_true',
                        help='backbone using FasterNet. Paper: https://arxiv.org/pdf/2303.03667')
    parser.add_argument('--FPN', action='store_true',
                        help='Feature Pyramid Network. Paper: https://arxiv.org/pdf/1612.03144')
    parser.add_argument('--FPN_ASF', action='store_true',
                        help='FPN with ASF. Paper: https://arxiv.org/pdf/2202.10304')

    parser.add_argument('--SegFormer', action='store_true',
                        help='Model is SegFormer. Paper: https://arxiv.org/abs/2105.15203')
    parser.add_argument('--ES_out_FPN', action='store_true', help='Model type')
    parser.add_argument('--ES_with_FPN', action='store_true', help='Model type')
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

    # 训练参数
    parser.add_argument('--mode', default='--train', type=str, help='train or eval')
    parser.add_argument('--eval', action='store_false', help='eval or not')
    parser.add_argument('--input_path', default="E:/anaconda/envs/pytorch/graduation_project/demo",
                        type=str, help='输入')
    parser.add_argument('--epoch', default=1200, type=int, help='训练的总轮数')
    parser.add_argument('--checkpoints', default=None, type=str, help='保存的参数模型')
    parser.add_argument('--pretrained', default=None, type=str, help='预训练的模型参数')
    parser.add_argument('--lr', default=0.00006, type=float, help='训练的学习率')
    parser.add_argument('--weight_decay', default=0.001, type=float, help='l2正则化系数')
    parser.add_argument('--augmentation', action='store_true', help='是否使用数据增强')
    parser.add_argument('--batch_size', default=4, type=int, help='批次大小')
    parser.add_argument('--num_workers', default=4, type=int, help='工作进程数')

    args = parser.parse_args()

    if args.mode == "--train":
        trainer = Trainer(args)
        trainer()
    elif args.mode == "--eval":
        evaler = Eval(args)
        evaler()


if __name__ == '__main__':
    main()
