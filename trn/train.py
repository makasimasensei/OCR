import os
from matplotlib import pyplot as plt

from Architecture.resnet.model import My_ResNet
from Architecture.MobileNet.mobilenet_model import My_MobileNet
from Architecture.FasterNet.fasternet_model import My_FasterNet
from Architecture.SegFormer.backbone import *
from LoadData.load_data import MyDataloader
from Loss.loss import DBLoss


class Train:
    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.local_ic15_data_path = args.input_path
        self.local_train_data_label_ = os.path.join(self.local_ic15_data_path, "train.txt")
        self.local_eval_data_label_ = os.path.join(self.local_ic15_data_path, "eval.txt")

        self.RED = '\x1b[31m'
        self.Green = '\x1b[32m'
        self.RESET = '\x1b[0m'

        if (args.MobileNetV3 + args.FasterNet + args.SegFormer + args.ResNet) > 1:
            raise Exception("MobileNetV3, FasterNet, SegFormer, ResNet, only one model can be chosen")
        elif (args.MobileNetV3 + args.FasterNet + args.SegFormer + args.ResNet) < 1:
            raise Exception("MobileNetV3, FasterNet, SegFormer, ResNet, choose at least one model")

        if args.SegFormer is True:
            if (args.mit_b0 + args.mit_b1 + args.mit_b2 + args.mit_b3 + args.mit_b4 + args.mit_b5 + args.bm_b0 + args.bm_b1 + args.bm_b2 + args.bm_b3 + args.bm_b4 + args.bm_b5 + args.ES_out_FPN + args.ES_with_FPN) > 1:
                raise Exception("only one mit model can be chosen")
            elif (args.mit_b0 + args.mit_b1 + args.mit_b2 + args.mit_b3 + args.mit_b4 + args.mit_b5 + args.bm_b0 + args.bm_b1 + args.bm_b2 + args.bm_b3 + args.bm_b4 + args.bm_b5 + args.ES_out_FPN + args.ES_with_FPN) < 1:
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
            elif args.ES_out_FPN is True:
                self.mynet = ES_out_FPN().to(self.device)
            elif args.ES_with_FPN is True:
                self.mynet = ES_with_FPN().to(self.device)

        if args.MobileNetV3 is True:
            self.mynet = My_MobileNet(args).to(self.device)

        if args.ResNet is True:
            self.mynet = My_ResNet().to(self.device)

        if args.FasterNet is True:
            self.mynet = My_FasterNet(args).to(self.device)

        self.db_loss = DBLoss().to(self.device)
        self.loss = 0
        self.loss_eval = 0
        self.base_lr = args.lr
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.augmentation = args.augmentation
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        if args.eval:
            self.eval = args.eval
            self.local_eval_dataloader = (MyDataloader(
                self.local_ic15_data_path, self.local_train_data_label_, self.local_eval_data_label_).
                                          eval_dataloader_fun(batch_size=self.batch_size, num_workers=self.num_workers))
        else:
            self.eval = False

        if args.pretrained is not None:
            self.checkpoint = torch.load(args.pretrained)
            self.mynet.load_state_dict(self.checkpoint['dic'])
        elif args.checkpoints is not None:
            self.checkpoint = torch.load(args.checkpoints)
            self.mynet.load_state_dict(self.checkpoint['dic'])
            self.lr = self.checkpoint['lr']
        self.optim = torch.optim.Adam(self.mynet.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def __call__(self, local_epoch):
        local_train_dataloader = (MyDataloader(
            self.local_ic15_data_path, self.local_train_data_label_, self.local_eval_data_label_).
                                  train_dataloader_fun(batch_size=self.batch_size, num_workers=self.num_workers,
                                                       augmentation=self.augmentation))
        local_lr = self.custom_lr_lambda(base_lr=self.base_lr, epoch=local_epoch)

        for data in local_train_dataloader:
            self.train_processing(data)

            for p in self.optim.param_groups:
                p['lr'] = local_lr

        print('学习率是:{}'.format(local_lr))

        if (local_epoch + 1) % 1 == 0:
            if self.eval:
                with torch.no_grad():
                    for data in self.local_eval_dataloader:
                        self.loss_eval += self.eval_processing(data)
                    print('测试集的损失函数是：{red}{loss}{reset}'.format(
                        red=self.RED,
                        loss=self.loss_eval / len(self.local_eval_dataloader) / self.batch_size,
                        reset=self.RESET)
                    )
            torch.save({'epoch': local_epoch, 'dic': self.mynet.state_dict(),
                        'loss': self.loss / len(local_train_dataloader) / self.batch_size,
                        'loss_eval': self.loss_eval / len(self.local_eval_dataloader) / self.batch_size,
                        'lr': local_lr,
                        'weight_decay': self.weight_decay,
                        'augmentation': self.augmentation
                        },
                       'E:/anaconda/envs/pytorch/graduation_project/checkpoints'
                       '/model_weights_{}.pth'.format(local_epoch + 1))
        self.loss_eval = 0
        print('训练集集的损失函数是：{green}{loss}{reset}'.format(
            green=self.Green,
            loss=self.loss / len(local_train_dataloader) / self.batch_size,
            reset=self.RESET)
        )
        self.loss = 0

    def train_processing(self, data_list):
        input_x = data_list['img_norm'].to(self.device)
        output_mynet = self.mynet(input_x)
        # plt.figure(figsize=(8, 8))
        # plt.subplot(2, 2, 1)
        # image = data_list['image'][0]
        # plt.imshow(image)
        # plt.subplot(2, 2, 2)
        # img1 = output_mynet[0, 0, :, :]
        # plt.imshow(img1.cpu().detach().numpy())
        # plt.subplot(2, 2, 3)
        # img2 = output_mynet[0, 1, :, :]
        # plt.imshow(img2.cpu().detach().numpy())
        # plt.subplot(2, 2, 4)
        # img3 = output_mynet[0, 2, :, :]
        # plt.imshow(img3.cpu().detach().numpy())
        # plt.tight_layout()
        # plt.show()

        loss_all = self.db_loss(data_list, output_mynet)
        self.loss += loss_all
        self.optim.zero_grad()
        loss_all.backward()
        self.optim.step()

    def eval_processing(self, data_list):
        input_x = data_list['img_norm'].to(self.device)
        output_mynet = self.mynet(input_x).to(self.device)

        loss_all = self.db_loss(data_list, output_mynet)
        return loss_all

    @staticmethod
    def custom_lr_lambda(base_lr, epoch, num_epoch=1200, power=1.0):
        return base_lr * (1 - epoch / num_epoch) ** power
