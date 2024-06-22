import time
from train import *
from timefun import *


class Trainer:
    def __init__(self, args):
        self.args = args
        self.input_path = args.input_path
        self.epochs = args.epoch
        self.checkpoints = args.checkpoints
        self.runtime = 0
        self.RED = '\x1b[31m'
        self.RESET = '\x1b[0m'

    def __call__(self):
        if self.checkpoints is not None:
            checkpoint = torch.load(self.checkpoints)
            current_epoch = checkpoint['epoch'] + 1
            current_loss = checkpoint['loss']
            current_eval = checkpoint['loss_eval']
            current_lr = checkpoint['lr']
            current_weight_decay = checkpoint['weight_decay']
            current_augmentation = checkpoint['augmentation']
            print(
                '-----训练轮次是：{red}{epoch}{reset}-----:'.format(red=self.RED, epoch=current_epoch, reset=self.RESET))
            print('-----训练集的损失函数是：{red}{loss}{reset}-----'.format(red=self.RED, loss=current_loss,
                                                                           reset=self.RESET))
            print('-----测试集的损失函数是：{red}{eval}{reset}-----'.format(red=self.RED, eval=current_eval,
                                                                           reset=self.RESET))
            print('-----学习率是：{red}{lr}{reset}-----'.format(red=self.RED, lr=current_lr, reset=self.RESET))
            print('-----l2正则化系数是：{red}{weight_decay}{reset}-----'.format(red=self.RED,
                                                                               weight_decay=current_weight_decay,
                                                                               reset=self.RESET))
            print('-----是否使用数据增强：{red}{augmentation}{reset}-----'.format(red=self.RED,
                                                                                 augmentation=current_augmentation,
                                                                                 reset=self.RESET))
        else:
            current_epoch = 0

        start_train = Train(self.args)
        for epoch in range(0, self.epochs - current_epoch):
            start_time = time.time()
            print("-----第{}轮训练开始-----".format(current_epoch + epoch + 1))
            start_train(current_epoch + epoch)
            torch.cuda.empty_cache()
            end_time = time.time()
            total_time = end_time - start_time
            self.runtime += total_time
            eta = (self.runtime / (epoch + 1)) * (self.epochs - epoch - current_epoch)
            print("第{}轮训练时间是：{}".format(epoch + current_epoch + 1, total_time))
            time_fun(int(eta))
