from trn.trainer import *
from eval import *
from utilities.initialize_parser import *


def main():
    args = initialize_parser()
    if args.mode == "--train":
        trainer = Trainer(args)
        trainer()
    elif args.mode == "--eval":
        evaler = Eval(args)
        evaler()


if __name__ == '__main__':
    main()
