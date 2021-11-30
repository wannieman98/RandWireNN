import trainer
import argparse


def main(param):
    model = trainer.Trainer()
    model = trainer.Trainer(num_epoch=param.num_epoch, lr=param.lr, batch_size=param.batch_size,
                            num_node=param.num_node, p=param.p, in_channels=param.in_channels, out_channels=param.out_,
                            graph_mode=param.graph_mode, is_train=param.is_train, name=param.name)
    model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Randomly Wired Neural Network')
    parser.add_argument('--num_epoch', type=int, defulat=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_node', type=int, default=8)
    parser.add_argument('--p', type=float, default=0.5)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--out_channels', type=int, default=8)
    parser.add_argument('--graph_mode', type=str, default='ER')
    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--name', type=str, default='rwnn')
    args = parser.parse_args()
    main(args)
