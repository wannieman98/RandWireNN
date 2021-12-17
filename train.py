import trainer
import argparse


def main(param):
    model = trainer.Trainer(p=param.p,
                            k=param.k,
                            m=param.m,
                            lr=param.lr,
                            path=param.path,
                            dataset=param.dataset,
                            channel=param.channel,
                            is_train=param.is_train,
                            num_node=param.num_node,
                            num_epoch=param.num_epoch,
                            graph_mode=param.graph_mode,
                            batch_size=param.batch_size,
                            in_channels=param.in_channels,
                            is_small_regime=param.is_small_regime,
                            checkpoint_path=param.checkpoint_path,
                            load=param.load)
    model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Randomly Wired Neural Network')
    parser.add_argument('--p', type=float, default=0.75)
    parser.add_argument('--k', type=float, default=4)
    parser.add_argument('--m', type=float, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--path', type=str, default='/content/')
    parser.add_argument('--dataset', type=str, default='voc')
    parser.add_argument('--channel', type=int, default=78)
    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--num_node', type=int, default=32)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--graph_mode', type=str, default='WS')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--is_small_regime', type=bool, default=True)
    parser.add_argument('--checkpoint_path', type=str, default="./data/checkpoints/")
    parser.add_argument('--load', type=bool, default=False)
    args = parser.parse_args()
    main(args)
