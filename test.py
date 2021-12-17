from data.data_util import fetch_dataloader, test_voc
import argparse
import torch


def main(args):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.model_path).to(device)
    model.eval()
    if args.dataset == 'voc':
        _, _, test_loader = fetch_dataloader('voc', args.data_path, 1)

        test_voc(test_loader, model, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test Randomly Wired Neural Network')
    parser.add_argument('--dataset', type=str, default='voc')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()
    main(args)