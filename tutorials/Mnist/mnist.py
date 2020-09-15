"""
@author: yangqiang
@contact: whuhit2020@gmail.com
@file: demo.py
@time: 2020/9/10 10:22
"""
import argparse
from utils.config import process_config
from agents import MnistAgent
from torchvision import datasets, transforms
import torch


def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        '-c',
        '--config',
        metavar='config_json_file',
        default='configs/mnist_exp_0.json',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()

    config = process_config(args.config)

    agent = MnistAgent(config)
    # agent.run()

    if True:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=1)

        for data, target in test_loader:
            out = agent.inference(data.cuda())
            pre = out.max(1, keepdim=True)
            print(pre[1][0].item(), target.item())
            # break


if __name__ == '__main__':
    main()
