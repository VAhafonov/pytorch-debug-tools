import argparse
import os

import cv2
import torch.nn
import torchvision
from torchvision.models import resnet50

from pytorch_debug_tools.SingleLayerVisualizer import SingleLayerVisualizer
from pytorch_debug_tools.utils.Resolution import Resolution


def load_pretrained_resnet50() -> torch.nn.Module:
    torchvision_version = torchvision.__version__
    major, minor, patch = torchvision_version.split('.')
    if int(major) > 0 or int(minor) >= 13:
        pretrain = resnet50(weights="IMAGENET1K_V2")
    else:
        pretrain = resnet50(pretrained=True)

    return pretrain


def main(args):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    visualizer = SingleLayerVisualizer(Resolution.from_str(args.resolution))

    # load pretrained model based on torchvision version
    pretrained_resnet50 = load_pretrained_resnet50()
    first_conv = pretrained_resnet50.layer1._modules['0'].conv1

    # visualize first conv weights from pretrained resnet50
    hist_from_pretrained_conv = visualizer.visualize(weight=first_conv.weight.detach().numpy(),
                                                     name='conv1', num_bins=args.num_bins)
    out_path = os.path.join(out_dir, 'pretrained_weight.png')
    cv2.imwrite(out_path, hist_from_pretrained_conv)
    print("Done dump for pretrained:", out_path)

    # reinit conv weights with random uniform and visualize
    torch.nn.init.uniform_(first_conv.weight)
    hist_from_uniform_init = visualizer.visualize(weight=first_conv.weight.detach().numpy(),
                                                  name='conv1', num_bins=args.num_bins)
    out_path = os.path.join(out_dir, 'random_uniform_weight.png')
    cv2.imwrite(out_path, hist_from_uniform_init)
    print("Done dump for random uniform:", out_path)

    # reinit conv weights with xavier uniform and visualize
    torch.nn.init.xavier_uniform_(first_conv.weight)
    hist_from_xavier_uniform_init = visualizer.visualize(weight=first_conv.weight.detach().numpy(),
                                                         name='conv1', num_bins=args.num_bins)
    out_path = os.path.join(out_dir, 'xavier_uniform_weight.png')
    cv2.imwrite(out_path, hist_from_xavier_uniform_init)
    print("Done dump for xavier uniform:", out_path)

    # reinit conv weights with kaiming uniform and visualize
    torch.nn.init.kaiming_uniform_(first_conv.weight)
    hist_from_kaiming_uniform_init = visualizer.visualize(weight=first_conv.weight.detach().numpy(),
                                                         name='conv1', num_bins=args.num_bins)
    out_path = os.path.join(out_dir, 'kaiming_uniform_weight.png')
    cv2.imwrite(out_path, hist_from_kaiming_uniform_init)
    print("Done dump for kaiming uniform:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run single layer visualization for first convolution of resnet50')
    parser.add_argument("--out_dir", type=str, help="Path to dir where output images will be located.",
                        default='d:/debug')
    parser.add_argument("--resolution", type=str, help="Resolution which output images will be saved, should be in "
                                                       "form HxW(for example 350x512). Best aspect ration is "
                                                       "somewhere near 0.68", default="350x512")
    parser.add_argument("--num_bins", type=int, help="Desired number of bins in histogram", default=50)
    args_ = parser.parse_args()
    main(args_)
