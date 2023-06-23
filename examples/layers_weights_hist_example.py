import torchvision
from torchvision.models import resnet50

from pytorch_debug_tools.LayersHistogramsVisualizer import LayersHistogramsVisualizer

if __name__ == "__main__":
    # check layers weights histograms for pretrained model
    if torchvision.__version__ < '0.13.0':
        pretrained_resnet50 = resnet50(pretrained=True)
    else:
        resnet50(weights="IMAGENET1K_V2")
