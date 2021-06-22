import matplotlib
import matplotlib.cm
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.utils import make_grid


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def colorize(value, vmin=0, vmax=255, cmap='magma_r'):
    value = value[:,:,0]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    #value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)

    img = value[:,:,:3]

    return img

def save_predictions_dataset(
        loader, model, folder ="saved_images_squeeze/",device="cuda"
):
    model.eval()
    for index, sample in enumerate(loader):
        x = torch.autograd.Variable(sample[0].cuda())
        y = torch.autograd.Variable(sample[1].cuda(non_blocking=True))

        with torch.no_grad():
            preds = model(x)
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{index}.png"
        )
        torchvision.utils.save_image(y, f"{folder}{index}.png")

    model.train()
