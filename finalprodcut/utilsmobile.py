import matplotlib
import matplotlib.cm
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.utils import make_grid

def DepthNorm(depth, maxDepth=1000.0):
    return maxDepth / depth

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

def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    value = value.cpu().numpy()[0,:,:]

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

    return img.transpose((2,0,1))

def colorizetest(value, vmin=0, vmax=255, cmap='magma_r'):
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

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def save_image(img, fp):
    grid = make_grid(img)
    img = grid.mul(24).add_(0.5).clamp_(0,255).permute(1,2,0).to('cpu',torch.uint8).numpy()
    img = Image.fromarray(img)
    img.save(fp)

def save_predictions_datasetmobile(
        loader, model, folder ="saved_images_squeeze/",device="cuda"
):
    model.eval()
    for index, sample in enumerate(loader):
        x = torch.autograd.Variable(sample['image'].cuda())
        y = torch.autograd.Variable(sample['depth'].cuda(non_blocking=True))

        with torch.no_grad():
            preds = model(x)
        save_image(
            preds, f"{folder}/pred_{index}.png"
        )
        save_image(1000.0/y, f"{folder}{index}.png")

    model.train()

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
