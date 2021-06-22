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
    
def save_checkpoint(state,filename="my_checkpoint.pth.tar"):
    print("Saving checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint,model):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def evaluate(output, target):
    valid_mask = ((target>0) + (output>0)) > 0

    output = 1e3 * output[valid_mask]
    target = 1e3 * target[valid_mask]
    abs_diff = (output - target).abs()

    mse = float((torch.pow(abs_diff, 2)).mean())
    rmse = math.sqrt(mse)
    mae = float(abs_diff.mean())
    absrel = float((abs_diff / target).mean())

    maxRatio = torch.max(output / target, target / output)
    delta1 = float((maxRatio < 1.25).float().mean())
    delta2 = float((maxRatio < 1.25 ** 2).float().mean())
    delta3 = float((maxRatio < 1.25 ** 3).float().mean())

    inv_output = 1 / output
    inv_target = 1 / target
    abs_inv_diff = (inv_output - inv_target).abs()
    irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
    imae = float(abs_inv_diff.mean())

    return rmse, mae, absrel, delta1, delta2, delta3, irmse, imae
