import torch
import torch.nn.functional as F
import torchvision
import matplotlib
import matplotlib.cm
from dataset import DepthDataset
from torch.utils.data import DataLoader
import math

def save_checkpoint(state,filename="my_checkpoint.pth.tar"):
    print("Saving checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint,model):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
        train_rgb_folder,
        train_depth_folder,
        test_rgb_folder,
        test_depth_folder,
        batch_size,
        transformation,
        num_workers=1,
        pin_memory=True,
):
    train_dataset = DepthDataset(
        rgb_dir=train_rgb_folder,
        depth_dir=train_depth_folder,
        transform=transformation,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    test_dataset = DepthDataset(
        rgb_dir=test_rgb_folder,
        depth_dir=test_depth_folder,
        transform=transformation
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, test_loader



def save_predictions_as_imgs3x3(
        loader, model, folder ="saved_images3x3/",device="cuda"
):
    model.eval()
    for index, sample in enumerate(loader):
        x = torch.autograd.Variable(sample['image'].cuda())
        y = torch.autograd.Variable(sample['depth'].cuda(non_blocking=True))

        with torch.no_grad():
            preds = torch.sigmoid(model(x))
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{index}.png"
        )
        torchvision.utils.save_image(1000/y,f"{folder}{index}.png")

    model.train()

def save_predictions_as_imgs7x7(
        loader, model, folder ="saved_images7x7/",device="cuda"
):
    model.eval()
    for index, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{index}.png"
        )
        torchvision.utils.save_image(y,f"{folder}{index}.png")

    model.train()

def save_predictions_as_imgs_squeeze(
        loader, model, folder ="saved_images_squeeze/",device="cuda"
):
    model.eval()
    for index, sample in enumerate(loader):
        x = torch.autograd.Variable(sample['image'].cuda())
        y = torch.autograd.Variable(sample['depth'].cuda(non_blocking=True))

        with torch.no_grad():
            preds = torch.sigmoid(model(x))
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{index}.png"
        )
        torchvision.utils.save_image(1000 / y, f"{folder}{index}.png")

    model.train()

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

def gradient(x):
    # tf.image.image_gradients(image)
    h_x = x.size()[-2]
    w_x = x.size()[-1]
    # gradient step=1
    l = x
    r = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    t = x
    b = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    dx, dy = torch.abs(r - l), torch.abs(b - t)
    # dx will always have zeros in the last column, r-l
    # dy will always have zeros in the last row,    b-t
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy
