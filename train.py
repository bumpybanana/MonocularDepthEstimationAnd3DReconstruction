import torch.cuda
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from dataset import DepthDataset
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
from NN7conv import DepthPredictionNet

#Hyperparameters


lr = 0.01
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 5
num_epochs = 100
num_workers = 0 # ~batch_size
img_height = 320
img_width = 320
pin_memory = True
load_model = False

train_rgb_folder =  "E:/nyuv2/train_rgb/train_rgb"
train_depth_folder = "E:/nyuv2/train_depth"
test_rgb_folder = "E:/nyuv2/test_rgb"
test_depth_folder = "E:/nyuv2/test_depth"

def train_fn(loader,model,optimizer,loss_fn,scaler):
    loop = tqdm(loader)

    for batch_index,(data,targets) in enumerate(loop):
        data = data.to(device=device)
        #targets = targets.float().unsqueeze(1).to(device=device)
        targets = targets.to(device=device)


        #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)


        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    transformation = transforms.Compose([transforms.RandomCrop(320),
                                         transforms.RandomHorizontalFlip(p=0.5),
                                         transforms.RandomVerticalFlip(p=0.5),
                                         transforms.RandomAffine(4.5,scale=(1, 1.5)),
                                       # transforms.ColorJitter()
                                         transforms.ToTensor()])

    model = DepthPredictionNet().to(device)
    loss_fn = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=1e-4)

    train_loader, test_loader = get_loaders(
        train_rgb_folder,
        train_depth_folder,
        test_rgb_folder,
        test_depth_folder,
        batch_size,
        transformation,
        num_workers,
        pin_memory
    )
    if load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"),model)
    #check_accuracy(test_loader,model,device=device)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        train_fn(train_loader,model,optimizer,loss_fn,scaler)

        #save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        #check accuracy
        #check_accuracy(test_loader,model,device=device)

        #print examples to a folder
        save_predictions_as_imgs(
            test_loader,model,folder="saved_images/",device=device
        )


if __name__ == "__main__":
    main()