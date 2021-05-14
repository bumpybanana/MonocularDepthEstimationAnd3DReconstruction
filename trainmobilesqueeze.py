import torch.cuda
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from dataset import DepthDataset
from trainutils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs_squeeze,
    evaluate
)
from mobilesqueeze import Model
from SSIM import ssim
import matplotlib.pyplot as plt

#Hyperparameters
lr = 0.001
#device = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size =16
num_epochs = 25
num_workers = 0 # ~batch_size
pin_memory = True
load_model = False

train_trans = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                  transforms.RandomAffine(4.5, scale=(1, 1.5)),
                                  transforms.RandomCrop(320),
                                  transforms.ToTensor()
                                  ])


test_trans = transforms.Compose([transforms.RandomCrop(320),
                                 transforms.ToTensor()
                                 ])

#train_rgb_folder = "C:/Users/alexk/PycharmProjects/test/data/train_rgb/"
#train_depth_folder = "C:/Users/alexk/PycharmProjects/test/data/train_depth"
#test_rgb_folder = "C:/Users/alexk/PycharmProjects/test/data/test_rgb"
#test_depth_folder = "C:/Users/alexk/PycharmProjects/test/data/test_depth"

#training auf cluster
train_rgb_folder = "/home/ak90gexy/data/train_rgb"
train_depth_folder = "/home/ak90gexy/data/train_depth"
test_rgb_folder = "/home/ak90gexy/data/test_rgb"
test_depth_folder = "/home/ak90gexy/data/test_depth"


train_dataset = DepthDataset(train_rgb_folder, train_depth_folder,transform=train_trans)
test_dataset = DepthDataset(test_rgb_folder, test_depth_folder,transform=test_trans)
trainloader = torch.utils.data.DataLoader(train_dataset,)
trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

def weights_init(m):

    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
   # elif isinstance(m, nn.BatchNorm2d):
   #     m.weight.data.fill_(1)
   #     m.bias.data.zero_()

def main():
    #print("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    model.apply(weights_init)
    l1_loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    epoch = 0
    if load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    for epoch in range(num_epochs):
       #train_fn(trainloader, model, optimizer, loss_fn)
        loop = tqdm(trainloader)
        running_loss = 0

        for batch_index, (data, targets) in enumerate(loop):
            data = data.to(device=device)
            # targets = targets.float().to(device=device)
            targets = targets.to(device=device)

            #normalize depth

            # forward
            predictions = model(data)
            l_depth = l1_loss(predictions, targets)
            l_ssim = torch.clamp((1 - ssim(predictions, targets, val_range=255)) * 0.5, 0, 1)

            loss = (1*l_ssim) + (0.2*l_depth)


            # backward
            optimizer.zero_grad()
            loss.backward() #calculate gradients
            optimizer.step() #update weights

            running_loss += loss.item() * data.size(0)

            # update tqdm loop
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss /len(trainloader)
        losses.append(epoch_loss)
        epoch = epoch +1
        print('Epoch: ', epoch)
        print('Total loss: ',running_loss)
        print('Loss per epoch: ',epoch_loss)

        plt.plot(losses)
        plt.savefig('lossmobile2.png')



        #save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint,"checkpointmobile2.pth.tar")

        #check accuracy
        check_accuracy(testloader, model, device=device)

        #print examples to a folder
        save_predictions_as_imgs_squeeze(
            testloader, model, folder="saved_images_mobilesqueeze2/", device=device
        )

        eval = evaluate(predictions,targets)
        print('Delta1 =', eval[3])
        print('RMSE =', eval[0])





if __name__ == "__main__":
    main()
