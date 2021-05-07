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
)
from squeezeunet import Model
import matplotlib.pyplot as plt

#Hyperparameters
lr = 0.01
device = "cpu"
#device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 5
num_epochs = 150
num_workers = 0 # ~batch_size
pin_memory = True
load_model = False
#transformation = transforms.RandomCrop(320)

#train_rgb_folder = "C:/Users/alexk/PycharmProjects/test/data/train_rgb/"
#train_depth_folder = "C:/Users/alexk/PycharmProjects/test/data/train_depth"
#test_rgb_folder = "C:/Users/alexk/PycharmProjects/test/data/test_rgb"
#test_depth_folder = "C:/Users/alexk/PycharmProjects/test/data/test_depth"

#training auf cluster
train_rgb_folder = "/home/ak90gexy/data/train_rgb"
train_depth_folder = "/home/ak90gexy/data/train_depth"
test_rgb_folder = "/home/ak90gexy/data/test_rgb"
test_depth_folder = "/home/ak90gexy/data/test_depth"

train_dataset = DepthDataset(train_rgb_folder, train_depth_folder)
test_dataset = DepthDataset(test_rgb_folder, test_depth_folder)

trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight)
    #if type(m) == nn.BatchNorm2d:
    #   torch.nn.init.xavier_uniform_(m.weight)

def main():
    #print("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    model.apply(init_weights)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    losses = []

    if load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    for epoch in range(num_epochs):
       #train_fn(trainloader, model, optimizer, loss_fn)
        loop = tqdm(trainloader)
        running_loss = 0
        epoch = 0
        for batch_index, (data, targets) in enumerate(loop):
            data = data.to(device=device)
            # targets = targets.float().to(device=device)
            targets = targets.to(device=device)

            # forward
            predictions = model(data)
            loss = loss_fn(predictions, targets)

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



        #save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        #check accuracy
        check_accuracy(testloader, model, device=device)

        #print examples to a folder
        save_predictions_as_imgs_squeeze(
            testloader, model, folder="saved_images_squeeze/", device=device
        )





if __name__ == "__main__":
    main()
