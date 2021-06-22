import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils
from torchvision import transforms
from tensorboardX import SummaryWriter
from utilsmobile import AverageMeter, colorize, save_predictions_dataset, load_checkpoint, save_checkpoint, evaluate
from mobilesqueeze import Model
from SSIM import ssim
from dataset import DepthDataset
import kornia.filters as kf
import argparse



train_rgb_folder = "/home/ak90gexy/data/train_rgb"
train_depth_folder = "/home/ak90gexy/data/train_depth"
test_rgb_folder = "/home/ak90gexy/data/test_rgb"
test_depth_folder = "/home/ak90gexy/data/test_depth"

train_trans = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                  transforms.RandomAffine(4.5, scale=(1, 1.2)),
                                  transforms.RandomCrop([480, 480]),
                                  transforms.Resize([320, 320]),
                                  transforms.ToTensor()
                                  ])
train_trans_rgb = transforms.Compose([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
                                      ])

test_trans = transforms.Compose([transforms.Resize([320, 320]),
                                 transforms.ToTensor()
                                 ])

# Load data
train_dataset = DepthDataset(train_rgb_folder, train_depth_folder, transform=train_trans,transform_rgb=train_trans_rgb)
test_dataset = DepthDataset(test_rgb_folder, test_depth_folder, transform=test_trans)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, )
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)



#initialize weights following Kaiming
def weights_init(m):

    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='mobilesqueeze')
    parser.add_argument('--epochs', default=450, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=8, type=int, help='batch size')
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create model
    model = Model().cuda()
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.decoder.apply(weights_init)
    print('Model created.')

    # Training parameters
    optimizer = torch.optim.Adam(model.decoder.parameters(), args.lr)
    batch_size = args.bs
    prefix = 'densenet_' + str(batch_size)



    # Logging
    writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)
    writer_val = SummaryWriter(comment='val', flush_secs=30)
    # Load model
    load_model = False
    if load_model:
        load_checkpoint(torch.load("200epochs.pth.tar"), model)
        model.eval()
    # Loss
    l1_criterion = nn.L1Loss()

    # Start training...
    for epoch in range(args.epochs):
        train_losses = AverageMeter()
        train_l1 = AverageMeter()
        train_SSIM = AverageMeter()
        train_grad = AverageMeter()
        test_losses = AverageMeter()
        test_l1 = AverageMeter()
        test_SSIM = AverageMeter()
        test_grad = AverageMeter()

        N = len(train_loader)

        # Switch to train mode
        model.train()

        for i, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()

            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched[0].cuda())
            depth_n = torch.autograd.Variable(sample_batched[1].cuda(non_blocking=True))
            
            # Predict
            output = model(image)

            # Compute the losses
            dx_true, dy_true = 6*kf.SpatialGradient()(depth_n)[:,:,0,:,:], 6*kf.SpatialGradient()(depth_n)[:,:,1,:,:]
            dx_pred, dy_pred = 6*kf.SpatialGradient()(output)[:,:,0,:,:], 6*kf.SpatialGradient()(output)[:,:,1,:,:]
            l_grad = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true))
            l_depth = l1_criterion(output,depth_n)
            l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range=1/10)) * 0.5, 0, 1)

            loss = (1 * l_ssim) + (0.3 * l_depth) + (1 *l_grad)

            # Update step
            train_losses.update(loss.data.item(), image.size(0))
            train_l1.update(0.1 * l_depth.data.item(), image.size(0))
            train_SSIM.update(l_ssim.data.item(), image.size(0))
            train_grad.update(l_grad.data.item(), image.size(0))
            loss.backward()
            optimizer.step()

            # Log progress
            niter = epoch * N + i
            if i % 10 == 0:
                # Print to console
                #print('Epoch: [{0}][{1}/{2}]\t'
                #      'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                #      'ETA {eta}\t'
                #      'Loss {loss.val:.4f} ({loss.avg:.4f})'
                #      .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))

                # Log to tensorboard
                writer.add_scalar('Train/Loss', train_losses.val, niter)

            if i % 300 == 0:
                LogProgress(model, writer, train_loader, niter)

            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if epoch % 25 == 0:
                save_checkpoint(checkpoint, '{0}epochs0.3ldepth.pth.tar'.format(epoch))

        model.eval()
        
        #compute  and log validation losses
        with torch.no_grad():
            for i, sampled_batch in enumerate(test_loader):
                val_image = torch.autograd.Variable(sampled_batch[0].cuda())
                val_depth = torch.autograd.Variable(sampled_batch[1].cuda(non_blocking=True))
                val_output = model(val_image)

                # Compute the loss
                dx_true, dy_true = 6 * kf.SpatialGradient()(val_depth)[:, :, 0, :, :], 6 * kf.SpatialGradient()(val_depth)[
                                                                                         :, :, 1, :, :]
                dx_pred, dy_pred = 6 * kf.SpatialGradient()(val_output)[:, :, 0, :, :], 6 * kf.SpatialGradient()(val_output)[:,
                                                                                        :, 1, :, :]
                l_grad = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true))
                l_depth = l1_criterion(val_output, val_depth)
                l_ssim = torch.clamp((1 - ssim(val_output, val_depth, val_range=1 / 10)) * 0.5, 0, 1)

                val_loss = (1 * l_ssim) + (0.1 * l_depth) + (1 * l_grad)
                test_losses.update(val_loss.data.item(), val_image.size(0))
                test_l1.update(0.1*l_depth.data.item(), val_image.size(0))
                test_SSIM.update(l_ssim.data.item(), val_image.size(0))
                test_grad.update(l_grad.data.item(), val_image.size(0))
                if i % 10 == 0:
                    writer_val.add_scalar('Train/Loss', test_losses.val, niter)
                if i % 300 == 0:
                    LogProgress(model, writer_val, test_loader, niter)

        # print examples to a folder
        save_predictions_dataset(
            test_loader, model, folder="trainfinal0.3ldepth/", device=device
        )
        # Record epoch's intermediate results
        LogProgress(model, writer, train_loader, niter)
        LogProgress(model, writer_val,test_loader,niter)
        writer.add_scalar('Train/Loss.avg', train_losses.avg, epoch)
        writer.add_scalar('Train/LossL1.avg', train_l1.avg, epoch)
        writer.add_scalar('Train/LossSSIM.avg', train_SSIM.avg, epoch)
        writer.add_scalar('Train/LossGrad.avg', train_grad.avg, epoch)
        writer_val.add_scalar('Train/Loss.avg',test_losses.avg, epoch)
        writer_val.add_scalar('Train/LossL1.avg', test_l1.avg, epoch)
        writer_val.add_scalar('Train/LossSSIM.avg', test_SSIM.avg, epoch)
        writer_val.add_scalar('Train/LossGrad.avg', test_grad.avg, epoch)


def LogProgress(model, writer, loader, epoch):
    model.eval()
    sequential = loader
    sample_batched = next(iter(sequential))
    image = torch.autograd.Variable(sample_batched[0].cuda())
    depth = torch.autograd.Variable(sample_batched[1].cuda(non_blocking=True))
    if epoch == 0: writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=4, normalize=True), epoch)
    if epoch == 0: writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth.data, nrow=4, normalize=False)),
                                    epoch)
    output = model(image)
    evaluation = evaluate(output, depth)
    writer.add_image('Train.3.Ours', colorize(vutils.make_grid(output.data, nrow=4, normalize=False)), epoch)
    writer.add_image('Train.3.Diff',
                     colorize(vutils.make_grid(torch.abs(output - depth).data, nrow=4, normalize=False)), epoch)
    writer.add_scalar('RMSE', evaluation[0], epoch)
    writer.add_scalar('AbsRel', evaluation[2], epoch)
    writer.add_scalar('delta1', evaluation[3], epoch)
    writer.add_scalar('delta2', evaluation[4], epoch)
    writer.add_scalar('delta3', evaluation[5], epoch)
    del image
    del depth
    del output


if __name__ == '__main__':
    main()
