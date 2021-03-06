#!/usr/bin/env python
# coding: utf-8
import time, os, argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(256, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(dataloader, model, loss_fn, optimizer, device, device_id):
    #size = len(dataloader.dataset)
    num_batches = len(dataloader)
    size = num_batches * dataloader.batch_size

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0 and device_id==0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, dataset_type, best_acc, device, rank):
    #size = len(dataloader.dataset)
    size = 0
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            size += len(X)
    test_loss /= num_batches
    correct /= size
    print(f"{dataset_type} Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    if dataset_type=='Valid' and correct > best_acc and rank==0:
        best_acc = correct
        torch.save(model.state_dict(), "model_best_dist.pth")

    return best_acc


def main(device_id, args):
    # ?????????????????????
    # args.world_size = args.gpus * args.nodes
    # rank = args.node_rank * args.gpus + device_id   # ??????????????????

    # ?????????????????????????????????????????????????????????
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK'))
    dist.init_process_group(
        backend='gloo',
           init_method='env://',
        world_size=world_size,
        rank=rank)

    training_data = datasets.MNIST(args.data_dir, train=True, download=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                           ]))
    test_data  = datasets.MNIST(args.data_dir, train=False, transform=transforms.Compose([
                     transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ]))

    print('train sample num: ', len(training_data))

    num_train = int(len(training_data) * 0.8)
    training_data, validing_data = random_split(training_data, [num_train, len(training_data) - num_train])

    batch_size = 64

    # ???????????? ???????????????????????????????????????????????????????????????
    train_sampler = torch.utils.data.distributed.DistributedSampler(training_data)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(validing_data)

    # ????????????????????? sampler ????????????
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    valid_dataloader = DataLoader(validing_data, batch_size=batch_size, shuffle=False, sampler=valid_sampler)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = NeuralNetwork()
    # ????????????????????????
    model = nn.parallel.DistributedDataParallel(model.to(device))  # cpu???device?????????None

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    start_time = time.time()
    print('training start time: ', time.asctime(time.localtime(start_time)))

    epochs = 3
    best_acc = 0
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device, device_id)
        best_acc = test(valid_dataloader, model, loss_fn, 'Valid', best_acc, device, rank)
    print("Done!")

    end_time = time.time()
    print('training start time: ', time.asctime(time.localtime(end_time)))
    print('training cost time: {:.0f}s'.format(end_time-start_time))

    # ???????????????????????????
    if rank==0:
        model = NeuralNetwork()
        #model.load_state_dict(torch.load("model_best.pth"))
        from collections import OrderedDict
        state_dict = torch.load("model_best_dist.pth")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        model.load_state_dict(new_state_dict)
        _ = test(test_dataloader, model, loss_fn, 'Test', None, device, rank)

if __name__ == '__main__':

    # ???????????????????????????????????????????????????
    parser = argparse.ArgumentParser()
    # ?????????
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    # ???????????????gpu???
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    # ????????????????????????
    parser.add_argument('-nr', '--node_rank', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--data-dir', default='./data'
                    help='location of the training dataset in the local filesystem (will be downloaded if needed)')
    args = parser.parse_args()


    
    # os.environ['MASTER_ADDR'] = '10.76.69.7'
    # os.environ['MASTER_PORT'] = '10008'

    # used in pyotrch Job-operator, ????????????????????????????????????
    

    mp.spawn(main, nprocs=args.gpus, args=(args,))
