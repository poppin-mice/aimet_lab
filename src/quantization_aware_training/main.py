from __future__ import division

import argparse

import copy

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torchvision import datasets, transforms

# Quantization related import
from aimet_torch.quantsim import QuantizationSimModel

import sys
sys.path.insert(0,"..")
from model.model import LeNet5

MODEL_PATH = '../../checkpoint/mnist_cnn.pt' # LeNet5 checkpoint

NUM_FINETUNE_CLASSES = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128

def compute_encoding(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return (100. * correct / len(test_loader.dataset))

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return (100. * correct / len(test_loader.dataset))

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset_train = datasets.FashionMNIST('../../data', train=True, download=True,
                       transform=transform)
    indices = torch.randperm(len(dataset_train))[:1000]
    compute_encoding_dataset = Subset(dataset_train, indices)
    dataset_test = datasets.FashionMNIST('../../data', train=False,
                       transform=transform)
    compute_encoding_loader = torch.utils.data.DataLoader(compute_encoding_dataset,**train_kwargs)
    train_loader = torch.utils.data.DataLoader(dataset_train, **test_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model = LeNet5().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    
    acc_before_quant = test(model, test_loader)
    print ("FP32 model accuracy: %f" % (acc_before_quant))

    model_copy = copy.deepcopy(model)
    model_copy.eval()
    sim = QuantizationSimModel(model_copy, default_output_bw=4, default_param_bw=4, dummy_input=torch.rand(1, 1, 28, 28).to(device))
    #print (sim.model)
    sim.compute_encodings(forward_pass_callback=compute_encoding, forward_pass_callback_args=compute_encoding_loader)
    acc_after_quant_train = test(sim.model, test_loader)
    print ("Output bit: %d, params bit: %d, model accuracy: %f" %(4, 4, acc_after_quant_train))

    optimizer = optim.Adadelta(sim.model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, sim.model, device, train_loader, optimizer, epoch)
        test(sim.model, test_loader)
        scheduler.step()

    acc_after_retrain = test(sim.model, test_loader)
    print ("Output bit: %d, params bit: %d, model accuracy after retrain: %f" %(4, 4, acc_after_retrain))

if __name__ == '__main__':
    main()
  