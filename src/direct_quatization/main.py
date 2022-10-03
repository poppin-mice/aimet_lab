from __future__ import division

import argparse

import numpy as np
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

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

def forward_pass(model, test_loader):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data) # Pass the data to get the statistical value

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

    return (100. * correct / len(test_loader.dataset))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
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
    train_dataset = Subset(dataset_train, indices)
    dataset_test = datasets.FashionMNIST('../../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)


    model = LeNet5().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    acc_before_quant = test(model, test_loader)
    print ("FP32 model's accuracy: %f \n" % (acc_before_quant))

    model_copy = copy.deepcopy(model)
    model_copy.eval()
    sim = QuantizationSimModel(model_copy, default_output_bw=8, default_param_bw=8, dummy_input=torch.rand(1, 1, 28, 28).to(device))
    sim.compute_encodings(forward_pass_callback=forward_pass, forward_pass_callback_args=train_loader)
    acc_after_quant_train = test(sim.model, test_loader)
    print ("Output bit: %d, params bit: %d, model's accuracy: %f \n" %(8, 8, acc_after_quant_train))

if __name__ == '__main__':
    main()
  