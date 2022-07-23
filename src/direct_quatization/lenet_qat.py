from __future__ import division

# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#  
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without 
#  modification, are permitted provided that the following conditions are met:
#  
#  1. Redistributions of source code must retain the above copyright notice, 
#     this list of conditions and the following disclaimer.
#  
#  2. Redistributions in binary form must reproduce the above copyright notice, 
#     this list of conditions and the following disclaimer in the documentation 
#     and/or other materials provided with the distribution.
#  
#  3. Neither the name of the copyright holder nor the names of its contributors 
#     may be used to endorse or promote products derived from this software 
#     without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#  
#  SPDX-License-Identifier: BSD-3-Clause
#  
#  @@-COPYRIGHT-END-@@
# =============================================================================
# pylint: disable=missing-docstring
""" These are code examples to be used when generating AIMET documentation via Sphinx """

import numpy as np

import argparse

import PIL
import numpy as np
from tqdm import tqdm
import copy

import torchvision

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from torchvision import datasets, transforms


from model import LeNet5

# Quantization related import
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.examples import mnist_torch_model


from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

MODEL_PATH = '../../checkpoint/mnist_cnn.pt' # LeNet5 checkpoint

NUM_FINETUNE_CLASSES = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128

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

def evaluate_model(model: torch.nn.Module, eval_iterations: int, use_cuda: bool = True) -> float:
  """
  This is intended to be the user-defined model evaluation function.
  AIMET requires the above signature. So if the user's eval function does not
  match this signature, please create a simple wrapper.

  Note: Honoring the number of iterations is not absolutely necessary.
  However if all evaluations run over an entire epoch of validation data,
  the runtime for AIMET compression will obviously be higher.

  :param model: Model to evaluate
  :param eval_iterations: Number of iterations to use for evaluation.
          None for entire epoch.
  :param use_cuda: If true, evaluate using gpu acceleration
  :return: single float number (accuracy) representing model's performance
  """
  if (eval_iterations is not None):
    target_sample_number = eval_iterations * batch_size
    num_smaple_data = min(target_sample_number, len(testset))
  else: 
    num_smaple_data = len(testset)
  
  subdataset = torch.utils.data.Subset(testset, range(len(testset))[:num_smaple_data])

  subtestloader = torch.utils.data.DataLoader(subdataset, batch_size=batch_size,
                                      shuffle=False, num_workers=32)

  if (use_cuda):
    model.to(device)
  
  correct = 0
  total = 0
  # since we're not training, we don't need to calculate the gradients for our outputs
  with torch.no_grad():
    for i, data in tqdm(enumerate(subtestloader, 0), total=len(subtestloader)):
      if (use_cuda):
        images, labels = data[0].to(device), data[1].to(device)
      else:
        images, labels = data
      # calculate outputs by running images through the network 
      outputs = model(images)
      # the class with the highest energy is what we choose as prediction
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  return (correct / total)

def quantize_model(model, output_bw: int, param_bw: int):
  acc_before_quant = test(model)
  print ("FP32 model accuracy: %f" % (acc_before_quant))

  model_copy = copy.deepcopy(model)
  sim = QuantizationSimModel(model_copy, default_output_bw=output_bw, default_param_bw=param_bw, dummy_input=torch.rand(1, 3, 32, 32).to(device))
  sim.compute_encodings(forward_pass_callback=evaluate_model, forward_pass_callback_args=5)
  acc_after_quant_train = evaluate_model(sim.model, 20, True)
  print ("Without model.eval(), Output bit: %d, params bit: %d, model accuracy: %f" %(output_bw, param_bw, acc_after_quant_train))
  
  model_copy = copy.deepcopy(model)
  model_copy.eval()
  sim = QuantizationSimModel(model_copy, default_output_bw=output_bw, default_param_bw=param_bw, dummy_input=torch.rand(1, 3, 224, 224).to(device))
  sim.compute_encodings(forward_pass_callback=evaluate_model, forward_pass_callback_args=5)
  acc_after_quant_eval = evaluate_model(sim.model, 20, True)
  print ("With model.eval(), Output bit: %d, params bit: %d, model accuracy: %f" %(output_bw, param_bw, acc_after_quant_eval))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
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
    dataset1 = datasets.FashionMNIST('../../data', train=True, download=True,
                       transform=transform)
    indices = torch.randperm(len(dataset1))[:100]
    train_dataset = Subset(dataset1, indices)
    dataset2 = datasets.FashionMNIST('../../data', train=False,
                       transform=transform)
    print (len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


    model = LeNet5().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    #model = mnist_torch_model.Net().to(device)
    #model.load_state_dict(torch.load(MODEL_PATH))

    #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    acc_before_quant = test(model, test_loader)
    print ("FP32 model accuracy: %f" % (acc_before_quant))

    model_copy = copy.deepcopy(model)
    model_copy.eval()
    sim = QuantizationSimModel(model_copy, default_output_bw=4, default_param_bw=2, dummy_input=torch.rand(1, 1, 28, 28).to(device))
    #print (sim.model)
    sim.compute_encodings(forward_pass_callback=test, forward_pass_callback_args=train_loader)
    acc_after_quant_train = test(sim.model, test_loader)
    print ("Output bit: %d, params bit: %d, model accuracy: %f" %(2, 4, acc_after_quant_train))

    
    #for epoch in range(1, args.epochs + 1):
    #    train(args, model, device, train_loader, optimizer, epoch)
    #    test(model, device, test_loader)
    #    scheduler.step()

    #if args.save_model:
    #    torch.save(model.state_dict(), "../../checkpoint/mnist_cnn.pt")


if __name__ == '__main__':
    main()
  