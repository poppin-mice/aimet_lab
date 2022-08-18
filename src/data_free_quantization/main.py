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

# Quantization related import
from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel

from aimet_torch import batch_norm_fold
from aimet_torch import utils

from aimet_torch import bias_correction
from aimet_torch.quantsim import QuantParams

from aimet_torch.cross_layer_equalization import equalize_model, CrossLayerScaling, HighBiasFold

import sys
sys.path.insert(0,"..")
from model.model import LeNet5

# Quantization related import

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

def get_quantized_model(model, output_bw, param_bw, data_loader):
    model_copy = copy.deepcopy(model)
    input_shape = (1, 1, 28, 28)
    dummy_input = torch.randn(input_shape).to(torch.device('cpu'))
    sim = QuantizationSimModel(model_copy, quant_scheme=QuantScheme.post_training_tf_enhanced, default_param_bw=param_bw,
                               default_output_bw=output_bw, dummy_input=dummy_input, in_place=False)
    sim.compute_encodings(forward_pass_callback=test, forward_pass_callback_args=data_loader)
    return (sim.model)

def create_fake_data_loader(dataset_size: int, batch_size: int, image_size=(1, 28, 28)):
    """
    Helper function to create fake data loader which is default image size (1, 28, 28)
    :param dataset_size     : total images in data set
    :param batch_size       : batch size
    :param image_size       : size of input
    :return:
    """
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset = datasets.FashionMNIST('../../data', train=True, download=True,
                       transform=transform)
    indices = torch.randperm(len(dataset))[:dataset_size]
    train_dataset = Subset(dataset, indices)
    data_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=False, num_workers=32)
    
    return data_loader

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

    #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    acc_before_quant = test(model, test_loader)
    print ("FP32 model accuracy: %f" % (acc_before_quant))

    model_copy = copy.deepcopy(model)
    model_copy.eval()

    input_shape = (1, 1, 28, 28)
    folded_pairs = batch_norm_fold.fold_all_batch_norms(model_copy, input_shape)
    print (folded_pairs)
    bn_dict = {}
    for conv_bn in folded_pairs:
        bn_dict[conv_bn[0]] = conv_bn[1]
    print (model_copy)
    _model = get_quantized_model(model_copy, 4, 4, train_loader)
    acc_after_fold = test(_model, test_loader)
    print ("Model accuracy after fold: %f" % (acc_after_fold))

    cls_set_info_list = CrossLayerScaling.scale_model(model_copy, input_shape)
    HighBiasFold.bias_fold(cls_set_info_list, bn_dict)
    _model = get_quantized_model(model_copy, 4, 4, train_loader)
    acc_after_equalized = test(_model, test_loader)
    print ("Model accuracy after equalized: %f" % (acc_after_equalized))

    dataset_size = 200
    batch_size = 64
    data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size, image_size=(1, 28, 28))
    params = QuantParams(weight_bw=4, act_bw=4, round_mode="nearest", quant_scheme=QuantScheme.post_training_tf_enhanced)
    # Perform Bias Correction
    bias_correction.correct_bias(model_copy, params, num_quant_samples=1024,
                                 data_loader=data_loader, num_bias_correct_samples=512)
    _model = get_quantized_model(model_copy, 4, 2, train_loader)
    acc_after_equalized = test(_model, test_loader)
    print ("Model accuracy after bias correction: %f" % (acc_after_equalized))

if __name__ == '__main__':
    main()
  