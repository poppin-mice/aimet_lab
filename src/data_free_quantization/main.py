from __future__ import division

import argparse

import numpy as np
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from torchvision import datasets, transforms

# Quantization related import
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel

from aimet_torch import batch_norm_fold

from aimet_torch import bias_correction
from aimet_torch.quantsim import QuantParams

from aimet_torch.cross_layer_equalization import CrossLayerScaling, HighBiasFold

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
    indices = torch.randperm(len(dataset_train))[:100]
    train_dataset = Subset(dataset_train, indices)
    dataset_test = datasets.FashionMNIST('../../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model = LeNet5().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))

    acc_before_quant = test(model, test_loader)
    print ("FP32 model's accuracy: %f" % (acc_before_quant))

    model_copy = copy.deepcopy(model)
    model_copy.eval()

    input_shape = (1, 1, 28, 28)
    folded_pairs = batch_norm_fold.fold_all_batch_norms(model_copy, input_shape)
    bn_dict = {}
    for conv_bn in folded_pairs:
        bn_dict[conv_bn[0]] = conv_bn[1]
    _model = get_quantized_model(model_copy, 6, 4, train_loader)
    acc_after_fold = test(_model, test_loader)
    print ("Model's accuracy after fold: %f" % (acc_after_fold))

    cls_set_info_list = CrossLayerScaling.scale_model(model_copy, input_shape)
    HighBiasFold.bias_fold(cls_set_info_list, bn_dict)
    _model = get_quantized_model(model_copy, 6, 4, train_loader)
    acc_after_equalized = test(_model, test_loader)
    print ("Model's accuracy after equalized: %f" % (acc_after_equalized))

    dataset_size = 200
    batch_size = 64
    data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size, image_size=(1, 28, 28))
    params = QuantParams(weight_bw=4, act_bw=6, round_mode="nearest", quant_scheme=QuantScheme.post_training_tf_enhanced)
    # Perform Bias Correction
    bias_correction.correct_bias(model_copy, params, num_quant_samples=1024,
                                 data_loader=data_loader, num_bias_correct_samples=512)
    _model = get_quantized_model(model_copy, 6, 4, train_loader)
    acc_after_equalized = test(_model, test_loader)
    print ("Model's accuracy after bias correction: %f" % (acc_after_equalized))

if __name__ == '__main__':
    main()
  