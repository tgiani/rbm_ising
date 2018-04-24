"""
here I try to generate a series of hidden states
given a set of initial visible (training set) and a
trained rbm

"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys

from tqdm import *

import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset

from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from math import exp, sqrt

import json
from pprint import pprint

import rbm_pytorch
import pandas as pd


def sample_probability(prob, random):
    """Get samples from a tensor of probabilities.
        :param probs: tensor of probabilities
        :param rand: tensor (of the same shape as probs) of random values
        :return: binary sample of probabilities
    """
    torchReLu = nn.ReLU()
    return torchReLu(Variable(torch.sign(prob - random))).data

def hidden_from_visible(visible, W, h_bias):
    # Enable or disable neurons depending on probabilities
    probability = torch.sigmoid(F.linear(visible, W, h_bias))
    random_field = torch.rand(probability.size())
    new_states = sample_probability(probability, random_field)
    return new_states, probability

def get_ising_variables(field, sign=-1):
    """ Get the Ising variables {-1,1} representation
    of the RBM Markov fields
    :param field: the RBM state (visible or hidden), numpy
    :param sign: sign of the conversion 
    :return: the Ising field
    """
    sign_field = np.full(field.shape, sign)

    return (2.0 * field + sign_field).astype(int)

def ising_magnetization(field):
    #axis=1 to return the average field for each state dimension N_concsamp x 1 
    m = np.abs((field).mean(axis=1))
    return np.array([m, m * m])

def ising_averages(mag_history, model_size, label=""):
    # magnetization
    mag_vector = mag_history[0, :]    # get a vector with the magnetization of each state
    mag_avg = mag_vector.mean()       # take the mean 
    mag_error = mag_vector.std()      # take the std
    return mag_avg, mag_error


# Parse command line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--json', dest='input_json', default='params.json', help='JSON file describing the sample parameters',
                    type=str)
parser.add_argument('--verbose', dest='verbose', default=False, help='Verbosity control',
                    type=bool, choices=[False, True])

args = parser.parse_args()
try:
    parameters = json.load(open(args.input_json))
except IOError as e:
    print("I/O error({0}): {1}".format(e.errno, e.strerror))
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise

if args.verbose:
    print(args)
    pprint(parameters)


# read RBM structure parameters
# for now they will be the same for every layer
hidden_layers = parameters['hidden_layers'] * parameters['hidden_layers']
image_size = parameters['ising']['size']
model_size = image_size * image_size
n = parameters['ising']['train_dimension']
h = [0 for i in range(parameters['layers']+1)]

# Training parameters: same for every training
learning_rate = 0.01
mom = 0.0   # momentum
damp = 0.0  # dampening factor
wd = 0.0    # weight decay 


# load the first training set coming from magneto
print("Loading Ising training set...")
train_loader = rbm_pytorch.CSV_Ising_dataset(parameters['ising']['train_data'], size=model_size)

# computing observables for the first training set
train_data = torch.zeros(n, model_size)
for i in range(n):
  train_data[i] = train_loader[i][0].view(-1, model_size)

m_history = ising_magnetization(get_ising_variables(train_data.numpy()))
m, mstd = ising_averages(m_history, model_size)
print("Step 0 : m = " + str(m) + ", m.std =" + str(mstd))


for ii in range(parameters['layers']+1):
   # load the RBM to be trained
   train_loader_batch = torch.utils.data.DataLoader(train_loader, shuffle=True, batch_size = parameters['batch'], drop_last=True) # necessary to use batch during training
   rbm = rbm_pytorch.RBM(k = parameters['kCD'], n_vis = model_size, n_hid = hidden_layers) # for the moment don't reduce the number of units

   train_op = optim.SGD(rbm.parameters(), lr=learning_rate,
                     momentum=mom, dampening=damp, weight_decay=wd)

   print("Starting training rbm number " + str(ii))
   pbar = tqdm(range(parameters['start_epoch'], parameters['epochs']))

   # Run the RBM training
   for epoch in pbar:
       loss_ = []
       full_reconstruction_error = []
       free_energy_ = []

       for i, (data, target) in enumerate(train_loader_batch):
           data_input = Variable(data.view(-1, model_size))
           # how to randomize?
           new_visible, hidden, h_prob, v_prob = rbm(data_input)

           # loss function: see Fisher eq 28 (Training RBM: an Introduction)
           # the average on the training set of the gradients is
           # the sum of the derivative averaged over the training set minus the average on the model
           # still possible instabilities here, so I am computing the gradient myself
           data_free_energy = rbm.free_energy_batch_mean(data_input)  # note: it does not include Z
           loss = data_free_energy - rbm.free_energy_batch_mean(new_visible)
           loss_.append(loss.data[0])
           free_energy_.append(data_free_energy.data[0])

           reconstruction_error = rbm.loss(data_input, new_visible)
           full_reconstruction_error.append(reconstruction_error.data[0])

           # Update gradients
           train_op.zero_grad()
           # manually update the gradients, do not use autograd
           rbm.backward(data_input, new_visible)
           train_op.step()
    
   # Save the final model
   torch.save(rbm.state_dict(), "trained_rbm.pytorch." + str(ii))


   # now load the trained rbm and generate a new training using p(h|v)
   print("Loading rbm number " + str(ii))
   rbm = rbm_pytorch.RBM(n_vis=model_size, n_hid=hidden_layers)
   rbm.load_state_dict(torch.load("trained_rbm.pytorch." + str(ii)))

   print("Generating new training set according to p(h|v)..")
   h[ii], h_prob = hidden_from_visible(train_data, rbm.W.data, rbm.h_bias.data)

   # load the second training set coming 
   print("Loading training set number " + str(ii+1))
   train_loader = rbm_pytorch.np_Ising_dataset(h[ii], size=model_size)

   # computing observables for the second training set
   train_data = torch.zeros(n, model_size)
   for j in range(n):
     train_data[j] = train_loader[j][0].view(-1, model_size)

   m_history = ising_magnetization(get_ising_variables(train_data.numpy()))
   m, mstd = ising_averages(m_history, model_size)
   print("Layer " + str(ii+1) + " : m = " + str(m) + ", m.std =" + str(mstd))








