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

def visible_from_hidden(hid, W, v_bias):
    # Enable or disable neurons depending on probabilities
    probability = torch.sigmoid(F.linear(hid, W.t(), v_bias))
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

def ising_averages(mag_history, model_size, label="", n=1):
    mag_matrix = mag_history[:, 0, :]        # get a matrix with just the magnetization, along the columns we have mag of different gibbs sampled states, along the lines differen conc samplings
    mag_gibbs_avg = mag_matrix.mean(axis=0)  # take the mean across gibbs sampled states
    mag_avg = mag_gibbs_avg.mean()               # take the mean across concurrent sampled states
    mag_error = mag_matrix.std(axis=0)[0]
    return mag_avg, mag_error


def gibbs_sampling(steps, model, cs=1):
    """
    Run gibbs sampling 
    """
    v = torch.zeros(cs, model.v_bias.data.shape[0])
    hidden = torch.zeros(cs, model.v_bias.data.shape[0])
    v_prob = v
    magv = []
    magh = []

    # Run the Gibbs sampling for a number of steps
    pbar = tqdm(xrange(steps))

    for s in pbar:
        
        h, h_prob = hidden_from_visible(v, model.W.data, model.h_bias.data)
        v, v_prob = visible_from_hidden(h, model.W.data, model.v_bias.data)
        # Save data
        if (s > parameters['thermalization'] and s % parameters['save_interval']==0):
            magv.append(ising_magnetization(get_ising_variables(v.numpy())))
            magh.append(ising_magnetization(get_ising_variables(h.numpy())))
            hidden = torch.cat((hidden,h),0)  # the produced hidden states are stack in a unique torch tensor
                                              # so that they can be used as input for the next training
    return np.asarray(magv), np.asarray(magh), hidden


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
print("=================Preliminary results =================================")
print("Loading Ising training set...")
train_loader = rbm_pytorch.CSV_Ising_dataset(parameters['ising']['train_data'], size=model_size)

# computing observables for the first training set
m_history = ising_magnetization(get_ising_variables(train_loader[:][0].view(-1, model_size).numpy()))
m = m_history[0, :].mean()
mstd = m_history[0, :].std()
print("Step 0 : m = " + str(m) + ", m.std =" + str(mstd))


for ii in range(parameters['layers']+1):
   # load the RBM to be trained
   train_loader_batch = torch.utils.data.DataLoader(train_loader, shuffle=True, batch_size = parameters['batch'], drop_last=True) # necessary to use batch during training
   rbm = rbm_pytorch.RBM(k = parameters['kCD'], n_vis = model_size, n_hid = hidden_layers) # for the moment don't reduce the number of units

   train_op = optim.SGD(rbm.parameters(), lr=learning_rate,
                     momentum=mom, dampening=damp, weight_decay=wd)
   print("=====================================================================")
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
   magv_history, magh_history, h[ii] = gibbs_sampling(parameters['steps'], rbm)  
   mv, mvstd = ising_averages(magv_history, model_size)
   mh, mhstd = ising_averages(magh_history, model_size)
   print("Results rbm number " + str(ii))
   print("Visible layer : m = " + str(mv) + ", m.std =" + str(mvstd))
   print("Hidden layer : m = " + str(mh) + ", m.std =" + str(mhstd))
   
   # load the second training set  
   print("Loading training set number " + str(ii+1))
   train_loader = rbm_pytorch.np_Ising_dataset(h[ii], size=model_size)
  
  





