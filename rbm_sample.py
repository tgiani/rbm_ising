###############################################################
#
# Restricted Binary Boltzmann machine in Pytorch
# Possible input sets: MNIST, ISING model configurations
#
#
# 2017 Guido Cossu <gcossu.work@gmail.com>
#
##############################################################


from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys

from tqdm import *

import argparse
import torch
import torch.utils.data
# import torchvision
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

#####################################################
MNIST_SIZE = 784  # 28x28
#####################################################

def bootstrap_resample(X, n=None):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if isinstance(X, pd.Series):
        X = X.copy()
        X.index = range(len(X.index))
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = np.array(X[resample_i])  
    return X_resample


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



###############################  WIP  ##########################################
def ising_matrix():
    N = parameters['ising']['size']
    a = np.zeros(shape=(N*N,N*N))
    for i in range (0,N):
       for j in range (0,N):
          a[N*i+j][N*i+(j+1)%N] -= 1
          a[N*i+j][N*((i+1)%N)+j] -= 1
    return torch.from_numpy(a)



def ising_free_energy(v, ising_matrix, nstates, beta=1.0):
    v = v.double()
    ising_matrix_mult = torch.sum(torch.mul(v,(v.mm(ising_matrix))), 1)
    return ising_matrix_mult.mul(-beta).exp()


##########################################################################




def energy(field):
    N = parameters['ising']['size']
    state = np.asarray(field).reshape((N, N))

    E = np.sum((-state*np.roll(state, -1, axis=0) - state*np.roll(state, -1, axis=1)))/float(N*N)
    #E = 0
    #for i in range (0,N):
    #  for j in range (0,N):
    #    E += -field[N*i+j]*field[N*i+(j+1)%N] - field[N*i+j]*field[N*(i+1)%N+j]
    #  E=E/float(N*N)

 
    return np.asarray([E, E*E])




def energy_concurrent_sampling(field, beta=1.0):
    field = np.asarray(field)
    return(beta*np.apply_along_axis(energy, 1, field).transpose())






def ising_averages(mag_history, en_history, model_size, label=""):

    
    # Bootstrap samples, to use in alternative to concurrent sampling:
    # just one single gibbs sampling is performed, generating a set s = {nsteps states obtained with gibbs sampling}. 
    # From this we get a series of set s_1,..,s_n using by bootstrapping s.
    # We use s_1,..,s_n insted of we analoguos set we would get with concurrent sampling

    n = 500                                                               # number of resampled sets s_i, analogue of the number of concurrent sampled states
    resample_size = parameters['steps']-parameters['thermalization']      # number of states in each resampled set s_i
    
    resampled_states_mag = []
    resampled_states_en = []
     
    ##########   need improvemts  ##############
    for i in range(n):
     resampled_states_mag.append(bootstrap_resample(mag_history[:, 0, :], n=resample_size))
     resampled_states_en.append(bootstrap_resample(en_history[:, 0, :], n=resample_size))
    
    sets_mag = np.asarray(resampled_states_mag)
    mag_avg_resampled = sets_mag.mean(axis=1)  # take the mean of the states in each set s_i
    mag = mag_avg_resampled.mean()     # take the mean across all the resampled sets
    mag_error = mag_avg_resampled.std()    # take the std across all the resampled sets

    sets_en = np.asarray(resampled_states_en)
    en_avg_resampled = sets_en.mean(axis=1)  # take the mean of the states in each set s_i
    en = en_avg_resampled.mean()     # take the mean across all the resampled sets
    en_error = en_avg_resampled.std()    # take the std across all the resampled sets

    print("Bootstrap error, number of bootstrapped resampling = ", n, " each with ", resample_size, " states")
    print(label, " ::: Magnetization: ", mag, " +- ", mag_error)
    print(label, " ::: Energy: ", en, " +- ", en_error)
    
    
    
    
    
    # without bootstrap, using concurrent sampling
    # magnetization
    mag_matrix = mag_history[:, 0, :]        # get a matrix with just the magnetization, along the columns we have mag of different gibbs sampled states, along the lines differen conc samplings
    mag_gibbs_avg = mag_matrix.mean(axis=0)  # take the mean across gibbs sampled states
    mag = mag_gibbs_avg.mean()               # take the mean across concurrent sampled states
    mag_error = mag_gibbs_avg.std()          # take std across concurrent sampled states
    # susceptibility
    susc_gibbs_avg = model_size*(mag_history[:, 1, :].mean(axis=0) - mag_gibbs_avg*mag_gibbs_avg)
    susc = susc_gibbs_avg.mean()             # take mean cross concurrent samplings
    susc_error = susc_gibbs_avg.std()        # take std across concurrent sampled states


    # energy
    en_matrix = en_history[:, 0, :]
    en_gibbs_avg = en_matrix.mean(axis=0)
    en = en_gibbs_avg.mean()
    en_error = en_gibbs_avg.std()
    # heat capacity
    cv_gibbs_avg = model_size*(en_history[:, 1, :].mean(axis=0) - en_gibbs_avg*en_gibbs_avg)
    cv = cv_gibbs_avg.mean()
    cv_error = cv_gibbs_avg.std()

    print("Concurrent sampling, number of concurrent samplings = ", parameters['concurrent samples'], " each with ", resample_size, " states")
    print(label, " ::: Magnetization: ", mag, " +- ", mag_error, " - Susceptibility:", susc, " +- ", susc_error)
    print(label, " ::: Energy: ", en, " +- ", en_error, " - Heat capacity:", cv, " +- ", cv_error)
    

def imgshow(file_name, img):
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % file_name
    Wmin = img.min
    Wmax = img.max
    plt.imsave(f, npimg, vmin=Wmin, vmax=Wmax)


def sample_probability(prob, random):
    """Get samples from a tensor of probabilities.

        :param probs: tensor of probabilities
        :param rand: tensor (of the same shape as probs) of random values
        :return: binary sample of probabilities
    """
    torchReLu = nn.ReLU()
    #Didn't work for Tomasso and I without adding in Variable()
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


def sample_from_rbm(steps, model, image_size, nstates=30, v_in=None):
    """ Samples from the RBM distribution function 

        :param steps: Number of Gibbs sampling steps.
        :type steps: int

        :param model: Trained RBM model.
        :type model: RBM class

        :param image_size: Linear size of output images
        :type image_size: int

        :param nstates: Number of states to generate concurrently
        :type nstates: int

        :param v_in: Initial states (optional)

        :return: Last generated visible state
    """

    if (parameters['initialize_with_training']):
        v = v_in
    else:
        # Initialize with zeroes
        v = torch.zeros(nstates, model.v_bias.data.shape[0])
        # Random initial visible state
        #v = F.relu(torch.sign(torch.rand(nstates,model.v_bias.data.shape[0])-0.5)).data

    v_prob = v
    
    magv = []
    magh = []
    env  = []
    enh  = []
    size = parameters['ising']['size']
   
    # Run the Gibbs sampling for a number of steps
    print("==== Running Gibbs sampling with steps = ", parameters['steps'], " concurrent samplings =", parameters['concurrent samples'], " thermalization =", parameters['thermalization']  )
    # progress bar
    pbar = tqdm(xrange(steps))

    for s in pbar:
        #print(s)
        #r = np.random.random()
        #if (r > 0.5):
        #    vin = torch.zeros(nstates, model.v_bias.data.shape[0])
        #    vin = torch.ones(nstates, model.v_bias.data.shape[0])
        #else:


        if (s % parameters['save interval'] == 0):
            if parameters['output_states']:
                imgshow(parameters['image_dir'] + "dream" + str(s),
                        make_grid(v.view(-1, 1, image_size, image_size)))
        #don't think states should be outputted at all if output_states is false
        #    else:
        #        imgshow(parameters['image_dir'] + "dream" + str(s),
        #                make_grid(v_prob.view(-1, 1, image_size, image_size)))
            if args.verbose:
                print(s, "OK")

        # Update k steps
        #for _ in xrange(200):
        h, h_prob = hidden_from_visible(v, model.W.data, model.h_bias.data)
        v, v_prob = visible_from_hidden(h, model.W.data, model.v_bias.data)
        #vin = v
        
        
            # Save data
        if (s > parameters['thermalization']):
            magv.append(ising_magnetization(get_ising_variables(v.numpy())))
            magh.append(ising_magnetization(get_ising_variables(h.numpy())))
            env.append(energy_concurrent_sampling(get_ising_variables(v.numpy()), size))
            
            

    return v, np.asarray(magv), np.asarray(magh), np.asarray(env)


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

hidden_layers = parameters['hidden_layers'] * parameters['hidden_layers']


# For the MNIST data set
if parameters['model'] == 'mnist':
    model_size = MNIST_SIZE
    image_size = 28
    if parameters['initialize_with_training']:
        print("Loading MNIST training set...")
        train_loader = datasets.MNIST('./DATA/MNIST_data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor()
                                      ]))
#############################
elif parameters['model'] == 'ising':
    # For the Ising Model data set
    image_size = parameters['ising']['size']
    model_size = image_size * image_size
    if parameters['initialize_with_training']:
        print("Loading Ising training set...")
        train_loader = rbm_pytorch.CSV_Ising_dataset(
            parameters['ising']['train_data'], size=model_size)

        # Compute magnetization and susceptibility
        train_mag = []
        for i in xrange(len(train_loader)):
            data = train_loader[i][0].view(-1, model_size)
            train_mag.append(ising_magnetization(get_ising_variables(data.numpy())))
        tr_magarray= np.asarray(train_mag)
        ising_averages(tr_magarray, model_size, "training_set")

# Read the model, example
rbm = rbm_pytorch.RBM(n_vis=model_size, n_hid=hidden_layers)




# load the model, if the file is present
try:
    print("Loading saved RBM network state from file",
          parameters['checkpoint'])
    rbm.load_state_dict(torch.load(parameters['checkpoint']))
except IOError as e:
    print("I/O error({0}): {1}".format(e.errno, e.strerror))
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise

print('Model succesfully loaded')

if parameters['initialize_with_training']:
    data = torch.zeros(parameters['concurrent samples'], model_size)
    for i in xrange(parameters['concurrent samples']):
        data[i] = train_loader[i + 100][0].view(-1, model_size)
    v, magv, magh = sample_from_rbm(parameters['steps'], rbm, image_size, v_in=data)
else:
    v, magv, magh, env = sample_from_rbm(
        parameters['steps'], rbm, image_size, parameters['concurrent samples'])



# test ising matrix. Still not work for multiple states, so it still no good to compute Z.
a = ising_matrix()
print(ising_free_energy(v,a, 1))



# Print statistics
ising_averages(magv, env, model_size, "v")
#ising_averages(magh, enh, model_size, "h")



#logz, logz_up, logz_down = annealed_importance_sampling_ising(k=1, betas = 10000, num_chains = 200)
#print("LogZ ", logz, logz_up, logz_down)



# Save data - in img directory
#since mag history will be N_gibbssample * N_concurrent * 2 we should output mag history for each concurrent sample
#for i in range(len(magv[0, 0, :])):
   # np.savetxt(parameters['image_dir'] + "Mag_history_sample_" + str(i), magv[:, :, i])


"""

Example of JSON input

{
    "model": "ising",
    "checkpoint": "trained_rbm.pytorch.last",
    "image_dir": "./DATA/Dream_Ising/",
    "hidden_layers": 20,
    "steps": 500,
    "save interval": 2,
    "concurrent samples": 30,
    "initialize_with_training": false, 
    "output_states": true,
    "ising": {
        "train_data": "state1.data",
        "size": 32
    }
}


"""
