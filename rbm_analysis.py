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
from rbm_pytorch import log_sum_exp
from rbm_pytorch import log_diff_exp


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


def energy(field):
    N = parameters['ising']['size']
    state = np.array(field).reshape((N, N))
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
    """ 
    Get observables using statistic of more idependent gibbs sampling.
    
    """
    
    resample_size = parameters['steps']-parameters['thermalization']      # number of states in each resampled set s_i
    # using concurrent sampling
    # magnetization
    mag_matrix = mag_history[:, 0, :]        # get a matrix with just the magnetization, along the columns we have mag of different gibbs sampled states, along the lines differen conc samplings
    mag_gibbs_avg = mag_matrix.mean(axis=0)  # take the mean across gibbs sampled states
    mag = mag_gibbs_avg.mean()               # take the mean across concurrent sampled states
    mag_error = mag_gibbs_avg.std()          # take std across concurrent sampled states

    # susceptibility
    susc_gibbs_avg = model_size*(mag_history[:, 1, :].mean(axis=0) - mag_gibbs_avg*mag_gibbs_avg)/parameters['temperature']
    susc = susc_gibbs_avg.mean()             # take mean cross concurrent samplings
    susc_error = susc_gibbs_avg.std()        # take std across concurrent sampled states

    # energy
    en_matrix = en_history[:, 0, :]
    en_gibbs_avg = en_matrix.mean(axis=0)
    en = en_gibbs_avg.mean()
    en_error = en_gibbs_avg.std()

    # heat capacity
    cv_gibbs_avg = model_size*(en_history[:, 1, :].mean(axis=0) - en_gibbs_avg*en_gibbs_avg)/(parameters['temperature']*parameters['temperature'])
    cv = cv_gibbs_avg.mean()
    cv_error = cv_gibbs_avg.std()
    
    return mag, mag_error, en, en_error, susc, susc_error, cv, cv_error


def ising_averages_gs(mag_history, en_history, model_size, label=""):
    """ 
    Get observables using statistic of a single gibbs sampling.
    In order to get the same statistic as metropolis, in the json file use:
    - steps = 2000000
    - save interval = 1000
    - thermalization = 20000
    - concurrent samples = 1

    """
    n_resamplings = parameters['steps']-parameters['thermalization']      # number of resampled states used in bootstrap
    # magnetization
    mag_matrix = mag_history[:, 0, :]        # get a matrix with just the magnetization, along the columns we have mag of different gibbs sampled states, along the lines differen conc samplings
    mag_gibbs_avg = mag_matrix.mean(axis=0)  # take the mean across gibbs sampled states
    mag = mag_gibbs_avg.mean()               # take the mean across concurrent sampled states
    mag_error = mag_matrix.std(axis=0)[0]/sqrt(mag_matrix.size)  # take std of the mean for gibbs sampling

    # susceptibility 
    susc_gibbs_avg = model_size*(mag_history[:, 1, :].mean(axis=0)- mag_gibbs_avg*mag_gibbs_avg)/parameters['temperature']
    susc = susc_gibbs_avg.mean() # take mean cross concurrent samplings 

    # error on susceptibility using bootstrap
    susc_gibbs_avg = np.zeros(n_resamplings)
    for i in range(n_resamplings):
     sample_i = bootstrap_resample(mag_matrix)
     average_m_i = sample_i.mean(axis=0)
     susc_gibbs_avg[i] = model_size*((sample_i**2).mean(axis=0)- average_m_i*average_m_i)/parameters['temperature']
    susc_error = susc_gibbs_avg.std()
    
    # energy
    en_matrix = en_history[:, 0, :]
    en_gibbs_avg = en_matrix.mean(axis=0)
    en = en_gibbs_avg.mean()
    en_error = en_matrix.std(axis=0)[0]/sqrt(en_matrix.size)

    # heat capacity
    cv_gibbs_avg = model_size*(en_history[:, 1, :].mean(axis=0) - en_gibbs_avg*en_gibbs_avg)/(parameters['temperature']*parameters['temperature'])
    cv = cv_gibbs_avg.mean()

    # error on heat capacity using bootstrap
    cv_gibbs_avg = np.zeros(n_resamplings)
    for i in range(n_resamplings):
     sample_i = bootstrap_resample(en_matrix)
     average_en_i = sample_i.mean(axis=0)
     cv_gibbs_avg[i] = model_size*((sample_i**2).mean(axis=0)- average_en_i*average_en_i)/(parameters['temperature']*parameters['temperature'])
    cv_error = cv_gibbs_avg.std()
    cv_b = cv_gibbs_avg.mean()
    
    #print("mag : " + str(mag) + " +- " + str(mag_error) + "; chi : " + str(susc) + " +- " + str(susc_error) )
    #print("energy : " + str(en) + " +- " + str(en_error) + "; cv : " + str(cv) + " +- " + str(cv_error))
    return mag, mag_error, en, en_error, susc, susc_error, cv_error

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
    bar = tqdm(xrange(steps))

    for s in bar:
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
        if (s > parameters['thermalization'] and s % parameters['save interval'] == 0):
            magv.append(ising_magnetization(get_ising_variables(v.numpy())))
            magh.append(ising_magnetization(get_ising_variables(h.numpy())))
            env.append(energy_concurrent_sampling(get_ising_variables(v.numpy())))
            
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


model_size = parameters['ising']['size'] * parameters['ising']['size']
rbm = rbm_pytorch.RBM(n_vis=model_size, n_hid=model_size)


if parameters['do_convergence_analysis']:

  ###################################################################
  #### Convergence analysis: observables/loglikelihood vs epochs ####
  ###################################################################

  print("Convergence analysis L=" + str(parameters['ising']['size']))
  print("Loading Ising training set...")
  train_loader = torch.utils.data.DataLoader(rbm_pytorch.CSV_Ising_dataset(parameters['ising']['train_data'], size=model_size), shuffle=True,
                                               batch_size=parameters['batch_size'], drop_last=True)


  analysis_file = open(parameters['output_dir'] + "analysis_" + str(parameters['temperature']) + "_L" + str(parameters['ising']['size']) + "/data.dat", 'w')
  analysis_file.write("trained rbms from " + str(parameters['checkpoint'])+ "\n")
  pbar = tqdm(range(parameters['start_epoch'], parameters['final_epoch']))
  n = 0

  npoints  = (parameters['final_epoch']-parameters['start_epoch'])/10

  epochs   = np.zeros(npoints)
  mag      = np.zeros(npoints)
  mag_err  = np.zeros(npoints)
  en       = np.zeros(npoints)
  en_err   = np.zeros(npoints)
  susc     = np.zeros(npoints)
  susc_err = np.zeros(npoints)
  cv       = np.zeros(npoints)
  cv_err   = np.zeros(npoints)
  log_likelihood_mean = np.zeros(npoints)


  for epoch in pbar:

    if epoch % 10 == 0:

        print("Loading saved network state from file", parameters['checkpoint'], epoch)
        rbm.load_state_dict(torch.load(parameters['checkpoint']+str(epoch)))
        free_energy_ = []

        # compute free energy averaging in each batch. 
        for i, (data, target) in enumerate(train_loader):
            data_input = Variable(data.view(-1, model_size))
            data_free_energy = rbm.free_energy_batch_mean(data_input)  # note: it does not include Z
            free_energy_.append(data_free_energy.data[0])

        v, magv, magh, env = sample_from_rbm(parameters['steps'], rbm, parameters['ising']['size'], parameters['concurrent samples'])
        mag[n], mag_err[n], en[n], en_err[n], susc[n], susc_err[n], cv[n], cv_err[n] =  ising_averages(magv, env, model_size, "v")
        epochs[n] = epoch

        free_energy_mean = np.mean(free_energy_)   # take the average avross the batches, so that we have the mean across the whole training set
        logz , logz_up, logz_down = rbm.annealed_importance_sampling(1, 10000, 100)
        log_likelihood_mean[n] = free_energy_mean - logz
        
        analysis_file.write(str(epoch) + "\t" +  str(mag[n]) + "\t" + str(mag_err[n])+ "\t" +  str(en[n]) + "\t" + str(en_err[n])+ "\t" +  str(susc[n]) + "\t" + str(susc_err[n])+ "\t" +  str(cv[n]) + "\t" + str(cv_err[n]) + "\t" + str(log_likelihood_mean[n]) + "\n")
        
        print(str(epoch) + "\t" +  str(mag[n]) + "\t" + str(mag_err[n])+ "\t" +  str(en[n]) + "\t" + str(en_err[n])+ "\t" +  str(susc[n]) + "\t" + str(susc_err[n])+ "\t" +  str(cv[n]) + "\t" + str(cv_err[n]) + "\t" + str(log_likelihood_mean[n]) + "\n")
        n+=1

  print("Plotting....")
  ## Observables vs number of epoch ##
  plt.figure(figsize=(15, 5))
  plt.errorbar(epochs, mag, yerr = mag_err)
  plt.title("Magnetization vs number of epochs")
  plt.savefig(parameters['output_dir'] + "analysis_" + str(parameters['temperature']) + "_L" + str(parameters['ising']['size']) + "/mag" + str(parameters['temperature']) + ".png")
  plt.close()
 
  plt.figure(figsize=(15, 5))
  plt.errorbar(epochs, en, yerr = en_err)
  plt.title("Energy vs number of epochs")
  plt.savefig(parameters['output_dir'] + "analysis_" + str(parameters['temperature']) + "_L" + str(parameters['ising']['size']) + "/en" + str(parameters['temperature']) + ".png")
  plt.close()

  plt.figure(figsize=(15, 5))
  plt.errorbar(epochs, susc, yerr = susc_err)
  plt.title("Susceptibility vs number of epochs")
  plt.savefig(parameters['output_dir'] + "analysis_" + str(parameters['temperature']) + "_L" + str(parameters['ising']['size']) + "/susc" + str(parameters['temperature']) + ".png")
  plt.close()

  plt.figure(figsize=(15, 5))
  plt.errorbar(epochs, cv, yerr = cv_err)
  plt.title("Heat capacity vs number of epochs")
  plt.savefig(parameters['output_dir'] + "analysis_" + str(parameters['temperature']) + "_L" + str(parameters['ising']['size']) + "/cv" + str(parameters['temperature']) + ".png")
  plt.close()

  plt.figure(figsize=(15, 5))
  plt.plot(epochs, log_likelihood_mean)
  plt.title("Loglikelihood vs number of epochs")
  plt.savefig(parameters['output_dir'] + "analysis_" + str(parameters['temperature']) + "_L" + str(parameters['ising']['size']) + "/loglikelihood" + str(parameters['temperature']) + ".png")
  plt.close()


  ## Observables vs loglikelihood ##

  plt.figure(figsize=(15, 5))
  plt.errorbar(log_likelihood_mean, mag, yerr = mag_err)
  plt.title("Magnetization vs log_likelihood")
  plt.savefig(parameters['output_dir'] + "analysis_" + str(parameters['temperature']) + "_L" + str(parameters['ising']['size']) + "/mag_ll_" + str(parameters['temperature']) + ".png")
  plt.close()

  plt.figure(figsize=(15, 5))
  plt.errorbar(log_likelihood_mean, en, yerr = en_err)
  plt.title("Energy vs log_likelihood")
  plt.savefig(parameters['output_dir'] + "analysis_" + str(parameters['temperature']) + "_L" + str(parameters['ising']['size']) + "/en_ll_" + str(parameters['temperature']) + ".png")
  plt.close()
 
  plt.figure(figsize=(15, 5))
  plt.errorbar(log_likelihood_mean, susc, yerr = susc_err)
  plt.title("Susceptibility vs log_likelihood")
  plt.savefig(parameters['output_dir'] + "analysis_" + str(parameters['temperature']) + "_L" + str(parameters['ising']['size']) + "/susc_ll_" + str(parameters['temperature']) + ".png")
  plt.close()

  plt.figure(figsize=(15, 5))
  plt.errorbar(log_likelihood_mean, cv, yerr = cv_err)
  plt.title("Heat capacity vs log_likelihood")
  plt.savefig(parameters['output_dir'] + "analysis_" + str(parameters['temperature']) + "_L" + str(parameters['ising']['size']) + "/cv_ll_" + str(parameters['temperature']) + ".png")
  plt.close()

  analysis_file.close()

else:

  ###########################################################
  ##### Temperature analysis: observables vs temperature ####
  ########################################################### 

  print("Analysis as function of temperature")
  analysis_file = open(parameters['output_dir'] + "analysis_T_" + str(parameters['ising']['size']) + "/data_T.dat", 'w')
  analysis_file.write("trained rbms from " + str(parameters['checkpoint'])+ "\n")

  npoints  = int((parameters['final_temperature']- parameters['start_temperature'])*10) + 1

  temperatures   = np.zeros(npoints)
  mag            = np.zeros(npoints)
  mag_err        = np.zeros(npoints)
  en             = np.zeros(npoints)
  en_err         = np.zeros(npoints)
  susc     = np.zeros(npoints)
  susc_err = np.zeros(npoints)
  cv       = np.zeros(npoints)
  cv_err   = np.zeros(npoints)


  for T in range(npoints):
        
        print("Loading saved network state from file", parameters['checkpoint'], T)
        rbm.load_state_dict(torch.load(parameters['checkpoint']+str(T)))
       
        v, magv, magh, env = sample_from_rbm(parameters['steps'], rbm, parameters['ising']['size'], parameters['concurrent samples'])
        mag[T], mag_err[T], en[T], en_err[T], susc[T], susc_err[T], cv[T], cv_err[T] =  ising_averages(magv, env, model_size, "v")
        temperatures[T] = parameters['start_temperature'] + float(T)/10 
       
        # need to rescale susc and cv according to the temperature
        susc = susc*parameters['temperature']/temperatures[T]
        susc_err = susc_err*parameters['temperature']/temperatures[T]
        cv = cv *(parameters['temperature']*parameters['temperature'])/(temperatures[T]*temperatures[T])
        cv_err = cv_err*(parameters['temperature']*parameters['temperature'])/(temperatures[T]*temperatures[T])

        analysis_file.write(str(temperatures[T]) + "\t" +  str(mag[T]) + "\t" + str(mag_err[T])+ "\t" +  str(en[T]) + "\t" + str(en_err[T])+ "\t" +  str(susc[T]) + "\t" + str(susc_err[T])+ "\t" +  str(cv[T]) + "\t" + str(cv_err[T]) + "\n")
        
        print(str(temperatures[T]) + "\t" +  str(mag[T]) + "\t" + str(mag_err[T])+ "\t" +  str(en[T]) + "\t" + str(en_err[T])+ "\t" +  str(susc[T]) + "\t" + str(susc_err[T])+ "\t" +  str(cv[T]) + "\t" + str(cv_err[T]) + "\n")
     

  print("Plotting....")

  ## Observables vs temperature ##
  plt.errorbar(temperatures, mag, yerr = mag_err)
  plt.title("Magnetization vs tempearture")
  plt.savefig(parameters['output_dir'] + "analysis_T_" + str(parameters['ising']['size']) + "/mag_" + str(parameters['ising']['size']) + ".png")
  plt.close()

  plt.errorbar(temperatures, en, yerr = en_err)
  plt.title("Energy vs tempearture")
  plt.savefig(parameters['output_dir'] + "analysis_T_" + str(parameters['ising']['size']) + "/energy_" + str(parameters['ising']['size']) + ".png")
  plt.close()

  plt.errorbar(temperatures, susc, yerr = susc_err)
  plt.title("Susceptibility vs tempearture")
  plt.savefig(parameters['output_dir'] + "analysis_T_" + str(parameters['ising']['size']) + "/susc_" + str(parameters['ising']['size']) + ".png")
  plt.close()

  plt.errorbar(temperatures, cv, yerr = cv_err)
  plt.title("Heat capacity vs tempearture")
  plt.savefig(parameters['output_dir'] + "analysis_T_" + str(parameters['ising']['size']) + "/cv_" + str(parameters['ising']['size']) + ".png")
  plt.close()

  analysis_file.close()



"""
example of input json file:

{

  "do_convergence_analysis": false,
  "checkpoint": "/path/to/the/training/history/or/to/the/full/trained/rbms"
  "model": "ising",
  "temperature" : 3.0,        # tempearture of the convergence analysis
  "start_epoch": 10,          # initial epoch from which analysis starts
  "final_epoch": 1990,        # final epoch
  "batch_size": 200,
  "steps": 10000,             # gibbs sampling steps
  "concurrent samples": 50,
  "save interval" : 1000,
  "output_states": false,
  "hidden_layers": 8,
  "thermalization": 500,
  "image_dir": "./Data_Ising_test/",
  "output_dir": "./Data_analysis/",
  "ising": { "train_data": "/path/to/training/set", 
             "size": 8 },
  "initialize_with_training": false,
  "start_temperature": 1.8,   # start temperature in temperature analysis
  "final_temperature": 3.0    #final temperature 
    

 }

usage:
python rbm_analysis --json file.json

"""






###### [WIP]: Implementation of D_kl ######


def ising_matrix():
    N = parameters['ising']['size']
    a = np.zeros(shape=(N*N,N*N))
    for i in range (0,N):
       for j in range (0,N):
          a[N*i+j][N*i+(j+1)%N] -= 1
          a[N*i+j][N*((i+1)%N)+j] -= 1
    return Variable(torch.from_numpy(a))  # added variable to have it workinh for Z


def ising_free_energy(v, ising_matrix, beta=1.0):
    v = v.double()
    ising_matrix_mult = torch.sum(torch.mul(v,(v.mm(ising_matrix))), 1)   
    return ising_matrix_mult.mul(-beta)  # = -beta*E_ising


def annealed_importance_sampling_ising(H, k = 1, betas = 10000, num_chains = 100):
        """
        Approximates the partition function for the given model using annealed importance sampling.

            .. seealso:: Accurate and Conservative Estimates of MRF Log-likelihood using Reverse Annealing \
               http://arxiv.org/pdf/1412.8566.pdf

        :param num_chains: Number of AIS runs.
        :type num_chains: int

        :param k: Number of Gibbs sampling steps.
        :type k: int

        :param betas: Number or a list of inverse temperatures to sample from.
        :type betas: int, numpy array [num_betas]
        """
        
        # Set betas
        if np.isscalar(betas):
            betas = np.linspace(0.0, 1.0, betas)
        ############## Convergence analysis ####################
        # Start with random distribution beta = 0
        #hzero = Variable(torch.zeros(num_chains, self.n_hid), volatile= True)
        #v = self.visible_from_hidden(hzero, beta= betas[0]);

        v = Variable(torch.sign(torch.rand(num_chains,rbm.n_vis)-0.5), volatile = True)  
        v = F.relu(v)

        # Calculate the unnormalized probabilties of v
        # HERE: need another function that does not average across batches....
        lnpv_sum = -ising_free_energy(v, H, betas[0])  #  denominator

        for beta in betas[1:betas.shape[0] - 1]:
            # Calculate the unnormalized probabilties of v
            lnpv_sum += ising_free_energy(v, H, beta)

           # Sample k times from the intermidate distribution
            for _ in range(0, k):
                h, ph, v, pv = rbm.new_state(v, beta)

            # Calculate the unnormalized probabilties of v
            lnpv_sum -= ising_free_energy(v, H, beta)

        # Calculate the unnormalized probabilties of v
        lnpv_sum += ising_free_energy(v, H, betas[betas.shape[0] - 1])

        lnpv_sum = np.float128(lnpv_sum.data.numpy())
        #print("lnpvsum", lnpv_sum)

        # Calculate an estimate of logz . 
        logz = log_sum_exp(lnpv_sum) - np.log(num_chains)

        # Calculate +/- 3 standard deviations
        lnpvmean = np.mean(lnpv_sum)
        lnpvstd = np.log(np.std(np.exp(lnpv_sum - lnpvmean))) + lnpvmean - np.log(num_chains) / 2.0
        lnpvstd = np.vstack((np.log(3.0) + lnpvstd, logz))
        #print("lnpvstd", lnpvstd)
        #print("lnpvmean", lnpvmean)
        #print("logz", logz)

        # Calculate partition function of base distribution
        baselogz = rbm.log_partition_function_infinite_temperature()

        # Add the base partition function
        logz = logz + baselogz
        logz_up = log_sum_exp(lnpvstd) + baselogz
        logz_down = log_diff_exp(lnpvstd) + baselogz

        return logz , logz_up, logz_down








"""

# test ising matrix
H = ising_matrix()
v = Variable(torch.sign(torch.rand(100,64)-0.5), volatile = True)                # need a variable since H is also a variable (to work properly with Z)
a = ising_free_energy(v,H)






logzis, logz_upis, logz_downis = annealed_importance_sampling_ising(H, k=1, betas = 10000, num_chains = 200)
print("LogZising ", logzis, logz_upis, logz_downis)


"""














