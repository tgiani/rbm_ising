from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys

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
from math import exp, sqrt, tanh, log

import json
from pprint import pprint

import rbm_pytorch
import pandas as pd
from rbm_pytorch import log_sum_exp
from rbm_pytorch import log_diff_exp

model_size = 64
hidden = 64

rbm = rbm_pytorch.RBM(n_vis=model_size, n_hid=hidden)
temperatures = [1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0]
T_ = np.zeros(13)
for i in range(13):
 T_[i] = 1./(2.*temperatures[i])


matrices = [0 for i in range(13)]
matrices1 = [0 for i in range(13)]

T = []
T_err = []
n=0
for i in temperatures:
 rbm.load_state_dict(torch.load("/scratch/RBM/RBM_paper/L8/best_machines/trained_rbm.pytorch.last."+str(i)))

 w = rbm.W.data
 wT = rbm.W.data.t()
 v_bias = rbm.v_bias.data
 h_bias = rbm.h_bias.data

 H = np.zeros((model_size,model_size))
 H1 = np.zeros((model_size,model_size))
 
 for j1 in range(model_size):
  for j2 in range(model_size):
   for i in range(hidden):
     H[j1][j2] += log((1.+ exp(h_bias[i]))*(1.+ exp(h_bias[i] + w[i][j1] + w[i][j2]))/((1.+ exp(h_bias[i]+w[i][j1]))*(1.+ exp(h_bias[i]+w[i][j2]))))
     H1[j1][j2] += log((1.+ exp(v_bias[i]))*(1.+ exp(v_bias[i] + w[j1][i] + w[j2][i]))/((1.+ exp(v_bias[i]+w[j1][i]))*(1.+ exp(v_bias[i]+w[j2][i]))))
    
 H = 1./8.*H 
 a = []
 for j in range(model_size):
  H[j][j] = 0
  H1[j][j] = 0

 for i in range(model_size-8):
  a.append(H[i+8][i])

 T.append(np.asarray(a).mean())
 T_err.append(np.asarray(a).std())  
 matrices[n]=H
 matrices1[n]=H1
 n += 1


row = 2
columns = 7

for i in range(13):
  plt.subplot(row,columns,i+1)
  plt.imshow(matrices1[i])
  plt.title(str(temperatures[i]))
  plt.colorbar()
  #plt.clim(0.1,0.3)

plt.show()
plt.close()


for i in range(13):
  plt.subplot(row,columns,i+1)
  plt.hist(matrices1[i].flatten())
  plt.title(str(temperatures[i]))
  #plt.colorbar()
  #plt.clim(0.1,0.3)

plt.show()
plt.close()


for i in range(13):
  plt.subplot(row,columns,i+1)
  plt.imshow(matrices[i],vmin=0,vmax=0.2)
  plt.title(str(temperatures[i]))
  #plt.colorbar()
  plt.clim(0.1,0.3)

plt.show()
plt.close()

for i in range(13):
  plt.subplot(row,columns,i+1)
  plt.hist(matrices[i].flatten())
  plt.title(str(temperatures[i]))
  #plt.colorbar()
  #plt.clim(0.1,0.3)

plt.show()
plt.close()
 
T = np.asarray(T)
T_err = np.asarray(T_err)

plt.figure(figsize=(15, 5))
plt.scatter(temperatures, T_)
plt.errorbar(temperatures, T, yerr = T_err, elinewidth = 0.8)
#plt.suptitle("-Log-Likelihood vs number of epochs", fontsize=20)
#plt.ylabel("-LL", fontsize=18)
#plt.xlabel("epoch", fontsize=18)
plt.show()
#plt.savefig("/home/s1792848/Documents/RBM/rbm_ising/figs/LL_1.8.png")
plt.close()







"""
H1 = np.zeros((model_size,model_size))
for j1 in range(model_size):
 for j2 in range(model_size):
  H1[j1][j2] = 1./4.*log( (1.+ exp(h_bias[j2]+w[j2][j1])) / (1.+ exp(h_bias[j2])) )

h = np.zeros(model_size)
for j1 in range(model_size):
 for j2 in range(model_size):
  h[j1] += H1[j1][j2]	

plt.plot(h)
plt.show()



H = np.zeros((model_size,model_size))

for j1 in range(model_size):
 for j2 in range(model_size):
  for i in range(hidden):
    H[j1][j2] += log((1.+ exp(v_bias[i]))*(1.+ exp(v_bias[i] + w[j1][i] + w[j2][i]))/((1.+ exp(v_bias[i]+w[j1][i]))*(1.+ exp(v_bias[i]+w[j2][i]))))

H = 1./8.*H

for j in range(model_size):
 H[j][j] = 0

plt.matshow(H,vmin=0,vmax=0.2)
plt.colorbar()
plt.show()
"""

