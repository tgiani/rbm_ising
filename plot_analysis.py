from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys


# data
epochs3, mag3, mag_err3, en3, en_err3, susc3, susc_err3, cv3, cv_err3, log_likelihood_mean3, ll_up3, ll_down3   = np.loadtxt("./test3_ft_0.001_k5/analysis/analysis_1.8_L8/data.dat",usecols=(0,1,2,3,4,5,6,7,8,9,10,11), unpack=True, skiprows=1)


epochs2, mag2, mag_err2, en2, en_err2, susc2, susc_err2, cv2, cv_err2, log_likelihood_mean2, ll_up2, ll_down2   = np.loadtxt("./test3_ft_0.01_k5/analysis/analysis_1.8_L8/data.dat",usecols=(0,1,2,3,4,5,6,7,8,9,10,11), unpack=True, skiprows=1)


epochs1, mag1, mag_err1, en1, en_err1, susc1, susc_err1, cv1, cv_err1, log_likelihood_mean1, ll_up1, ll_down1   = np.loadtxt("./test3/analysis/analysis_1.8_L8/data.dat",usecols=(0,1,2,3,4,5,6,7,8,9,10,11), unpack=True, skiprows=1)


epochs = np.concatenate((epochs1, epochs2, epochs3), axis=0)
mag = np.concatenate((mag1,mag2,mag3 ), axis=0)
mag_err = np.concatenate((mag_err1,mag_err2,mag_err3), axis=0)
en = np.concatenate((en1, en2, en3), axis=0)
en_err = np.concatenate((en_err1, en_err2, en_err3), axis=0)
susc = np.concatenate((susc1, susc2, susc3), axis=0)
susc_err = np.concatenate((susc_err1, susc_err2,susc_err3), axis=0)
cv = np.concatenate((cv1, cv2, cv3), axis=0)
cv_err = np.concatenate((cv_err1, cv_err2, cv_err3), axis=0)
log_likelihood_mean = np.concatenate((log_likelihood_mean1, log_likelihood_mean2, log_likelihood_mean3), axis=0)
ll_up = np.concatenate((ll_up1, ll_up2, ll_up3), axis=0)
ll_down = np.concatenate((ll_down1, ll_down2, ll_down3), axis=0)



# expected values from magneto
npoints = epochs.size

mag_ = 0.95669*np.ones(npoints)
en_  = -1.85910*np.ones(npoints)
chi_ = 0.1167*np.ones(npoints)  
cv_  = 0.44058*np.ones(npoints)


## Observables vs number of epoch ##

plt.figure(figsize=(15, 5))
plt.scatter(epochs, mag)
plt.errorbar(epochs, mag, yerr = mag_err, elinewidth = 0.8)
plt.plot(epochs, mag_)
plt.ylabel("m", fontsize=18)
plt.xlabel("epoch", fontsize=18)
plt.suptitle('Magnetization vs number of epochs', fontsize=20)
#plt.show()
plt.savefig("./test3/analysis/analysis_1.8_L8/mag_1.8.png")
plt.close()

plt.figure(figsize=(15, 5))
plt.scatter(epochs, en)
plt.errorbar(epochs, en, yerr = en_err, elinewidth = 0.8)
plt.plot(epochs, en_)
plt.ylabel("energy", fontsize=18)
plt.xlabel("epoch", fontsize=18)
plt.suptitle("Energy vs number of epochs", fontsize=20)
#plt.show()
plt.savefig("/home/s1792848/Documents/RBM/rbm_ising/figs/energy_1.8.png")
plt.close()


plt.figure(figsize=(15, 5))
plt.scatter(epochs, susc)
plt.errorbar(epochs, susc, yerr = susc_err, elinewidth = 0.8)
plt.plot(epochs, chi_)
plt.ylabel("chi", fontsize=18)
plt.xlabel("epoch", fontsize=18)
plt.suptitle("Susceptibility vs number of epochs", fontsize=20)
#plt.show()
plt.savefig("/home/s1792848/Documents/RBM/rbm_ising/figs/chi_1.8.png")
plt.close()


plt.figure(figsize=(15, 5))
plt.scatter(epochs, cv)
plt.errorbar(epochs, cv, yerr = cv_err, elinewidth = 0.8)
plt.plot(epochs, cv_)
plt.ylabel("Cv", fontsize=18)
plt.xlabel("epoch", fontsize=18)
plt.suptitle("Heat capacity vs number of epochs", fontsize=20)
#plt.show()
plt.savefig("/home/s1792848/Documents/RBM/rbm_ising/figs/cv_1.8.png")
plt.close()

"""
plt.figure(figsize=(15, 5))
plt.scatter(epochs, log_likelihood_mean)
plt.suptitle("-Log-Likelihood vs number of epochs", fontsize=20)
plt.ylabel("-LL", fontsize=18)
plt.xlabel("epoch", fontsize=18)
plt.savefig("./analysis_1.8_L8/LL_1.8.png")
plt.close()
"""

plt.figure(figsize=(15, 5))
plt.scatter(epochs, log_likelihood_mean)
plt.errorbar(epochs, log_likelihood_mean, yerr = [ll_up, ll_down], elinewidth = 0.8)
plt.suptitle("-Log-Likelihood vs number of epochs", fontsize=20)
plt.ylabel("-LL", fontsize=18)
plt.xlabel("epoch", fontsize=18)
#plt.show()
plt.savefig("/home/s1792848/Documents/RBM/rbm_ising/figs/LL_1.8.png")
plt.close()



