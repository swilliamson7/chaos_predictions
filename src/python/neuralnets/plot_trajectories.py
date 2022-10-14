import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace
from data import run_pendulum 

datadir="/Users/mattgoldberg/Projects/CSEM/research/CRIOS/chaos_predictions/src/python/neuralnets/checkpoints/matt_test/h2_h2/"


predictions=np.load(datadir + "predictions.npy")
targets=np.load(datadir + "targets.npy")
train_omega=np.load(datadir + "train_omega.npy")
train_theta=np.load(datadir + "train_theta.npy")

tmax=10
nt=100
# wInitMu, wInitSigma = 0, 0.5
# thetaInitMu, thetaInitSigma = 0, 1
betaMu, betaSigma = 3*np.pi/4, 0.1
gammaMu, gammaSigma = 1.5, 0.03 # note: different values of gamma will lead to different chaotic regimes
wDMu, wDSigma = 2*np.pi, 0.1
w0Mu, w0Sigma = 3*np.pi, 0.1


uniq_pred = np.unique(predictions, axis=0)
nShow=len(uniq_pred)
trajectories=np.empty([len(uniq_pred), nt])
trajectories=np.empty([nShow, nt])
displaynames=[]

for idx, IC in enumerate(uniq_pred):
    if idx > nShow-1:
        continue
    omega0, theta0 = IC
    _, theta =run_pendulum(tmax, nt, betaMu, gammaMu, wDMu, w0Mu, omega0, theta0)
    trajectories[idx,:] = theta
    displaynames.append("({:.7g},{:.7g})".format(omega0, theta0))

plt.clf()
traj_plot = plt.plot(range(nt),trajectories.T)
plt.legend(iter(traj_plot), displaynames)
plt.show()

set_trace()

#plt.clf(); traj_plot = plt.plot(range(nt),trajectories.T); plt.legend(iter(traj_plot), displaynames); plt.show()
