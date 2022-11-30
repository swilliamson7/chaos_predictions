"""
Generate time series data
"""
import numpy as np
import torch
import torch.utils.data as data_utils
from scipy.integrate import odeint


def pendulum(stateIn, t, beta, gamma, wD, w0):
        """
        pendulum state equation
        """
        w, theta = stateIn
        stateOut = np.array([-2*beta*w - w0**2*np.sin(theta) + gamma*w0**2*np.cos(wD*t), w])
        return stateOut

def run_pendulum(tmax=10, nt=500, beta=3*np.pi/4, gamma=1.01, wD=2*np.pi, w0=3*np.pi, wInit=0, thetaInit=0):
    t=np.linspace(0,tmax,nt)
    f = odeint(pendulum, (wInit, thetaInit), t, args=(beta, gamma, wD, w0))
    _, theta=f.T
    return t,theta


def generate_dataset(nData=3000, tmax=10, nt=500):
    # create randomly perturbed initial conditions wInit, thetaInit
    wInitMu, wInitSigma = 0, 0.01
    wInitRnd=np.random.normal(wInitMu, wInitSigma, nData)
    thetaInitMu, thetaInitSigma = 0, 0.01
    thetaInitRnd=np.random.normal(thetaInitMu, thetaInitSigma, nData)

    # create randomly perturbed values for parameters beta, gamma, wD, w0
    betaMu, betaSigma = 3*np.pi/4, 0.1
    betaRnd=np.random.normal(betaMu, betaSigma, nData)
    gammaMu, gammaSigma = 1.01, 0.03
    gammaRnd=np.random.normal(gammaMu, gammaSigma, nData)
    wDMu, wDSigma = 2*np.pi, 0.1
    wDRnd=np.random.normal(wDMu, wDSigma, nData)
    w0Mu, w0Sigma = 3*np.pi, 0.1
    w0Rnd=np.random.normal(w0Mu, w0Sigma, nData)


    # create dataset with nData "points" (each point is a time series of length nt)
    dataset=np.empty([nData, nt])
    params=np.empty([nData, 6])

    for i in range(nData):
        params[i]=np.array([betaRnd[i], gammaRnd[i], wDRnd[i], w0Rnd[i], wInitRnd[i], thetaInitRnd[i]])
        _, theta=run_pendulum(tmax, nt, betaRnd[i], gammaRnd[i], wDRnd[i], w0Rnd[i], wInitRnd[i], thetaInitRnd[i])
        dataset[i]=theta
    return params, dataset


def split_dataset(inputs, outputs, n_train=800, n_val=1000, n_test=1200):
    X_train = inputs[:n_train]
    X_val = inputs[n_train:n_val]
    X_test = inputs[n_val:n_test]

    y_train = outputs[:n_train]
    y_val = outputs[n_train:n_val]
    y_test = outputs[n_val:n_test]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_dataset():
    params, dataset = generate_dataset()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(params, dataset)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def create_pytorch_loader(X, y, batch_size=100, shuffle=True):
    params = torch.from_numpy(X.astype('float32'))
    data = torch.from_numpy(y.astype('float32'))

    dataset = data_utils.TensorDataset(params, data)
    dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def create_pytorch_data(batch_size=100):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_dataset()
    train_loader = create_pytorch_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = create_pytorch_loader(X_val, y_val, batch_size=batch_size*2, shuffle=False)
    test_loader = create_pytorch_loader(X_test, y_test, batch_size=batch_size*2, shuffle=False)

    return train_loader, val_loader, test_loader



def create_pytorch_loader(X, y, batch_size=100, shuffle=True):
    patterns = torch.from_numpy(X.astype('float32'))
    targets = torch.from_numpy(y.astype('float32'))

    dataset = data_utils.TensorDataset(patterns, targets)
    dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def create_pytorch_data(batch_size=100):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_dataset()
    train_loader = create_pytorch_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = create_pytorch_loader(X_val, y_val, batch_size=batch_size*2, shuffle=False)
    test_loader = create_pytorch_loader(X_test, y_test, batch_size=batch_size*2, shuffle=False)

    return train_loader, val_loader, test_loader
