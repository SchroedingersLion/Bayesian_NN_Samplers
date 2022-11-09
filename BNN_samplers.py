"""
This file contains schemes to sample from the posterior distribution of Bayesian neural networks (BNN).
The samplers receive the model, training and test data loaders, the loss criterion, and the hyperparameters as input.
The models need to provide a member function "evaluate(data_loader)" which computes loss and classification accuracy
on the passed data loader.
Output of the schemes is a tuple holding train loss, test loss, train accuracy, and test accuracy per epoch as obtained
by the "evaluate(data_loader)" function of the models.
The unadjusted schemes also yield arrays for kinetic energies per network layer and squared L2 parameter norms per layer.

SGHMC - Stochastic Gradient Hamiltonian Monte Carlo introduced by Chen et al., 2014.
SGHMC_scaled = Scaled version of SGHMC to compare it to SGD with momentum, see also Chen et al., 2014.
OBABO - The OBABO splitting integrator for Langevin Dynamics as introduced by Bussi & Parrinello, 2007.
HMC - Hamiltonian Monte Carlo, see eg. Neal 2012. It can also be used with stochastic gradients (not to be
      confused with SGHMC).


train_optimizer - Function to train the model conventionally by using the passed optimizer (such as torch.optimizer.SGD).

Open issue: The kinetic energies and L2 norms are computed layer-wise. Each column corresponds to one layer.
            Currently, the number of layers is trivially computed, but assumes a fully-connected model with 
            two parameters per layer (i.e. weight and bias). For other network types, this needs to be adjusted.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  ## use GPU if available

#%% SGHMC

class SGHMC(nn.Module):
    """
    SGHMC as introduced by Chen et al., 2014.
    """
    
    def __init__(self, model, train_loader, test_loader, criterion, lr, weight_decay, gamma, epochs):
        super(SGHMC, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.epochs = epochs
        
        
    def train(self):
        """
        Main train/sampling routine. Returns arrays of losses and accuracies on both train and test set. 
        Also returns arrays of kinetic energies and squared-L2 norm of network parameters (each column corresponds
        corresponds to single network layer).
        """        
        
        loss_train = np.zeros(self.epochs+1)                 
        accu_train = np.zeros(self.epochs+1)
        loss_test = np.zeros(self.epochs+1)
        accu_test = np.zeros(self.epochs+1)
        
        (loss_train[0], accu_train[0]) = self.model.evaluate(self.train_loader)     # initial losses and accuracies
        (loss_test[0], accu_test[0]) = self.model.evaluate(self.test_loader)        
        
        no_layers = int( len(list(self.model.parameters())) / 2 )
        kin_energy = np.zeros((self.epochs+1, no_layers));  L2weight = np.zeros((self.epochs+1, no_layers))                                                                                          
        
        datasize = len(self.train_loader.dataset)
        
        squeeze = True if type(self.criterion) == torch.nn.modules.loss.BCELoss else False   # squeeze network output
                                                                                             # for BCELoss (not required
                                                                                             # for NLLLoss)
        
        start_time = time.time()
        print("Starting SGHMC sampling...")
        
        for p in self.model.parameters():               # create momentum buffers
            p.buf = torch.randn(p.size()).to(device)
            
        for (idx,p) in enumerate(self.model.parameters()):                                      # initial kin. energies
            kin_energy[0,idx//2] += (p.buf**2).sum() ; L2weight[0,idx//2] = (p.data**2).sum()   # and L2 norms    
        
    
        for epoch in range(1, self.epochs+1):
            self.model.train()
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(device), target.to(device)
                self.fill_gradients(data, target, squeeze, datasize)
                self.update_params()
        
            (loss_train[epoch], accu_train[epoch]) = self.model.evaluate(self.train_loader)     # take measurements
            (loss_test[epoch], accu_test[epoch]) = self.model.evaluate(self.test_loader)       
            
            for (idx,p) in enumerate(self.model.parameters()):                                  
                kin_energy[epoch,idx//2] += (p.buf**2).sum() ; L2weight[epoch,idx//2] = (p.data**2).sum() 
            
            if epoch%5==0:
                print("SGHMC EPOCH {} DONE!".format(epoch))
        
        end_time = time.time()
        print("Training took {} seconds, i.e {} minutes, with {} seconds per epoch!"
              .format(end_time-start_time, (end_time-start_time)/60, (end_time-start_time)/self.epochs))
            
        return (loss_train, loss_test, accu_train, accu_test, kin_energy, L2weight)



    def update_params(self):
        """
        Updates the parameters. Assumes forces are stored in parameter gradients.
        """

        for p in self.model.parameters():
        
            eps = torch.randn(p.size()).to(device)
            p.buf.mul_(1-self.lr*self.gamma)
            p.buf.add_(-self.lr*p.grad.data + (2.0 * self.lr * self.gamma)**.5 * eps)
            p.data.add_(self.lr*p.buf)



    def fill_gradients(self, data, target, squeeze, datasize):
        """
        Fills gradients of the model on batch (data, target).
        """
        
        self.model.zero_grad()
        output = self.model(data)
        if squeeze: output=output.squeeze()
        loss = self.criterion(output, target)*datasize
        loss.backward()
        
        for p in list(self.model.parameters()):                 # weight decay / prior component
            p.grad.data.add_(p.data, alpha=self.weight_decay) 
            



class SGHMC_scaled(nn.Module):
    """
    SGHMC as introduced by Chen et al., 2014. Scaled version proposed by Chen et al. to compare it to SGD
    with momentum. Often used in the literature.
    """
    
    def __init__(self, model, train_loader, test_loader, criterion, lr, weight_decay, alpha, epochs):
        super(SGHMC_scaled, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.epochs = epochs
        
        
    def train(self):
        """
        Main train/sampling routine. Returns arrays of losses and accuracies on both train and test set. 
        Also returns arrays of kinetic energies and squared-L2 norm of network parameters (each column corresponds
        corresponds to single network layer).
        """        
        
        loss_train = np.zeros(self.epochs+1)                 
        accu_train = np.zeros(self.epochs+1)
        loss_test = np.zeros(self.epochs+1)
        accu_test = np.zeros(self.epochs+1)
        
        (loss_train[0], accu_train[0]) = self.model.evaluate(self.train_loader)         # initial losses and accuracies
        (loss_test[0], accu_test[0]) = self.model.evaluate(self.test_loader)        
        
        no_layers = int( len(list(self.model.parameters())) / 2 )
        kin_energy = np.zeros((self.epochs+1, no_layers));  L2weight = np.zeros((self.epochs+1, no_layers))  
        
        datasize = len(self.train_loader.dataset)
        
        squeeze = True if type(self.criterion) == torch.nn.modules.loss.BCELoss else False   # squeeze network output
                                                                                             # for BCELoss (not required
                                                                                             # for NLLLoss)
        
        start_time = time.time()
        print("Starting SGHMC sampling...")
        
        for p in self.model.parameters():
            p.buf = torch.randn(p.size()).to(device)*np.sqrt(self.lr)

        
        for (idx,p) in enumerate(self.model.parameters()):                                              # initial kin.
            kin_energy[0,idx//2] += (p.buf**2/self.lr).sum() ; L2weight[0,idx//2] = (p.data**2).sum()   # energies and
                                                                                                        # L2 norms
    
        for epoch in range(1, self.epochs+1):
            self.model.train()
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(device), target.to(device)
                self.fill_gradients(data, target, squeeze, datasize)
                self.update_params()
        
            (loss_train[epoch], accu_train[epoch]) = self.model.evaluate(self.train_loader)     # take measurements
            (loss_test[epoch], accu_test[epoch]) = self.model.evaluate(self.test_loader)
                
            for (idx,p) in enumerate(self.model.parameters()):
                kin_energy[epoch,idx//2] += (p.buf**2/self.lr).sum() ; L2weight[epoch,idx//2] = (p.data**2).sum()
            
            if epoch%5==0:
                print("SGHMC EPOCH {} DONE!".format(epoch))
        
        end_time = time.time()
        print("Training took {} seconds, i.e {} minutes, with {} seconds per epoch!"
              .format(end_time-start_time, (end_time-start_time)/60, (end_time-start_time)/self.epochs))
            
        return (loss_train, loss_test, accu_train, accu_test, kin_energy, L2weight)



    def update_params(self):
        """
        Updates the parameters. Assumes forces are stored in parameter gradients.
        """

        for p in self.model.parameters():
        
            eps = torch.randn(p.size()).to(device)
            p.buf.mul_(1-self.alpha)
            p.buf.add_(-self.lr*p.grad.data + (2.0 * self.lr * self.alpha)**.5 * eps)
            p.data.add_(p.buf) 
            
            
            
    def fill_gradients(self, data, target, squeeze, datasize):
        """
        Fills gradients of the model on batch (data, target).
        """
        
        self.model.zero_grad()
        output = self.model(data)
        if squeeze: output=output.squeeze()
        loss = self.criterion(output, target)*datasize
        loss.backward()
        
        for p in list(self.model.parameters()):                 # weight decay / prior component
            p.grad.data.add_(p.data, alpha=self.weight_decay) 
            
            
#%% OBABO

class OBABO(nn.Module):
    
    def __init__(self, model, train_loader, test_loader, criterion, lr, weight_decay, gamma, epochs):
        super(OBABO, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.epochs = epochs
        
        self.sqrt_a = np.sqrt( np.exp(-self.gamma*self.lr) )                # constants used in integrator
        self.sqrt_one_minus_a = np.sqrt( 1 - np.exp(-self.gamma*self.lr) )
        
        
    def train(self):
        """
        Main train/sampling routine. Returns arrays of losses and accuracies on both train and test set. 
        Also returns arrays of kinetic energies and squared-L2 norm of network parameters (each column corresponds
        corresponds to single network layer).
        """        
        
        loss_train = np.zeros(self.epochs+1)                
        accu_train = np.zeros(self.epochs+1)                 
        loss_test = np.zeros(self.epochs+1)
        accu_test = np.zeros(self.epochs+1)
        
        (loss_train[0], accu_train[0]) = self.model.evaluate(self.train_loader)         # initial losses and accuracies
        (loss_test[0], accu_test[0]) = self.model.evaluate(self.test_loader)        
        
        no_layers = int( len(list(self.model.parameters())) / 2 )
        kin_energy = np.zeros((self.epochs+1, no_layers));  L2weight = np.zeros((self.epochs+1, no_layers))  
        
        datasize = len(self.train_loader.dataset)
        
        squeeze = True if type(self.criterion) == torch.nn.modules.loss.BCELoss else False   # squeeze network output
                                                                                             # for BCELoss (not required
                                                                                             # for NLLLoss)
        
        start_time = time.time()
        print("Starting OBABO sampling...")     
        
        (data, target) = next(iter(self.train_loader))          # compute initial gradients
        data, target = data.to(device), target.to(device)
        self.fill_gradients(data, target, squeeze, datasize)
                
        for p in self.model.parameters():                       # create momentum buffers
            p.buf = torch.zeros(p.size()).to(device)

        for (idx,p) in enumerate(self.model.parameters()):                                      # initial kin. energies
            kin_energy[0,idx//2] += (p.buf**2).sum() ; L2weight[0,idx//2] = (p.data**2).sum()   # and L2 norms 
        
        for epoch in range(1, self.epochs+1):                   # sampling loop
            self.model.train()
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                
                self.update_params_OBA()    # OBA-steps
                
                data, target = data.to(device), target.to(device)   # compute new gradients
                self.fill_gradients(data, target, squeeze, datasize)
                
                self.update_params_BO()     # BO-steps
        
            (loss_train[epoch], accu_train[epoch]) = self.model.evaluate(self.train_loader)     # take measurements.
            (loss_test[epoch], accu_test[epoch]) = self.model.evaluate(self.test_loader)        
            
            for (idx,p) in enumerate(self.model.parameters()):                                  
                kin_energy[epoch,idx//2] += (p.buf**2).sum() ; L2weight[epoch,idx//2] = (p.data**2).sum() 
            
            if epoch%5==0:
                print("OBABO EPOCH {} DONE!".format(epoch))
      
        
        end_time = time.time()
        print("Training took {} seconds, i.e {} minutes, with {} seconds per epoch!"
              .format(end_time-start_time, (end_time-start_time)/60, (end_time-start_time)/self.epochs))
            
        return (loss_train, loss_test, accu_train, accu_test, kin_energy, L2weight)
    
    
    
    def update_params_OBA(self):
        """
        Performs O-, B-, and A-step on model. Forces are assumed to be stored in parameter gradients.
        """      
       
        for p in self.model.parameters():
            
            eps = torch.randn(p.size()).to(device)                                  # O-step
            p.buf.mul_(self.sqrt_a)                        
            p.buf.add_(eps, alpha=self.sqrt_one_minus_a) 
            
            p.buf.add_(p.grad.data, alpha=-0.5*self.lr)                             # B-step
                                
            p.data.add_(p.buf, alpha=self.lr)                                       # A-step



    def update_params_BO(self):
        """
        Performs B- and O-step on model. Forces are assumed to be stored in parameter gradients.
        """

        for p in self.model.parameters():
            
            p.buf.add_(p.grad.data, alpha=-0.5*self.lr)                             # B-step                                
                              
            eps = torch.randn(p.size()).to(device)                                  # O-step
            p.buf.mul_(self.sqrt_a)                        
            p.buf.add_(eps, alpha=self.sqrt_one_minus_a)    
  


    def fill_gradients(self, data, target, squeeze, datasize):
        """
        Fills gradients of the model on batch (data, target).
        """
        
        self.model.zero_grad()
        output = self.model(data)
        if squeeze: output=output.squeeze()
        loss = self.criterion(output, target)*datasize
        loss.backward()
        
        for p in list(self.model.parameters()):                 # weight decay / prior component
            p.grad.data.add_(p.data, alpha=self.weight_decay)  

#%% HMC

class HMC(nn.Module):
    
    def __init__(self, model, train_loader, test_loader, criterion, lr, weight_decay, epochs, L):
        super(HMC, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.L = L
        
        
    def train(self):
        """
        Main train/sampling routine. Returns arrays of losses and accuracies on both train and test set.
        """
        
        datasize = len(self.train_loader.dataset)
        
        N = self.epochs / (self.L * self.train_loader.batch_size / datasize + 1)    # number of outer loop iterations,
        N = int(np.rint( N + 0.5 ))                                                 # i.e. samples to take, for the desired
                                                                                    # number of epochs
        loss_train = np.zeros(N+1)                
        accu_train = np.zeros(N+1)                 
        loss_test = np.zeros(N+1)
        accu_test = np.zeros(N+1)
        
        (loss_train[0], accu_train[0]) = self.model.evaluate(self.train_loader)     # initial losses and accuracies
        (loss_test[0], accu_test[0]) = self.model.evaluate(self.test_loader)        
        
        squeeze = True if type(self.criterion) == torch.nn.modules.loss.BCELoss else False   # squeeze network output
                                                                                              # for BCELoss (not required
                                                                                              # for NLLLoss)
        
        for p in self.model.parameters():                       # create momentum buffers
            p.buf = torch.zeros(p.size()).to(device)


        (data, target) = next(iter(self.train_loader))          # compute initial gradients 
        data, target = data.to(device), target.to(device)
        self.fill_gradients(data, target, squeeze, datasize)
        

        param_list_curr = copy.deepcopy(list(self.model.parameters()))  # buffer for the params before the BAB steps
        
        U0 = self.get_energy(squeeze)  # get pot. energy
        
        ctr = 0  # count accepted moves
        
        
        start_time = time.time()
        print("Starting HMC sampling... \nPerform {} epochs at batch size {} and L={},"\
              " i.e. {} outer loop iterations.".format(self.epochs, self.train_loader.batch_size, self.L, N))     
        
        # OUTER LOOP
        for n in range(1,N+1):  
            
            
            K0 = 0
            for p in list(self.model.parameters()):                 # resample momentum and 
                p.buf = torch.randn(p.size()).to(device)            # store initial kin. energy
                K0 += (p.buf**2).sum()
            K0 *= 0.5
                
            ## INNER LOOP / L leapfrog steps     
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if batch_idx == self.L: break
                
                for p in list(self.model.parameters()):
                    p.buf.add_(p.grad.data, alpha=-0.5*self.lr)                     # B-step
                    p.data.add_(p.buf, alpha=self.lr)                               # A-step
                    
                data, target = data.to(device), target.to(device)                   # compute new gradients
                self.fill_gradients(data, target, squeeze, datasize)
                
                for p in list(self.model.parameters()):                   
                    p.buf.add_(p.grad.data, alpha=-0.5*self.lr)                     # B-step
            
            
            ## Metropolis            
            (accept, U1) = self.MH_decision(U0, K0, squeeze)
            
            if( accept ):          # accept sample
               
                (loss_train[n], accu_train[n]) = self.model.evaluate(self.train_loader)  # losses/accuracies of proposed
                (loss_test[n], accu_test[n]) = self.model.evaluate(self.test_loader)     # sample
                
                param_list_curr = copy.deepcopy(list(self.model.parameters()))  # update parameters with proposal
                U0 = U1
                ctr += 1
                
            else:                   # reject sample
                
                (loss_train[n], accu_train[n]) = (loss_train[n-1], accu_train[n-1])   # take losses/accuracies of
                (loss_test[n], accu_test[n]) = (loss_test[n-1], accu_test[n-1])       # previous sample.
                
                with torch.no_grad():                                           
                    for (idx,p) in enumerate(list(self.model.parameters())):
                        p.copy_(param_list_curr[idx])                                 # restore model parameters
                

            
            if n%5==0:
                print("HMC ITERATION {} DONE!".format(n))
        
        end_time = time.time()
        print("Training took {} seconds, i.e {} minutes, with {} seconds per epoch!"
              .format(end_time-start_time, (end_time-start_time)/60, (end_time-start_time)/self.epochs))
        print("Total acceptance rate: {}".format(ctr/N))
            
        return (loss_train, loss_test, accu_train, accu_test)
        


    def fill_gradients(self, data, target, squeeze, datasize):
        """
        Fills gradients of the model on batch (data, target).
        """
        
        self.model.zero_grad()
        output = self.model(data)
        if squeeze: output=output.squeeze()
        loss = self.criterion(output, target)*datasize
        loss.backward()
        
        for p in list(self.model.parameters()):                 # weight decay / prior component
            p.grad.data.add_(p.data, alpha=self.weight_decay)  
        
            
        
    def MH_decision(self, U0, K0, squeeze):
        """
        Evaluates Metropolis criterion. Returns boolean (true=accept, false=reject) and new potential energy.
        """

        U1 = self.get_energy(squeeze)  # get pot. energy
        K1 = 0
        for p in list(self.model.parameters()):     # get kin. energies
            K1 += (p.buf**2).sum() 
        K1 = 0.5*K1.item()
        
        MH = torch.exp( -1 * (U1 - U0 + K1-K0) )    # evaluate MH-term
        
        accept = torch.rand(1) < min(1., MH)    # true for accept
        
        return (accept, U1)      
        
            
        
    def get_energy(self, squeeze):
        """
        Computes potential energy for evaluation of Metropolis criterion.
        """
        
        # first the likelihood part...
        loss = 0
        self.criterion.reduction = "sum"
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                if squeeze: output=output.squeeze()
                loss += self.criterion(output, target)
        U = loss.item()
        self.criterion.reduction = "mean"
        
        # ... then the prior / L2 part
        L2 = 0
        for p in list(self.model.parameters()):
            L2 += (p.data**2).sum()
        
        U += 0.5 * self.weight_decay * L2.item()
        
        return U


#%% Optimizer
        
def train_optimizer(model, optimizer, train_loader, test_loader, criterion, epochs):
    
    loss_train = np.zeros(epochs+1)                 
    accu_train = np.zeros(epochs+1)                 
    loss_test = np.zeros(epochs+1)
    accu_test = np.zeros(epochs+1)
    
    (loss_train[0], accu_train[0]) = model.evaluate(train_loader)
    (loss_test[0], accu_test[0]) = model.evaluate(test_loader)        
    
    datasize = len(train_loader.dataset)
    
    squeeze = True if type(criterion) == torch.nn.modules.loss.BCELoss else False   # squeeze network output
                                                                                         # for BCELoss (not required
                                                                                         # for NLLLoss)
    
    start_time = time.time()
    print("Start optimizing...")      

    for epoch in range(1, epochs+1):
        model.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data)
            if squeeze: output=output.squeeze()
            loss = criterion(output, target)*datasize
            loss.backward()
            optimizer.step()
    
        (loss_train[epoch], accu_train[epoch]) = model.evaluate(train_loader)
        (loss_test[epoch], accu_test[epoch]) = model.evaluate(test_loader)
        
        if epoch%5==0:
            print("OPTIMIZER EPOCH {} DONE!".format(epoch))
    
    end_time = time.time()
    print("Training took {} seconds, i.e {} minutes, with {} seconds per epoch!"
          .format(end_time-start_time, (end_time-start_time)/60, (end_time-start_time)/epochs))
    print(loss_train)     
    return (loss_train, loss_test, accu_train, accu_test)


