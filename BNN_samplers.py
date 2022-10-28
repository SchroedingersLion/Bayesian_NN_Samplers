"""
This file contains schemes to sample from the posterior distribution of Bayesian neural networks (BNN).
The samplers receive the model, training and test data loaders, the loss criterion, and the hyperparameters as input.
The models need to provide a member function "evaluate(data_loader)" which computes loss and classification accuracy
on the passed data loader.
Output of the schemes is a tuple holding train loss, test loss, train accuracy, and test accuracy per epoch as obtained
by the "evaluate(data_loader)" function of the models.

SGHMC - Stochastic Gradient Hamiltonian Monte Carlo introduced by Chen et al., 2014.
OBABO - The OBABO splitting integrator for Langevin Dynamics as introduced by Bussi & Parrinello, 2007.

train_optimizer - Function to train the model conventionally by using the passed optimizer (such as torch.optimizer.SGD).

Open issue: Currently, the tensor shape accepted by BCE loss and NLL loss differ. If BCE loss is given as "criterion"
the "squeeze()" function needs to be used on model output (see below). 
"""

import torch
import torch.nn as nn
import numpy as np
import time
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  ## use GPU if available

#%%

class SGHMC(nn.Module):
    
    def __init__(self, model, train_loader, test_loader, criterion, lr, weight_decay, alpha, epochs):
        super(SGHMC, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.epochs = epochs
        
        
    def train(self):
        
        loss_train = np.zeros(self.epochs+1)                 
        accu_train = np.zeros(self.epochs+1)
        loss_test = np.zeros(self.epochs+1)
        accu_test = np.zeros(self.epochs+1)
        
        (loss_train[0], accu_train[0]) = self.model.evaluate(self.train_loader)
        (loss_test[0], accu_test[0]) = self.model.evaluate(self.test_loader)        
        
        kin_energy = np.zeros((self.epochs+1,3));  L2weight = np.zeros((self.epochs+1,3))   # DEBUG
        
        datasize = len(self.train_loader.dataset)
        
        squeeze = True if type(self.criterion) == torch.nn.modules.loss.BCELoss else False   # squeeze network output
                                                                                             # for BCELoss (not required
                                                                                             # for NLLLoss)
        
        start_time = time.time()
        print("Starting SGHMC sampling...")
        
        for p in self.model.parameters():
            p.buf = torch.randn(p.size()).to(device)*np.sqrt(self.lr)
            
        for (idx,p) in enumerate(self.model.parameters()):
            kin_energy[0,idx//2] += (p.buf**2).sum() ; L2weight[0,idx//2] = (p.data**2).sum()    # DEBUG
        
    
        for epoch in range(1, self.epochs+1):
            print("enter epich ", epoch)
            self.model.train()
            # if epoch==20: 
            #     self.lr = self.lr/200 ; print("At epoch 20: reduced lr by 200")    # DEBUG
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(device), target.to(device)
                self.model.zero_grad()
                output = self.model(data) 
                if squeeze: output=output.squeeze()
                loss = self.criterion(output, target)*datasize
                loss.backward()
                self.update_params()
        
            (loss_train[epoch], accu_train[epoch]) = self.model.evaluate(self.train_loader)
            (loss_test[epoch], accu_test[epoch]) = self.model.evaluate(self.test_loader)
            
            for (idx,p) in enumerate(self.model.parameters()):
                kin_energy[epoch,idx//2] += (p.buf**2).sum() ; L2weight[epoch,idx//2] = (p.data**2).sum()    # DEBUG
            
            if epoch%5==0:
                print("SGHMC EPOCH {} DONE!".format(epoch))
        
        end_time = time.time()
        print("Training took {} seconds, i.e {} minutes, with {} seconds per epoch!"
              .format(end_time-start_time, (end_time-start_time)/60, (end_time-start_time)/self.epochs))
            
        return (loss_train, loss_test, accu_train, accu_test, kin_energy, L2weight)     # DEBUG


    def update_params(self):

        for p in self.model.parameters():
        
            d_p = p.grad.data
            d_p.add_(p.data, alpha=self.weight_decay)
            eps = torch.randn(p.size()).to(device)
            p.buf.mul_(1-self.alpha)
            p.buf.add_(-self.lr*d_p + (2.0 * self.lr * self.alpha)**.5 * eps)
            p.data.add_(p.buf)
        




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
        
        loss_train = np.zeros(self.epochs+1)                
        accu_train = np.zeros(self.epochs+1)                 
        loss_test = np.zeros(self.epochs+1)
        accu_test = np.zeros(self.epochs+1)
        
        (loss_train[0], accu_train[0]) = self.model.evaluate(self.train_loader)
        (loss_test[0], accu_test[0]) = self.model.evaluate(self.test_loader)        
        
        datasize = len(self.train_loader.dataset)
        
        squeeze = True if type(self.criterion) == torch.nn.modules.loss.BCELoss else False   # squeeze network output
                                                                                             # for BCELoss (not required
                                                                                             # for NLLLoss)
        
        start_time = time.time()
        print("Starting OBABO sampling...")     
        
        (data, target) = next(iter(self.train_loader))          # compute initial gradients
        data, target = data.to(device), target.to(device)
        self.model.zero_grad()
        output = self.model(data)
        if squeeze: output=output.squeeze()
        loss = self.criterion(output, target)*datasize
        loss.backward()
        
        for p in self.model.parameters():                       # create momentum buffers
            p.buf = torch.zeros(p.size()).to(device)
        
        for epoch in range(1, self.epochs+1):
            self.model.train()
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                
                self.update_params_OBA()    # OBA-steps
                
                data, target = data.to(device), target.to(device)   # compute new gradients
                self.model.zero_grad()
                output = self.model(data)
                if squeeze: output=output.squeeze()
                output = self.model(data)
                loss = self.criterion(output, target)*datasize
                loss.backward()
                
                self.update_params_BO()     # BO-steps
        
            (loss_train[epoch], accu_train[epoch]) = self.model.evaluate(self.train_loader)
            (loss_test[epoch], accu_test[epoch]) = self.model.evaluate(self.test_loader)
            
            if epoch%5==0:
                print("OBABO EPOCH {} DONE!".format(epoch))
        
        end_time = time.time()
        print("Training took {} seconds, i.e {} minutes, with {} seconds per epoch!"
              .format(end_time-start_time, (end_time-start_time)/60, (end_time-start_time)/self.epochs))
            
        return (loss_train, loss_test, accu_train, accu_test)
    
    
    def update_params_OBA(self):        ## ERROR!!!! gradients are still full from last BO method!!!
        # print("CARE!! FIX GRADIENTS!!")
       
        for p in self.model.parameters():
        
            d_p = p.grad.data
            d_p.add_(p.data, alpha=self.weight_decay)
            
            eps = torch.randn(p.size()).to(device)                                  # O-step
            p.buf.mul_(self.sqrt_a)                        
            p.buf.add_(eps, alpha=self.sqrt_one_minus_a) 
            
            p.buf.add_(d_p, alpha=-0.5*self.lr)                                     # B-step
                                
            p.data.add_(p.buf, alpha=self.lr)                                       # A-step


    def update_params_BO(self):
       
        for p in self.model.parameters():
            
            d_p = p.grad.data
            d_p.add_(p.data, alpha=self.weight_decay)
            
            p.buf.add_(d_p, alpha=-0.5*self.lr)             # B-step                                
                              
            eps = torch.randn(p.size()).to(device)          # O-step
            p.buf.mul_(self.sqrt_a)                        
            p.buf.add_(eps, alpha=self.sqrt_one_minus_a)    
         



class HMC(nn.Module):
    
    def __init__(self, model, train_loader, test_loader, criterion, lr, weight_decay, gamma, epochs, L):
        super(HMC, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.epochs = epochs
        self.L = L
        
        
    def train(self):
        
        datasize = len(self.train_loader.dataset)
        
        N = self.epochs / (self.L * self.train_loader.batch_size / datasize + 1)    # number of outer loop iterations,
        N = np.rint( N + 0.5 )                                                      # i.e. samples to take, for the desired
                                                                                    # number of epochs.
        loss_train = np.zeros(N+1)                
        accu_train = np.zeros(N+1)                 
        loss_test = np.zeros(N+1)
        accu_test = np.zeros(N+1)
        
        (loss_train[0], accu_train[0]) = self.model.evaluate(self.train_loader)
        (loss_test[0], accu_test[0]) = self.model.evaluate(self.test_loader)        
        
        squeeze = True if type(self.criterion) == torch.nn.modules.loss.BCELoss else False   # squeeze network output
                                                                                             # for BCELoss (not required
                                                                                             # for NLLLoss)
        
        for p in self.model.parameters():                       # create momentum buffers
            p.buf = torch.zeros(p.size()).to(device)


        (data, target) = next(iter(self.train_loader))          # compute initial gradients 
        data, target = data.to(device), target.to(device)
        self.model.zero_grad()
        output = self.model(data)
        if squeeze: output=output.squeeze()
        loss = self.criterion(output, target)*datasize
        loss.backward()
        for p in list(self.model.parameters()):
            p.grad.data.add_(p.data, alpha=self.weight_decay)
        

        param_list_curr = copy.deepcopy(list(self.model.parameters()))  # buffer for the params before the BAB steps
        
        U0 = get_energy(self.model, self.train_loader, self.criterion, self.weight_decay)  # get pot. energy
        
        
        
        start_time = time.time()
        print("Starting HMC sampling...")     
        
        for n in range(1,N+1):
            
            
            K0 = 0
            for p in list(self.model.parameters()):                 # resample momentum and 
                p.buf = torch.randn(p.size()).to(device)            # store initial kin. energy
                K0 += (p.buf**2).sum()
            K0 *= 0.5
                
            ## L leapfrog steps     
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if batch_idx == self.L: break
                
                for p in list(self.model.parameters()):
                    p.buf.add_(p.grad.data, alpha=-0.5*self.lr)                     # B-step
                    p.data.add_(p.buf, alpha=self.lr)                               # A-step
                    
                data, target = data.to(device), target.to(device)               # compute new gradients
                self.model.zero_grad()
                output = self.model(data)
                if squeeze: output=output.squeeze()
                loss = self.criterion(output, target)*datasize
                loss.backward()
                
                for p in list(self.model.parameters()):
                    p.grad.data.add_(p.data, alpha=self.weight_decay)                    
                    p.buf.add_(p.grad.data, alpha=-0.5*self.lr)                     # B-step
            
            
            ## Metropolis
            U1 = get_energy(self.model, self.train_loader, self.criterion, self.weight_decay)  # get pot. energy
            K1 = 0
            for p in list(self.model.parameters()):
                K1 += (p.buf**2).sum() 
            K1 = 0.5*K1.item()
            
            MH = torch.exp( -1 * (U1 - U0 + K1-K0) )
            
            if( torch.rand(1) < min(1., MH) ):          # accept sample
               
                (loss_train[n], accu_train[n]) = self.model.evaluate(self.train_loader)
                (loss_test[n], accu_test[n]) = self.model.evaluate(self.test_loader)
                
                param_list_curr = copy.deepcopy(list(self.model.parameters()))
                U0 = U1
                
            else:
                (loss_train[n], accu_train[n]) = (loss_train[n-1], accu_train[n-1])
                (loss_test[n], accu_test[n]) = (loss_test[n-1], accu_test[n-1])
                
                with torch.no_grad():
                    for (idx,p) in enumerate(list(self.model.parameters())):
                        p.copy_(param_list_curr[idx])
                

            
            if N%5==0:
                print("HMC ITERATION {} DONE!".format(n))
        
        end_time = time.time()
        print("Training took {} seconds, i.e {} minutes, with {} seconds per epoch!"
              .format(end_time-start_time, (end_time-start_time)/60, (end_time-start_time)/self.epochs))
            
        return (loss_train, loss_test, accu_train, accu_test)
        
        
        
        
def get_energy(model, dataloader, criterion, weight_decay):
    """
    Computes potential energy for given model, data set, criterion, and weight decay.
    """
    # first the likelihood part...
    loss = 0
    criterion.reduction = "sum"
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target)
    U = loss.item()
    criterion.reduction = "mean"
    
    # ... then the prior / L2 part
    L2 = 0
    for p in list(model.parameters()):
        L2 += (p.data**2).sum()
    
    U += 0.5 * weight_decay * L2.item()
    
    return U



        
def train_optimizer(model, optimizer, train_loader, test_loader, criterion, epochs):
    
    loss_train = np.zeros(epochs+1)                 
    accu_train = np.zeros(epochs+1)                 
    loss_test = np.zeros(epochs+1)
    accu_test = np.zeros(epochs+1)
    
    (loss_train[0], accu_train[0]) = model.evaluate(train_loader)
    (loss_test[0], accu_test[0]) = model.evaluate(test_loader)        
    
    datasize = len(train_loader.dataset)
    
    start_time = time.time()
    print("Start optimizing...")      

    for epoch in range(1, epochs+1):
        model.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            # output = model(data).squeeze()
            output = model(data)
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
        
    return (loss_train, loss_test, accu_train, accu_test)


