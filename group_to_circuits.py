import numpy as np
from qibo import models, gates
from qibo.optimizers import optimize
import random
import time
from numpy.linalg import inv
import matplotlib.pyplot as plt

start_time = time.time()

def ansatz(params, trainable=True):
    '''
    define the ansatz
    '''
    yield gates.RY(0, theta=params[0], trainable=trainable)
    yield gates.RY(1, theta=params[1], trainable=trainable)
    yield gates.CZ(0, 1)
    yield gates.RY(0, theta=params[2], trainable=trainable)
    yield gates.RY(1, theta=params[3], trainable=trainable)

def ansatz_dagger(params, trainable=True):
    '''
    define the inverse of the ansatz
    '''
    yield gates.RY(0, theta=-params[2], trainable=trainable)
    yield gates.RY(1, theta=-params[3], trainable=trainable)
    yield gates.CZ(0, 1)
    yield gates.RY(0, theta=-params[0], trainable=trainable)
    yield gates.RY(1, theta=-params[1], trainable=trainable)

def sub_circuit_0(ru):
    c=models.Circuit(2)
    c.add(ansatz(ru, trainable=False))
    c.add(ansatz(np.zeros(4)))
    c.add(ansatz(np.zeros(4)))
    c.add(ansatz_dagger(ru, trainable=False))
    return c

def sub_circuit_1(ru):
    c=models.Circuit(2)
    c.add(ansatz(ru, trainable=False))
    c.add(ansatz(np.zeros(4)))
    c.add(ansatz(np.zeros(4)))
    c.add(ansatz(np.zeros(4)))
    c.add(ansatz(np.zeros(4)))
    c.add(ansatz_dagger(ru, trainable=False))
    return c

def sub_circuit_2(ru):
    c=models.Circuit(2)
    c.add(ansatz(ru, trainable=False))
    c.add(ansatz(np.zeros(4)))
    c.add(ansatz(np.zeros(4)))
    c.add(ansatz(np.zeros(4)))
    c.add(ansatz(np.zeros(4)))
    c.add(ansatz(np.zeros(4)))
    c.add(ansatz(np.zeros(4)))
    c.add(ansatz_dagger(ru, trainable=False))
    return c
    
def matrix(params):
    '''
    return 3 full matrices Ua, Ub and Uc for a given set of parameters
    '''
    a = params[:4]
    b = params[4:8]
    c = params[8:]
    Ua=models.Circuit(2)
    Ua.add(ansatz(a,trainable=False))
    Ub=models.Circuit(2)
    Ub.add(ansatz(b,trainable=False))
    Uc=models.Circuit(2)
    Uc.add(ansatz(c,trainable=False))
    return Ua.unitary(), Ub.unitary(), Uc.unitary()
    
def check_identity(params):
    '''
    check if the training result makes all circuits of words equal to identity
    '''
    s=1000
    a = params[:4]
    b = params[4:8]
    c = params[8:]
    total_loss=0
    for i in range(s):
        sub_loss=0
        ru=2*np.pi*np.random.random(4)
        #(a)
        c1=sub_circuit_0(ru)
        c1_params=list(a)+list(a)
        c1.set_parameters(c1_params)
        state1=c1().state()
        sub_loss=sub_loss-np.abs(state1[0])
        #(b)
        c2=sub_circuit_0(ru)
        c2_params=list(b)+list(b)
        c2.set_parameters(c2_params)
        state2=c2().state()
        sub_loss=sub_loss-np.abs(state2[0])
        #(c)
        c3=sub_circuit_1(ru)
        c3_params=list(c)+list(c)+list(c)+list(c)
        c3.set_parameters(c3_params)
        state3=c3().state()
        sub_loss=sub_loss-np.abs(state3[0])
        #(d)
        c4=sub_circuit_1(ru)
        c4_params=list(b)+list(c)+list(b)+list(c)
        c4.set_parameters(c4_params)
        state4=c4().state()
        sub_loss=sub_loss-np.abs(state4[0])
        #(e)
        c5=sub_circuit_1(ru)
        c5_params=list(a)+list(b)+list(a)+list(b)
        c5.set_parameters(c5_params)
        state5=c5().state()
        sub_loss=sub_loss-np.abs(state5[0])
        #(f)
        c6=sub_circuit_2(ru)
        c6_params=list(a)+list(c)+list(c)+list(c)+list(a)+list(c)
        c6.set_parameters(c6_params)
        state6=c6().state()
        sub_loss=sub_loss-np.abs(state6[0])
        total_loss=total_loss+sub_loss
    return -total_loss/(6*s)
    
def check_faithful(params):
    '''
    check if the representation is faithful
    if the result is 1, then the representation is unfaithful
    corresponding to FIG.7
    '''
    s=1000
    a = params[:4]
    b = params[4:8]
    c = params[8:]
    loss_1=0
    loss_2=0
    loss_3=0
    for i in range(s):
        ru=2*np.pi*np.random.random(6)
        #(a)
        c1=sub_circuit_0(ru)
        c1_params=list(a)+list(b)
        c1.set_parameters(c1_params)
        state1=c1().state()
        loss_1=loss_1-np.abs(state1[0])
        #(b)
        c2=sub_circuit_0(ru)
        c2_params=list(c)+list(c)
        c2.set_parameters(c2_params)
        state2=c2().state()
        loss_2=loss_2-np.abs(state2[0])
    return -loss_1/s,-loss_2/s
    
def loss(params):
    '''
    define the loss function for training
    corresponding to FIG.5
    '''
    s=3
    a = params[:4]
    b = params[4:8]
    c = params[8:]
    total_loss=0
    for i in range(s):
        sub_loss=0
        ru=2*np.pi*np.random.random(4)
        #(a)
        c1=sub_circuit_0(ru)
        c1_params=list(a)+list(a)
        c1.set_parameters(c1_params)
        state1=c1().state()
        sub_loss=sub_loss-np.abs(state1[0])
        #(b)
        c2=sub_circuit_0(ru)
        c2_params=list(b)+list(b)
        c2.set_parameters(c2_params)
        state2=c2().state()
        sub_loss=sub_loss-np.abs(state2[0])
        #(c)
        c3=sub_circuit_1(ru)
        c3_params=list(c)+list(c)+list(c)+list(c)
        c3.set_parameters(c3_params)
        state3=c3().state()
        sub_loss=sub_loss-np.abs(state3[0])
        #(d)
        c4=sub_circuit_1(ru)
        c4_params=list(b)+list(c)+list(b)+list(c)
        c4.set_parameters(c4_params)
        state4=c4().state()
        sub_loss=sub_loss-np.abs(state4[0])
        #(e)
        c5=sub_circuit_1(ru)
        c5_params=list(a)+list(b)+list(a)+list(b)
        c5.set_parameters(c5_params) 
        state5=c5().state()
        sub_loss=sub_loss-np.abs(state5[0])
        #(f)
        c6=sub_circuit_2(ru)
        c6_params=list(a)+list(c)+list(c)+list(c)+list(a)+list(c)
        c6.set_parameters(c6_params)
        state6=c6().state()
        sub_loss=sub_loss-np.abs(state6[0])
        total_loss=total_loss+sub_loss
    return total_loss/(6*s)

initial_params = 2*np.pi*np.random.random(12)
best, params, extra=optimize(loss, initial_params, method='cma')

print(best)
print('--------------------')
print('trained parameters:',params)
print('if it is close to 1 then words are close to identity:',(check_identity(params)))
print('if it is close to 1 then representation is unfaithful:',(check_faithful(params)))

record_params=[5.84551571,-8.06312288,-12.12870095,-17.06961836,16.17214308,13.9280257,-9.88895781,4.92153015,13.69949734,-8.06312283,2.00846589,33.19586423]
print('--------------------')
print('recorded parameters:',record_params)
print('if it is close to 1 then words are close to identity:',(check_identity(record_params)))
print('if it is close to 1 then representation is unfaithful:',(check_faithful(record_params)))

theta=2*np.pi*random.random()
analytic_params=[3*np.pi-theta, 3*theta, theta-np.pi, 4*np.pi-3*theta, theta, 3*theta-np.pi, 2*np.pi-theta, 3*np.pi-3*theta, (np.pi-2*theta)/2, 3*theta, (np.pi+2*theta)/2, 4*np.pi-3*theta]
print('--------------------')
print('analytic parameters:',analytic_params)
print('if it is close to 1 then words are close to identity:',(check_identity(analytic_params)))
print('if it is close to 1 then representation is unfaithful:',(check_faithful(analytic_params)))

theta=0
analytic_params=[3*np.pi-theta, 3*theta, theta-np.pi, 4*np.pi-3*theta, theta, 3*theta-np.pi, 2*np.pi-theta, 3*np.pi-3*theta, (np.pi-2*theta)/2, 3*theta, (np.pi+2*theta)/2, 4*np.pi-3*theta]
print('--------------------')
print('theta=0')
print('if it is close to 1 then words are close to identity:',(check_identity(analytic_params)))
print('if it is close to 1 then representation is unfaithful:',(check_faithful(analytic_params)))
Ua,Ub,Uc=matrix(analytic_params)
Ua=Ua.real.astype(int)
Ub=Ub.real.astype(int)
Uc=Uc.real.astype(int)
print('U(a)!=1')
print(Ua)
print('U(a^2)=1')
print(Ua @ Ua)
print('U(b)!=1')
print(Ub)
print('U(b^2)=1')
print(Ub @ Ub)
print('U(c)!=1')
print(Uc)
print('U(c^2)!=1')
print(Uc @ Uc)
print('U(c^4)=1')
print(Uc @ Uc @ Uc @ Uc)
print('U(ab)=U(ba)!=1')
print(Ua @ Ub)
print(Ub @ Ua)
print('U(ac)=U(ca)!=1')
print(Ua @ Uc)
print(Uc @ Ua)

print("--- %s seconds ---" % (time.time() - start_time))



