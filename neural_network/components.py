import sys
from tkinter import W
sys.path.insert(0, '../components/')
from importlist import *

# Activation Functions
def sigmoid(Z):
    A = 1 / (1+np.exp(-Z))
    return A

def relu(Z):
    A = np.maximum(0,Z)
    return A
######################

def initialize_params(model_architecture):
    
    params = {}

    L = model_architecture["layer_count"]
    nodes = model_architecture["node_count"]

    for l in range(1,L):
        params["W" + str(l)] = np.random.randn(nodes[l], nodes[l-1])* 0.01
        params["b" + str(l)] = np.zeros((nodes[l], 1))

    return params

def forward_propagation(X_train,params,model_architecture):

    L = model_architecture["layer_count"]

    activation_funcs = model_architecture["activation_function"]

    # For first layer A0 is Input layer X
    A = X_train

    # iterate over all layers except input layer
    for l in range(1,L):

        A_prev = A
        W = params["W" + str(l)]
        b = params["b" + str(l)]

        Z = np.dot(W,A_prev) + b
        A = activation_funcs[l](Z)
       
    return 1

def backward_propagation(model_architecture):

    L = model_architecture["layer_count"]
    activation_funcs = model_architecture["activation_function"]

    grads = {}

    for l in reversed(range(L)):
        #dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)], current_cache, "relu")
        
        if name_of(activation_funcs[l]) == "relu":
            Z = 
            
            dZ[Z <= 0] = 0

        elif name_of(activation_funcs[l]) == "sigmoid":
            
            dZ =


        grads["dA" + str(l)]
        grads["dW" + str(l)]
        grads["db" + str(l)]
    # dZ
    # dW
    # dB
    # dA
    return 1