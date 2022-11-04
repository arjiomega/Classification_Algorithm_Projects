from ctypes.wintypes import DWORD
import sys
from tkinter import W
sys.path.insert(0, '../components/')
from importlist import *

def normalize(X):
    X_norm = X / np.linalg.norm(X,axis=1,ord=2,keepdims= True)
    return X_norm

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
    activation_funcs = model_architecture["activation_function"]

    for l in range(1,L):
        if activation_funcs[l].__name__ == "sigmoid":
            multiplier = 0.01
        elif activation_funcs[l].__name__ == "relu":
            multiplier = np.sqrt(2/nodes[l-1])

        params["W" + str(l)] = np.random.randn(nodes[l], nodes[l-1]) * multiplier
        params["b" + str(l)] = np.zeros((nodes[l], 1))

    return params

def forward_propagation(X_train,params,model_architecture):

    L = model_architecture["layer_count"]

    activation_funcs = model_architecture["activation_function"]

    # List of A for different layers
    A_list = []
    Z_list = []
    A_list.append(X_train)
    Z_list.append(None)

    # iterate over all layers except input layer
    for l in range(1,L):

        W = params["W" + str(l)]
        b = params["b" + str(l)]

        Z = np.dot(W,A_list[l-1]) + b
        Z_list.append(Z)
        A_list.append(activation_funcs[l](Z))
       
    return A_list, Z_list



def backward_propagation(params,Y_train,A_list,Z_list,model_architecture,lambd_=None):

    L = model_architecture["layer_count"]
    activation_funcs = model_architecture["activation_function"]

    m = Y_train.shape[1]

    grads = {}
 
    counter = 0
    for l in range((L-1),0,-1):
        if activation_funcs[l].__name__ == "sigmoid":

            if counter < 1:
                counter += 1

                dL_dA = - (np.divide(Y_train,A_list[l]) - np.divide((1-Y_train),(1-A_list[l])))
                dA_dZ = A_list[l] * (1-A_list[l])
                grads["dL_dZ" + str(l)] = dL_dA * dA_dZ

            else:
                dZ_dA = params["W" + str(l+1)]
                dA_dZ = A_list[l] * (1-A_list[l])
                grads["dL_dZ" + str(l)] = grads["dL_dZ" + str(l+1)] * dZ_dA * dA_dZ

        # assuming that output layer does not use relu activation (consider this in future)
        elif activation_funcs[l].__name__ == "relu":
            dZ_dA = params["W" + str(l+1)]
            dA_dZ = (Z_list[l] >= 0).astype(int)
            grads["dL_dZ" + str(l)] = np.dot(dZ_dA.T, grads["dL_dZ" + str(l+1)]) * dA_dZ

        dZ_dW = A_list[l-1]
        dZ_db = 1

        if lambd_ == None:
            dL_dW = 1/m * np.dot(grads["dL_dZ" + str(l)], dZ_dW.T)
        else:
            dL_dW = 1/m * np.dot(grads["dL_dZ" + str(l)], dZ_dW.T) + ( (lambd_/m) * params["W" + str(l)] )

        dL_db = 1/m *  grads["dL_dZ" + str(l)] 

        grads["dL_dW" + str(l)] = dL_dW
        grads["dL_db" + str(l)] = dL_db

    return grads

def update_params(params_input,grads,learning_rate,model_architecture):

    L = model_architecture["layer_count"]
    params = params_input.copy()

    for l in range(1,L):
        params["W" + str(l)] = params["W" + str(l)] - (learning_rate * grads["dL_dW" + str(l)])
        params["b" + str(l)] = params["b" + str(l)] - (learning_rate * grads["dL_db" + str(l)])

    return params

def cost_solver(A,Y,params,hyperparams,model_architecture):

    m = A[-1].shape[1]
    lambd_ = hyperparams["lambd_"]
    L = model_architecture["layer_count"]

    sum_params = 0

    for l in range(1,L):
        sum_params +=   np.sum(np.square(  params["W" + str(l)]  ))  

    L2_regularization = 1/m * lambd_/2 * sum_params

    cost = - 1/m * np.sum( np.dot(Y,np.log(A[-1].T)) + np.dot((1-Y),np.log(1-A[-1].T)) ) + L2_regularization

    return cost