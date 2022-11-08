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

def forward_propagation(X_train,params,model_architecture,dropout_keep = 1.0):

    L = model_architecture["layer_count"]

    activation_funcs = model_architecture["activation_function"]

    # List of A for different layers
    A_list = []
    Z_list = []
    D_list = []
    A_list.append(X_train)
    Z_list.append(None)
    D_list.append(None)

    # iterate over all layers except input layer
    for l in range(1,L):

        W = params["W" + str(l)]
        b = params["b" + str(l)]

        Z = np.dot(W,A_list[l-1]) + b
        Z_list.append(Z)


        # Dropout random nodes (if dropout_keep do not do dropout)
        A = activation_funcs[l](Z)

        if l < (L-1):
            D = np.random.rand(A.shape[0],A.shape[1])
            D = (D < dropout_keep).astype(int)

            A = np.multiply(A,D)
            A = A / dropout_keep

        else:
            D = None

        A_list.append(A)
        D_list.append(D)

    FPcache = (A_list, Z_list, D_list)

    return FPcache



def backward_propagation(params,Y_train,FPcache,model_architecture,lambd_=None,dropout_keep = 1.0):

    L = model_architecture["layer_count"]
    activation_funcs = model_architecture["activation_function"]

    (A_list, Z_list, D_list) = FPcache

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
                dL_dZ = grads["dL_dZ" + str(l+1)] * dZ_dA * dA_dZ

                dL_dZ = np.multiply(dL_dZ,D_list[l])
                dL_dZ /= dropout_keep

                grads["dL_dZ" + str(l)] = dL_dZ

        # assuming that output layer does not use relu activation (consider this in future)
        elif activation_funcs[l].__name__ == "relu":

            '''# dropout part
            if l < (L-1):
                dL_dA = np.multiply(dL_dA,D_list[l])
                dL_dA /= dropout_keep'''

            dZ_dA = params["W" + str(l+1)]
            dA_dZ = (Z_list[l] >= 0).astype(int)
            dL_dZ = np.dot(dZ_dA.T, grads["dL_dZ" + str(l+1)]) * dA_dZ

            dL_dZ = np.multiply(dL_dZ,D_list[l])
            dL_dZ /= dropout_keep

            grads["dL_dZ" + str(l)] = dL_dZ

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

def optimizer_func(params,grads,hyperparams,optimizer,L,t):

    learning_rate = hyperparams["learning_rate"]
    epsilon = hyperparams["epsilon"]

    V = {}
    S = {}

    V_corr = {}
    S_corr = {}

    # Initialize Optimizer Grads
    for l in range(1,L):
        if (optimizer == "momentum" or optimizer ==  "adam") :
            V["dW" + str(l)] = np.zeros((params['W' + str(l)].shape))
            V["db" + str(l)] = np.zeros((params['b' + str(l)].shape))

        if (optimizer == "rmsprop" or optimizer ==  "adam"):
            S["dW" + str(l)] = np.zeros((params['W' + str(l)].shape))
            S["db" + str(l)] = np.zeros((params['b' + str(l)].shape))


    # Momentum
    if (optimizer == "momentum" or optimizer ==  "adam") :

        beta_1 = hyperparams["beta_1"]

        for l in range(1,L):

            V["dW" + str(l)] = beta_1 * V["dW" + str(l)] + ( (1-beta_1) * grads["dL_dW" + str(l)] )
            V["db" + str(l)] = beta_1 * V["db" + str(l)] + ( (1-beta_1) * grads["dL_db" + str(l)] )

            if optimizer == "momentum":

                params["W" + str(l)] = params["W" + str(l)] - (learning_rate * V["dW" + str(l)])
                params["b" + str(l)] = params["b" + str(l)] - (learning_rate * V["db" + str(l)])

    # RMSprop
    if (optimizer == "rmsprop" or optimizer ==  "adam"):

        beta_2 = hyperparams["beta_2"]

        for l in range(1,L):
            S["dW" + str(l)] = beta_2 * S["dW" + str(l)] + ( (1-beta_2) * np.power(grads["dL_dW" + str(l)],2) )
            S["db" + str(l)] = beta_2 * S["db" + str(l)] + ( (1-beta_2) * np.power(grads["dL_db" + str(l)],2) )

            if optimizer == "rmsprop":
                params["W" + str(l)] = params["W" + str(l)] - (learning_rate * (grads["dL_dW" + str(l)]/(np.sqrt(S["dW" + str(l)])+epsilon)) )
                params["b" + str(l)] = params["b" + str(l)] - (learning_rate * (grads["dL_db" + str(l)]/(np.sqrt(S["db" + str(l)])+epsilon)) )
                


    # Adam (Adaptive Moment Estimation)
    if optimizer == "adam":

        for l in range(1,L):
            ## Bias Correction
            V["dW" + str(l)] = V["dW" + str(l)] / (1-np.power(beta_1,t))
            V["db" + str(l)] = V["db" + str(l)] / (1-np.power(beta_1,t))

            S["dW" + str(l)] = S["dW" + str(l)] / (1-np.power(beta_2,t))
            S["db" + str(l)] = S["db" + str(l)] / (1-np.power(beta_2,t))

            params["W" + str(l)] = params["W" + str(l)] - (learning_rate * (V["dW" + str(l)]/(np.sqrt(S["dW" + str(l)])+epsilon)) )
            params["b" + str(l)] = params["b" + str(l)] - (learning_rate * (V["db" + str(l)]/(np.sqrt(S["db" + str(l)])+epsilon)) )


    return params

def update_params(t,params_input,grads,hyperparams,model_architecture,optimizer=None):

    L = model_architecture["layer_count"]
    params = params_input.copy()
    learning_rate = hyperparams["learning_rate"]

    if optimizer == None:
        for l in range(1,L):
            params["W" + str(l)] = params["W" + str(l)] - (learning_rate * grads["dL_dW" + str(l)])
            params["b" + str(l)] = params["b" + str(l)] - (learning_rate * grads["dL_db" + str(l)])

    else:
        t += 1
        params = optimizer_func(params,grads,hyperparams,optimizer,L,t)

    return params

def cost_solver(FPcache,Y,params,hyperparams,model_architecture):

    (A, Z_list, D_list) = FPcache

    m = A[-1].shape[1]
    lambd_ = hyperparams["lambd_"]
    L = model_architecture["layer_count"]

    

    sum_params = 0

    for l in range(1,L):
        sum_params +=   np.sum(np.square(  params["W" + str(l)]  ))  

    L2_regularization = 1/m * lambd_/2 * sum_params

    cost = - 1/m * np.sum( np.dot(Y,np.log(A[-1].T)) + np.dot((1-Y),np.log(1-A[-1].T)) ) + L2_regularization

    return cost

def gd_method(Dataset,method="minibatch",mini_batch_size = 64):

    X = Dataset["X_train"]
    Y = Dataset["Y_train"]

    m = X.shape[1]

   

    if method == "minibatch":

        permutation = list(np.random.permutation(m))
        shuffle_X = X[:,permutation]
        shuffle_Y = Y[:,permutation]

        mini_batch_list = []

        mini_batch_total = math.floor(m/mini_batch_size)

        '''
        if m = 1279
        mini_batch_size = 64
        then mini_batch_total = 19.984375
        (use math floor to round of to 19 then process remainder)
        '''

        for i in range(mini_batch_total):

            mini_X = shuffle_X[:, (i*mini_batch_size) : (i+1)*mini_batch_size]
            mini_Y = shuffle_Y[:, (i*mini_batch_size) : (i+1)*mini_batch_size]

            mini_batch = (mini_X, mini_Y)
            mini_batch_list.append(mini_batch)
        
        # for the remainder
        if m % mini_batch_size != 0:
            mini_X = shuffle_X[:, ((i+1)*mini_batch_size) : ]
            mini_Y = shuffle_Y[:, ((i+1)*mini_batch_size) : ]

            mini_batch = (mini_X, mini_Y)
            mini_batch_list.append(mini_batch)


    return mini_batch_list