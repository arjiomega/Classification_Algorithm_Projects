import sys



from components import *
from functionlist import initialize_parameters
from neural_network.components import backward_propagation, forward_propagation

sys.path.insert(0, '../components/')
from dataset_initialization import * 
from importlist import *


def nn_run(data2train):
    '''
    Dataset - library containing numpy arrays X_train,Y_train,X_test,Y_test
    X (n,m) || Y (1,m)
    '''

    Dataset = initialize_dataset(int(data2train))

    X_train = Dataset["X_train"]
    Y_train = Dataset["Y_train"]
    X_test = Dataset["X_test"]
    Y_test = Dataset["Y_test"]

    # Model Architecture (FIX: must be an input from user)
    model_architecture = {
        "layer_count": 5, # 4 layer + 1 input layer
        "node_count": [X_train.shape[0],20, 7, 5, 1],
        "activation_function": [None,relu,relu,relu,sigmoid]
    }

    # initialize parameters
    params = initialize_params(model_architecture)
    
    # forward propagation
    forward_propagation(X_train,params,model_architecture)

    # backward propagation
    backward_propagation(model_architecture)
    return 1