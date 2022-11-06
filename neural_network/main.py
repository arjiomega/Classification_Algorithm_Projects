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

    X_train = normalize(X_train)


    # Model Architecture (FIX: must be an input from user)
    model_architecture = {
        "layer_count": 5, # 4 layer + 1 input layer
        "node_count": [X_train.shape[0],20, 7, 5, 1],
        "activation_function": [None,relu,relu,relu,sigmoid]
    }

    hyperparams = {
        "learning_rate": 0.1,
        "num_iters": 100000,
        # for regularization
        "lambd_": 0.1,
        # for dropout
        "dropout_keep": 1.0,
        # for optimizer
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 10e-08
    }

    costs = []

    # initialize parameters
    params = initialize_params(model_architecture)

    for i in range(hyperparams["num_iters"]):

        # forward propagation
        FPcache = forward_propagation(X_train,params,model_architecture,hyperparams["dropout_keep"])

        #compute cost
        cost = cost_solver(FPcache,Y_train,params,hyperparams,model_architecture)

        # backward propagation
        grads = backward_propagation(params,Y_train,FPcache,model_architecture,hyperparams["lambd_"])
        
        # update params
        params = update_params(i,params,grads,hyperparams,model_architecture,optimizer="rmsprop") # optimizer="adam"
        
        
        costs.append(cost)
        if i % 10000 == 0:
            print(f"Cost after iteration {i}: {cost}")
            


    iteration_range = np.arange(0,hyperparams["num_iters"],1)

    plt.plot(iteration_range, costs)
  
    # naming the x axis
    plt.xlabel('Iteration')
    # naming the y axis
    plt.ylabel('Cost')

    #plt.ylim(0.6, 0.8) 
    
    # giving a title to my graph
    plt.title('Cost per iteration')
    
    # function to show the plot
    plt.show()

    return 1