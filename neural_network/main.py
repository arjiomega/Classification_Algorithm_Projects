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
        "learning_rate": 0.01,
        "num_iters": 25000,
        "lambd_": 0.1
    }

    costs = []

    for i in range(hyperparams["num_iters"]):

        # initialize parameters
        params = initialize_params(model_architecture)
        
        # forward propagation
        A_list, Z_list = forward_propagation(X_train,params,model_architecture)

        # backward propagation
        grads = backward_propagation(params,Y_train,A_list,Z_list,model_architecture,hyperparams["lambd_"])
        
        # update params
        params = update_params(params,grads,hyperparams["learning_rate"],model_architecture)
        
        #compute cost
        cost = cost_solver(A_list,Y_train,params,hyperparams,model_architecture)
        costs.append(cost)
        if i % 1000 == 0:
            print(f"Cost after iteration {i}: {cost}")
            


    iteration_range = np.arange(0,hyperparams["num_iters"],1)

    plt.plot(iteration_range, costs)
  
    # naming the x axis
    plt.xlabel('Iteration')
    # naming the y axis
    plt.ylabel('Cost')

    plt.ylim(0.6, 0.8) 
    
    # giving a title to my graph
    plt.title('Cost per iteration')
    
    # function to show the plot
    plt.show()

    return 1