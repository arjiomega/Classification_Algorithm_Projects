from re import M
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

    m = X_train.shape[1]

    # Model Architecture (FIX: must be an input from user) 
    # changed position temporarily for mini-batch
    # model_architecture = {
    #     "layer_count": 5, # 4 layer + 1 input layer
    #     "node_count": [X_train.shape[0],20, 7, 5, 1],
    #     "activation_function": [None,relu,relu,relu,sigmoid]
    # }

    hyperparams = {
        "learning_rate": 0.1,
        "num_epochs": 100000,
        # for regularization
        "lambd_": 0.1,
        # for dropout
        "dropout_keep": 0.8,
        # for optimizer
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 10e-08
    }

    costs = []

    # initialize parameters (temporarily removed for mini-batch)
    #params = initialize_params(model_architecture)



    


    for i in range(hyperparams["num_epochs"]):


        # for mini-batch only
        ## different combination of mini-batch and reset total cost
        mini_batch_list = gd_method(Dataset)
        cost_total = 0


        # for mini-batch only
        for mini_batch_index in range(len(mini_batch_list)):
 
            # mini-batch training set
            (X_train,Y_train) = mini_batch_list[mini_batch_index]

            # model architecture
            model_architecture = {
                "layer_count": 5, # 4 layer + 1 input layer
                "node_count": [X_train.shape[0],20, 7, 5, 1],
                "activation_function": [None,relu,relu,relu,sigmoid]
            }

            # initialize parameters
            params = initialize_params(model_architecture)


            # forward propagation
            FPcache = forward_propagation(X_train,params,model_architecture,hyperparams["dropout_keep"])

            #compute cost
            cost = cost_solver(FPcache,Y_train,params,hyperparams,model_architecture)

            ## for mini-batch only
            cost_total += cost


            # backward propagation
            grads = backward_propagation(params,Y_train,FPcache,model_architecture,hyperparams["lambd_"])
            
            # update params
            params = update_params(i,params,grads,hyperparams,model_architecture,optimizer="adam") # optimizer="adam"
            

        cost_avg = cost_total / m

        
        costs.append(cost_avg)

        if i % 1000 == 0:
            print(f"Cost after iteration {i}: {cost_avg}")
            


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