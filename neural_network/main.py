from re import M
import sys



from components import *
from functionlist import initialize_parameters
from neural_network.components import backward_propagation, forward_propagation, initialize_params

sys.path.insert(0, '../components/')
from dataset_initialization import * 
from importlist import *


def nn_run(data2train): # X (n,m) Y (1,m)
    '''
    Dataset - library containing numpy arrays X_train,Y_train,X_test,Y_test
    X (n,m) || Y (1,m)
    '''

    Dataset = initialize_dataset(int(data2train))

    X_train = Dataset["X_train"]
    Y_train = Dataset["Y_train"]
    #X_test = Dataset["X_test"]
    #Y_test = Dataset["Y_test"]

    #X_train = normalize(X_train)

    print(X_train.shape,Y_train.shape)

    m = X_train.shape[1]

    # Model Architecture (FIX: must be an input from user) 
    model_architecture = {
        "layer_count": 5, # 4 layer + 1 input layer
        "node_count": [X_train.shape[0],20, 7, 5,Y_train.shape[0]],
        "activation_function": [None,relu,relu,relu,sigmoid]
    }

    hyperparams = {
        "learning_rate": 0.0001,
        "num_epochs": 50000,
        # for regularization
        "lambd_": 0.1,
        # for dropout
        "dropout_keep": 1.0,
        # for optimizer
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 10e-08,
        # learning decay
        "learning_decay": False,
        "decay_schedule": False,
        "decay_rate": 0.3,
        "decay_interval": 5000
    }

    costs = []

    mini_batch_params = []

    mini_batch_list = gd_method(Dataset)
    print("mini batch prep done!")

    params = initialize_params(model_architecture)

    for epoch in range(hyperparams["num_epochs"]):



        mini_batch_list = gd_method(Dataset)
        cost_total = 0
        #mini_batch_list = [(X_train,Y_train)] # batch gd


        for mini_batch_index in range(len(mini_batch_list)):

            X_train, Y_train =  mini_batch_list[mini_batch_index]

            # train model
            #print(X_train.shape,Y_train.shape)
            cost, params = train(X_train,Y_train, params, hyperparams,model_architecture,epoch)
            cost_total += cost

        # learning_rate_decay
        if hyperparams["learning_decay"]:
            hyperparams["learning_rate"] = learning_decay(hyperparams,epoch)
            
            

        cost_avg = cost_total / m

        
        costs.append(cost_avg)

        

        #if epoch % 1000 == 0:
        print(f"Cost after iteration {epoch}: {cost_avg}")
            #print("learning rate: ",hyperparams["learning_rate"])
            
            


    iteration_range = np.arange(0,hyperparams["num_epochs"],1)

    plt.plot(iteration_range, costs)
  

    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost per iteration')
    plt.show()

    return 1