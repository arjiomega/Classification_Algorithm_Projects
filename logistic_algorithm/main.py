import sys

sys.path.insert(0, '../')


from components import *

sys.path.insert(0, 'components/')
from dataset_initialization import * 
from importlist import *

def log_alg_run(data2train):
    '''
    
    args:
    Dataset - library containing numpy arrays X_train,Y_train,X_test,Y_test
    X (n,m) || Y (1,m)
    '''
    
    # print("-------------------------------------------------\n")
    # print("\nUsing Logistic Algorithm using Numpy (Vectorized)\n")

    # learning_rate = input("learning rate (recommended 0.01 below for stability): ")
    # num_iterations = input("\nnumber of iterations: ")
    # print_cost = input("\nprint cost? (leave blank and press enter if not): ")
    # print_accuracy = input("\nprint accuracy? (leave blank and press enter if not): ")

    #print("\n-------------------------------------------------\n")

    Dataset = initialize_dataset(int(data2train))

    learning_rate = 0.01
    num_iters = 2500
    print_cost = ""
    print_accuracy = "yes"

    X_train = Dataset["X_train"]
    Y_train = Dataset["Y_train"]
    X_test = Dataset["X_test"]
    Y_test = Dataset["Y_test"]



    n,m = X_train.shape

    w = np.zeros((n,1))
    b = 0.

    params = {
        "w": w,
        "b": b
    }
    hyperparams = {
        "num_iters": num_iters,
        "learning_rate": learning_rate
    }

    # find optimized w and b by gradient descent
    params = optimized_params(params,hyperparams,Dataset)

    # turn solved values into 1s and 0s
    predictions = {
        "predict_train":predict(params,X_train),
        "predict_test":predict(params,X_test)
        }


    performance(predictions,Dataset)

    
    return 1
