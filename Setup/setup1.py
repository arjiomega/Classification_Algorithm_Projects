



def setup(Dataset):

    X_train = Dataset["X_train"]

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


    model_architecture = {
                    "layer_count": 5, # 4 layer + 1 input layer
                    "node_count": [X_train.shape[0],20, 7, 5, 1],
                    "activation_function": [None,relu,relu,relu,sigmoid]
                }
    

    
    return hyperparams, model_architecture