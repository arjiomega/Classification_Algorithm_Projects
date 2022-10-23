from functionlist import *



class solver:
    def log_alg_np():
        return 1
    def log_alg_npv(X_train,Y_train,X_test,Y_test):
        print("-------------------------------------------------\n")
        print("\nUsing Logistic Algorithm using Numpy (Vectorized)\n")

        # learning_rate = input("learning rate (recommended 0.01 below for stability): ")
        # num_iterations = input("\nnumber of iterations: ")
        # print_cost = input("\nprint cost? (leave blank and press enter if not): ")
        # print_accuracy = input("\nprint accuracy? (leave blank and press enter if not): ")

        print("\n-------------------------------------------------\n")

        learning_rate = 0.01
        num_iterations = 2500
        print_cost = ""
        print_accuracy = "yes"


        t_start = time.time()

        log_alg(X_train,Y_train,X_test,Y_test, int(num_iterations),float(learning_rate),bool(print_cost),bool(print_accuracy))
        
        print("\nrun time: ",time.time() - t_start, "s")
        
        return 1

    def log_alg_sk():
        return 1
    def log_alg_tf():
        return 1

    def nn(X_train,Y_train,X_test,Y_test):
        n_x = X_train.shape[1]     # For image: num_px * num_px * 3
        n_h = 7
        n_y = 1
        layers_dims = (n_x, n_h, n_y)
        learning_rate = 0.0075
        #X = 1599,11 ; Y = 1599,
        X_train = X_train.T
        Y_train = (np.reshape(Y_train,(-1,1))).T

        parameters, costs = two_layer_model(X_train, Y_train, layers_dims, num_iterations = 2500, print_cost=True)

        plot_costs(costs, learning_rate)

        return 1

solverList = [solver.log_alg_np, solver.log_alg_npv, solver.log_alg_sk, solver.log_alg_tf,solver.nn]

def run(data2train,class_solver):
    
    X, Y = csv2arrays.redWine(int(data2train))
    print(X.shape,Y.shape)
    X = normalize(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

    solverList[class_solver](X_train,Y_train,X_test,Y_test)

    return 1