from functionlist import *



class solver:
    def log_alg_np():
        return 1
    def log_alg_npv(X_train,Y_train,X_test,Y_test):
        print("-------------------------------------------------\n")
        print("\nUsing Logistic Algorithm using Numpy (Vectorized)\n")

        learning_rate = input("learning rate (recommended 0.01 below for stability): ")
        num_iterations = input("\nnumber of iterations: ")
        print_cost = input("\nprint cost? (leave blank and press enter if not): ")
        print_accuracy = input("\nprint accuracy? (leave blank and press enter if not): ")

        log_alg(X_train,Y_train,X_test,Y_test, int(num_iterations),float(learning_rate),bool(print_cost),bool(print_accuracy))

        return 1

    def log_alg_sk():
        return 1
    def log_alg_tf():
        return 1

solverList = [solver.log_alg_np, solver.log_alg_npv, solver.log_alg_sk, solver.log_alg_tf]

def run(data2train,class_solver):
    
    X, Y = csv2arrays.redWine(int(data2train))
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

    solverList[class_solver](X_train,Y_train,X_test,Y_test)

    return 1