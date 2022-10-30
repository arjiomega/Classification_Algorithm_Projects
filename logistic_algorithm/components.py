import sys
sys.path.insert(0, '../components/')
from importlist import *

def sigmoid(z):

    s = 1 / (1+np.exp(-z))

    return s

def optimized_params(params, hyperparams, Dataset, print_cost = True):

    X_train = Dataset["X_train"]
    Y_train = Dataset["Y_train"]

    w = copy.deepcopy(params["w"])
    b = copy.deepcopy(params["b"])

    n,m = X_train.shape

    num_iters = hyperparams["num_iters"]
    learning_rate = hyperparams["learning_rate"]

    costs = []


    for i in range(num_iters):

        A = sigmoid(np.dot(w.T,X_train)+b)

        cost = - 1/m * np.sum(np.dot(Y_train,np.log(A.T))+np.dot((1-Y_train),np.log(1-A.T)))

        dJ_dw = 1/m * (np.dot(X_train,(A-Y_train).T))
        dJ_db = 1/m * np.sum(A-Y_train)

        w -= learning_rate*dJ_dw
        b -= learning_rate*dJ_db

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after iteration {i}: {cost}")


        params = {"w": w,
                  "b": b}

    return params


def predict(params, X, threshold = 0.5):
    
    w = params["w"]
    b = params["b"]

    n,m = X.shape

    A = sigmoid(np.dot(w.T,X)+b)

    Y_predict = (A > threshold).astype(int)

    return Y_predict

def performance(predictions,Dataset):

    predict_train = predictions["predict_train"]
    predict_test = predictions["predict_test"]

    Y_train = Dataset["Y_train"]
    Y_test = Dataset["Y_test"]

    print(f"\ntrain accuracy: { 100 - np.mean(np.abs(predict_train - Y_train)) * 100 }%")
    print(f"train accuracy: { 100 - np.mean(np.abs(predict_test - Y_test)) * 100 }%")

    return 1