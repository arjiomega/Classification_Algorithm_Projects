from importlist import *




DataList = ["data/winequality-red.csv"]

class csv2arrays:

    def redWine(data2train):
        data = pd.read_csv(DataList[data2train])

        # replace quality data
        data['quality'].replace(to_replace={3:0,4:0,5:0,6:0,7:1,8:1}, inplace = True)

        X = data[['fixed acidity',
                'volatile acidity',
                'citric acid',
                'residual sugar',
                'chlorides',
                'free sulfur dioxide',
                'total sulfur dioxide',
                'density',
                'pH',
                'sulphates',
                'alcohol']]

        Y = data['quality']

        X_train = X.values
        Y_train = Y.values

        return X_train,Y_train


def sigmoid(z):
    '''
    vars:
	m = number of training examples
	n = number of features for every training example
    args:
    z (1,m)
    return:
    s (1,m)
    '''

    s = 1/(1+np.exp(-z))


    return s

def propagate(w,b,X,Y):
    '''
    vars:
    m = number of training examples
    n = number of features for every training example
    A (1,m) = array of sigmoid of (w.T X + b) for each training example
    dw (n,1)
    db ()
    Argument:
    w (n,1)
    b ()
    X (m,n)
    Y (m,) => convert rank 1 array to rank 2 array Y_reshape(m,1)

    Return:
    grads {"dw": dw, "db": db}
    cost ()
    '''
    m, n = X.shape

    A = sigmoid(np.dot(w.T,X.T)+b)

    Y_reshape = np.reshape(Y, (-1,1))

    cost = - 1/m * np.sum(np.dot(Y_reshape.T,np.log(A.T))+np.dot((1-Y_reshape.T),np.log(1-A.T)))

    dw = 1/m * (np.dot(X.T,(A.T-Y_reshape)))
    db = 1/m * np.sum(A.T-Y_reshape)

    grads = {"dw": dw,
                "db": db}

    return grads, cost


def predict(w,b,X):
	'''
	vars:
	m = number of training examples
	n = number of features for every training example
	A (1,m) = array of sigmoid of (w.T X + b) for each training example
	funcs:
    sigmoid() sigmoid function
	args:
	w (n,1)
	b ()
	X (m,n)

	return:
	Y_prediction (m,1) 

	'''
	m,n = X.shape
	Y_prediction = np.zeros((m,1))

	A = sigmoid(np.dot(w.T,X.T)+b)

	for i in range(A.shape[1]):
		if A[0,i] > 0.5:
			Y_prediction[i,0] = 1
		else:
			Y_prediction[i,0] = 0
	
	return Y_prediction


def optimize(w,b,X,Y,num_iterations=100,learning_rate=0.009,print_cost=False):
    '''
    var:
    m = number of training examples
	n = number of features for every training example
    dw (n,1)
    db ()
    args:
    w (n,1)
    b ()
    return:
    params {"w": w, "b": b}
    grads {"dw": dw, "db": db}
    costs []
    '''
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w,b,X,Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db



        if i % 100 == 0:
            costs.append(cost)
        
            # Print the cost every 100 training iterations
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
                "b": b}
                
    grads = {"dw": dw,
                "db": db}


    return params, grads, costs

def accuracy(Y_prediction_train,Y_prediction_test,Y_train,Y_test):
    Y_train = np.reshape(Y_train, (-1,1))
    Y_test = np.reshape(Y_test, (-1,1))
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    return 1

def log_alg(X_train,Y_train,X_test,Y_test, num_iterations=2000,learning_rate=0.5,print_cost=False,print_accuracy=False):
    '''
    vars:
    w (n,1)
    b ()
    m = number of training examples
	n = number of features for every training example
    funcs:
    optimize() gradient descent for finding best w,b and cost 
    predict() converts training results to 1s and 0s
    args:
    X_train, X_test (m,n)
    Y_train, Y_test (1279,) => convert rank 1 array to rank 2 array Y_reshape(m,1)
    return:

    d {"costs": costs,
            "Y_prediction_test": Y_prediction_test, 
            "Y_prediction_train" : Y_prediction_train, 
            "w" : w, 
            "b" : b,
            "learning_rate" : learning_rate,
            "num_iterations": num_iterations}
    '''
    m,n = X_train.shape


    w = np.zeros((n,1))
    b = 0.

    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w =params["w"]
    b =params["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    if print_accuracy:
        accuracy(Y_prediction_train,Y_prediction_test,Y_train,Y_test)
        
    d = {"costs": costs,
            "Y_prediction_test": Y_prediction_test, 
            "Y_prediction_train" : Y_prediction_train, 
            "w" : w, 
            "b" : b,
            "learning_rate" : learning_rate,
            "num_iterations": num_iterations}

    return d