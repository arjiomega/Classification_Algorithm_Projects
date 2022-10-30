from importlist import *
from functionlist import normalize

def run(data2train,class_solver):
    
    X, Y = csv2arrays.redWine(int(data2train))
    #X, Y = csv2arrays.tomAndJerry(int(data2train))

    X = normalize(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)


    solverList[class_solver](X_train,Y_train,X_test,Y_test)

    return 1