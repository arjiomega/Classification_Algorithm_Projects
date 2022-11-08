from ml_algorithms import *
from functionlist import *

# doesnt like to have same file name
# sys.path.insert(0, 'logistic_algorithm/')
# from main import *

sys.path.insert(0, 'neural_network/')
from main import *

sys.path.insert(0, 'components/')
from timer import *
from importlist import *


#print("-------------------------------------------------\n")

# print("\nchoose data to train \n\
#     0: Red Wine Quality \n\
#     1: \n \
#     \n")
#data2train = input("Input: ")

# print("\nChoose classification algorithm solver \n\
#     0: Logistic Algorithm Numpy (Non-Vectorized) \n\
#     1: Logistic Algorithm Numpy (Vectorized) \n\
#     2: Logistic Algorithm (SciKit) \n\
#     3: Logistic Algorithm (TensorFlow) \n\
#     4: Neural Network
#     \n")
#class_solver = input("Input: ")

data2train = "0"
#class_solver = "1"

#timer(log_alg_run,data2train)

timer(nn_run,data2train)

