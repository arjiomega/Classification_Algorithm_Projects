from importlist import *
from ml_algorithms import *
from functionlist import *

print("-------------------------------------------------\n")

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
#     \n")
#class_solver = input("Input: ")

data2train = "0"
class_solver = "4"

run(int(data2train),int(class_solver))






