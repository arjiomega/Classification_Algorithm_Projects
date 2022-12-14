import pandas as pd
import numpy as np
import copy,math

import os
import sys
import time

#visualization
import matplotlib.pyplot as plt

# image to numpy array
from PIL import Image
from numpy import asarray

# store array
import h5py

#multiprocessing
import multiprocessing

# flatting arrays for gradient checking
from pandas.core.common import flatten

#scikit learn imports 
import sklearn
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
