from importlist import *

def gradient_check(params,grads,X,Y,epsilon):

    theta = np.array(list(flatten(params.values())))
    d_theta = np.array(list(flatten(grads.values())))

    for i in range(theta.shape[0]):

    #d_theta_approx = 

    return 1