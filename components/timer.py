from importlist import *

def timer(function,data2train):

    t_start = time.time()

    exec("function(data2train)")

    print("\nrun time: ",time.time() - t_start, "s")