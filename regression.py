from pygrn import grns, problems, evolution, config
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def f(t, f: float = 0.5, k: int = 2):
    """ Fourrier decomposition 

    Args:
        t (_type_): time
        k (int, optional): degree of decomposition. Defaults to 2.

    Returns:
        values
    """
    y = 0
    for i in range(0, k):

        y += np.sin((2*i + 1) * 2*np.pi*f*t)/(2*i + 1)
    
    
    y /= (4/np.pi)
    return y

class MyRegression(problems.base.Problem):
    def __init__(self, x_train, y_train):
        super().__init__("regression")
        self.namestr = "regression"
        self.nin = 1
        self.nout = 1

        self.x_train = x_train
        self.y_train = y_train

        
    def eval(self, grn):

        grn.setup()
        grn.warmup(25)
        fit = 0.0
        for i in range(max(self.x_train.shape)):
            grn.set_input(self.x_train[i])
            grn.step()
            fit += np.abs(grn.get_output() - self.y_train[i]).item()

        return -fit

def main():



    t = np.linspace(-5, 5, 500)
    y = f(t)
    

    grn = lambda : grns.ClassicGRN()
    problem = MyRegression(t, y)


    grneat = evolution.Evolution(problem, grn)
    grneat.run(100)
    
    # problem.eval(grneat.best_grn)
    
    # y_eval = f(t_eval)

    plt.plot(t, y)
    plt.show()



if __name__ == "__main__":
    main()


