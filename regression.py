from pygrn import grns, problems, evolution, config
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

def f(t, f: float = 1, k: int = 2):
    """ Fourrier decomposition 

    Args:
        t (_type_): time
        k (int, optional): degree of decomposition. Defaults to 2.

    Returns:
        values
    """
    y = np.zeros(t.shape[0])
    for i in range(0, k):

        y += np.sin((2*i + 1) * 2*np.pi*f*t)/(2*i + 1)
    
    
    y /= (4/np.pi)
    
    # transform values between 0 and 1
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    
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
        grn.warmup(10)
        fit = 0.0
        for i in range(max(self.x_train.shape)):
            grn.set_input(self.x_train[i])
            grn.step()
            fit += np.linalg.norm(grn.get_output() - self.y_train[i]).item() / max(self.x_train.shape)

        return 1-fit

def main():



    t = np.linspace(0, 1, 100)
    y = f(t)
    

    grn = lambda : grns.ClassicGRN()
    problem = MyRegression(t, y)


    grneat = evolution.Evolution(problem, grn)
    best_fit, best_ind = grneat.run(1000)

    best_fit_history = grneat.best_fit_history
    logger.info("best fit: ", best_fit)
    # problem.eval(grneat.best_grn)
    
    # y_eval = f(t_eval)
    best_grn = best_ind.grn

    best_grn.setup()
    best_grn.warmup(25)
    y_eval = []
    for i in range(max(t.shape)):
        best_grn.set_input(t[i])
        best_grn.step()
        y_eval.append(best_grn.get_output())


    plt.plot(t, y_eval, label="prediction")
    plt.plot(t, y, label="target")

    plt.legend()
    plt.figure(figsize=(10, 5))
    plt.plot(best_fit_history)
    plt.show()



if __name__ == "__main__":
    main()


