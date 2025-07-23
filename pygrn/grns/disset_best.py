import numpy as np
import numba

from copy import deepcopy
from .base import GRN
from loguru import logger

class DissetGRN(GRN):
    """
        Disset CPS grn implementation
        [1] J. Disset, D. Wilson, S. Cussat-Blanc, S. Sanchez, H. Luga, and Y. Duthen, “A Comparison of Genetic Regulatory Network Dynamics and Encoding,” in GECCO ’17 Proceedings of the Genetic and Evolutionary Computation Conference, Berlin, Germany: ACM, July 2017, pp. 91–98. doi: 10.1145/3071178.3071322.

        encoding of protein tag is  [0,1]
        and minmax step 

    """
    concentration = []
    next_concentration = []
    enhance_match = []
    inhibit_match = []
    

    
    def __init__(self):
        pass
    
    

    def reset(self):
        self.concentration = np.ones(
            len(self.identifiers)) * (1.0/len(self.identifiers))
        self.next_concentration = np.zeros(len(self.identifiers))
        return self

    def warmup(self, nsteps):
        self.set_input(np.zeros(self.num_input))
        for _ in range(nsteps):
            self.step()
    

    # @numba.jit(nopython=True)
    def setup(self):
        self.inhibit_match = np.zeros(
            [len(self.identifiers), len(self.identifiers)])
        self.enhance_match = np.zeros(
            [len(self.identifiers), len(self.identifiers)])
        for k in range(len(self.identifiers)):
            for j in range(len(self.identifiers)):
                self.enhance_match[k, j] = self.idsize - np.abs(
                    self.enhancers[k] - self.identifiers[j])
                self.inhibit_match[k, j] = self.idsize - np.abs(
                    self.inhibitors[k] - self.identifiers[j])

        enh_max = np.max(self.enhance_match)
        inh_max = np.max(self.inhibit_match)

        # we compute it with beta outside 
        for k in range(len(self.identifiers)):
            for j in range(len(self.identifiers)):
                self.enhance_match[k, j] = np.exp(self.beta * self.enhance_match[k, j] - enh_max) #
                self.inhibit_match[k, j] = np.exp(self.beta * self.inhibit_match[k, j] - inh_max) # a = 1 
                # self.enhance_match[k, j] = np.exp(- self.beta * self.enhance_match[k, j]) #
                # self.inhibit_match[k, j] = np.exp(- self.beta * self.inhibit_match[k, j]) # a = 0 
                

        self.reset()

    def get_signatures(self):
        return self.enhance_match - self.inhibit_match

    def get_concentrations(self):
        return self.concentration

    def set_input(self, inputs):
        self.concentration[0:self.num_input] = inputs
        return self

    def get_output(self):
        return self.concentration[self.num_input:(
            self.num_output + self.num_input)].copy()


    # @numba.jit(nopython=True)
    def step(self, nsteps=1):
        """Runs the GRN concentration update"""
        


        for _ in range(nsteps):
            if len(self.next_concentration) != len(self.concentration):
                self.next_concentration = np.zeros(len(self.concentration))
            
            sum_concentration = 0.0
            for k in range(len(self.identifiers)):

                # if the proteins is an input proteins she is not impacted by the network
                if k < self.num_input:
                    self.next_concentration[k] = self.concentration[k]
                else:
                    # we compute the concentration change 
                    enhance = 0.0
                    inhibit = 0.0
                    for j in range(len(self.identifiers)):
                        # if the proteins is not an ouptut  because it does not regulate the network we compute hk and gk for all execpt the output 
                        if j < self.num_input or j >= (self.num_output + self.num_input):
                            enhance += (self.concentration[j] *
                                        self.enhance_match[j, k])
                            inhibit += (self.concentration[j] *
                                        self.inhibit_match[j, k])
                                      
                    diff = self.delta  * (enhance - inhibit) / len(self.identifiers) 

                    # ensuring that the concentrations are between 0 and 1
                    next_concentration = min(self.concentration[k] + diff, 1.0)
                    self.next_concentration[k] = max(0.0, next_concentration)
                    sum_concentration += self.next_concentration[k]



            # if sum_concentration > 0:
            #     for k in range(len(self.identifiers)):
            #         if k >= self.num_input:
            #             self.next_concentration[k] = min(
            #                 1.0, self.next_concentration[k] / sum_concentration)

            # logger.debug("Concentration: {},  \t next_concentration: {}".format(self.concentration, self.next_concentration))
            self.concentration = self.next_concentration
        return self

    def clone(self):
        return deepcopy(self)
