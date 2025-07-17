import numpy as np
import numba

from copy import deepcopy
from .base import GRN
from loguru import logger

class ClassicGRN(GRN):
    """Classic CPU-based GRN

    Dynamics equations are written mostly in loop form
    input index is 0:n_inputs
    output id is n_inputs : n_output 
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
        for i in range(nsteps):
            self.step()

    def setup(self):
        """ IDSIZE is missing here NO ? """
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

        for k in range(len(self.identifiers)):
            for j in range(len(self.identifiers)):
                self.enhance_match[k, j] = np.exp(
                    - self.beta * self.enhance_match[k, j])
                self.inhibit_match[k, j] = np.exp(
                    - self.beta * self.inhibit_match[k, j])

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
            self.num_output + self.num_input)]


    # @numba.jit(nopython=True)
    def step(self, nsteps=1):
        """Runs the GRN concentration update"""
        


        for step in range(nsteps):
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
                    # logger.debug("enhance: {}, inhibit: {}, diff: {}".format(enhance, inhibit, diff))
                    # logger.debug("enhance match: {}".format(self.enhance_match))
                    
                    self.next_concentration[k] = max(0.0,
                                                    self.concentration[k] + diff)
                    sum_concentration += self.next_concentration[k]
            if sum_concentration > 0:
                for k in range(len(self.identifiers)):
                    if k >= self.num_input:
                        self.next_concentration[k] = min(
                            1.0, self.next_concentration[k] / sum_concentration)

            # logger.debug("Concentration: {},  \t next_concentration: {}".format(self.concentration, self.next_concentration))
            self.concentration = self.next_concentration
        return self

    def clone(self):
        return deepcopy(self)
