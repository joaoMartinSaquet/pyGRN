import numpy as np
from numba import types, njit, jit
from numba.experimental import jitclass
from copy import deepcopy
from .base import GRN
from loguru import logger

spec = [
    ('num_input', types.int64),
    ('num_output', types.int64),
    ('num_regulatory', types.int64),
    ('beta', types.float64),
    ('delta', types.float64),
    ('dt', types.float64),
    ('idsize', types.int64),
    ('concentration', types.float64[:]),  # 1D array of floats
    ('next_concentration', types.float64[:]),  # 1D array of floats
    ('enhance_match', types.float64[:, :]),  # 2D array of floats
    ('inhibit_match', types.float64[:, :]),  # 2D array of floats 
    ('inhibitors',  types.float64[:]),  # dynamic list of ints
    ('enhancers',  types.float64[:]),   # dynamic list of ints
    ('identifiers',  types.float64[:]),       # Array of strings 
]
# @jitclass(spec)
class DissetGRN(GRN):
    """
        Disset CPS grn implementation
        [1] J. Disset, D. Wilson, S. Cussat-Blanc, S. Sanchez, H. Luga, and Y. Duthen, “A Comparison of Genetic Regulatory Network Dynamics and Encoding,” in GECCO ’17 Proceedings of the Genetic and Evolutionary Computation Conference, Berlin, Germany: ACM, July 2017, pp. 91–98. doi: 10.1145/3071178.3071322.

        encoding of protein tag is  [0,1]
        and minmax step 

    """
    inhibitors = []
    enhancers = []
    identifiers = []

    num_input = 0
    num_output = 0
    num_regulatory = 0

    beta = 0.5
    delta = 0.5

    dt = 1

    idsize = 0
    concentration = []
    next_concentration = []
    enhance_match = []
    inhibit_match = []
    idsize = 1

    
    def __init__(self):
        self.idsize = 1
        # Initialize with default sizes
        self.num_input = 1
        self.num_output = 1
        self.num_regulatory = 0
        self.idsize = 1
        self.concentration = np.zeros(1, dtype=np.float64)
        self.next_concentration = np.zeros(1, dtype=np.float64)
        self.enhance_match = np.zeros((1, 1), dtype=np.float64)
        self.inhibit_match = np.zeros((1, 1), dtype=np.float64)
        self.inhibitors = np.zeros(1, dtype=np.float64)
        self.enhancers = np.zeros(1, dtype=np.float64)
        self.identifiers = np.zeros(1, dtype=np.float64)
        self.beta = 0.5
        self.delta = 0.5
        self.dt = 1.0
    

    def reset(self):
        self.concentration = np.ones(
            len(self.identifiers)) * (1.0/len(self.identifiers))
        self.next_concentration = np.zeros(len(self.identifiers))
        return self

    def warmup(self, nsteps):
        self.set_input(np.zeros(self.num_input))
        for _ in range(nsteps):
            self.step()
    

    def setup(self):


        self.inhibit_match = np.zeros(
            [len(self.identifiers), len(self.identifiers)])
        self.enhance_match = np.zeros(
            [len(self.identifiers), len(self.identifiers)])
        # get uij
        for k in range(len(self.identifiers)):
            for j in range(len(self.identifiers)):
                self.enhance_match[k, j] = np.abs(
                    self.enhancers[k] - self.identifiers[j])
                self.inhibit_match[k, j] = np.abs(
                    self.inhibitors[k] - self.identifiers[j])


        enh_max = np.max(self.enhance_match)
        inh_max = np.max(self.inhibit_match)

        # we compute it with beta outside 
        for k in range(len(self.identifiers)):
            for j in range(len(self.identifiers)):

                # affinity computation
                # # a = 0
                # enhance_affinity = - self.beta * self.enhance_match[k, j]
                # inhibit_affinity = - self.beta * self.inhibit_match[k, j]

                # a = 1
                enhance_affinity = - self.beta *(self.idsize - self.enhance_match[k, j]) - enh_max
                inhibit_affinity = - self.beta *(self.idsize - self.inhibit_match[k, j]) - inh_max

                # # a = 2
                # enhance_affinity = - self.beta *(self.idsize - self.enhance_match[k, j]- enh_max) 
                # inhibit_affinity = - self.beta *(self.idsize - self.inhibit_match[k, j]- inh_max) 
                
                # # influence function 
                # # f = 0
                # self.enhance_match[k, j] = np.exp(enhance_affinity)
                # self.inhibit_match[k, j] = np.exp(inhibit_affinity)

                # f = 1
                self.enhance_match[k, j] =  np.tanh(enhance_affinity) + 1
                self.inhibit_match[k, j] =  np.tanh(inhibit_affinity) + 1
               
                # # # f = 2
                # self.enhance_match[k, j] =  np.tanh(enhance_affinity/2) + 1
                # self.inhibit_match[k, j] =  np.tanh(inhibit_affinity/2) + 1 # divide by 2 is equivalent to 2/(1+exp(-x))
                
                

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


    # @njit
    def step(self, nsteps=1):
        """Runs the GRN concentration update"""
        

        self.concentration = step(self.concentration, self.enhance_match, self.inhibit_match, self.delta, self.num_input, self.num_output)
#         for _ in range(nsteps):
#             # self.concentration = step(self.concentration, self.enhance_match, self.inhibit_match, self.delta, self.num_input, self.num_output)
#             if len(self.next_concentration) != len(self.concentration):
#                 self.next_concentration = np.zeros(len(self.concentration))
#             # 
#             sum_concentration = 0.0
#             for k in range(len(self.identifiers)):
# # 
#                 # if the proteins is an input proteins she is not impacted by the network
#                 if k < self.num_input:
#                     self.next_concentration[k] = self.concentration[k]
#                 else:
#                     # we compute the concentration change 
#                     enhance = 0.0
#                     inhibit = 0.0
#                     for j in range(len(self.identifiers)):
#                         # if the proteins is not an ouptut  because it does not regulate the network we compute hk and gk for all execpt the output 
#                         if j < self.num_input or j >= (self.num_output + self.num_input):
#                             enhance += (self.concentration[j] *
#                                         self.enhance_match[j, k])
#                             inhibit += (self.concentration[j] *
#                                         self.inhibit_match[j, k])
#                                     #   
#                     diff = self.delta  * (enhance - inhibit) / len(self.identifiers) 
# # 
#                     # ensuring that the concentrations are between 0 and 1
#                     next_concentration = min(self.concentration[k] + diff, 1.0)
#                     self.next_concentration[k] = max(0.0, next_concentration)
#                     sum_concentration += self.next_concentration[k]
# # 
# # 
# # 
#             # if sum_concentration > 0:
#             #     for k in range(len(self.identifiers)):
#             #         if k >= self.num_input:
#             #             self.next_concentration[k] = min(
#             #                 1.0, self.next_concentration[k] / sum_concentration)
# # 
#             # logger.debug("Concentration: {},  \t next_concentration: {}".format(self.concentration, self.next_concentration))
#             self.concentration = self.next_concentration
        return self

    def clone(self):
        return deepcopy(self)




# @njit
# def step(concentrations, enhance_match, inhibit_match, delta, num_input, num_output):


#     # next_concentration = np.zeros(len(concentrations), dtype=np.float64)
#     next_concentration = np.zeros_like(concentrations, dtype=np.float64)
#     sum_concentration = 0.0

#     for k in range(len(concentrations)):
#         if k < num_input:
#             next_concentration[k] = concentrations[k]
#         else:
#             enhance = 0.0
#             inhibit = 0.0
#             for j in range(len(concentrations)):
#                 if j < num_input or j >= (num_output + num_input):
#                     enhance += (concentrations[j] * enhance_match[j, k])
#                     inhibit += (concentrations[j] * inhibit_match[j, k])
#             diff = delta * (enhance - inhibit) / len(concentrations)

#         # ensuring that the concentrations are between 0 and 1
#         next_concentration = min(concentrations[k] + diff, 1.0)
#         next_concentration[k] = max(0.0, next_concentration)
#         sum_concentration += next_concentration[k]
    
    
#     return next_concentration


@njit
def step(concentrations, enhance_match, inhibit_match, delta, num_input, num_output):
    # Initialize next concentration array
    next_concentration = np.zeros_like(concentrations, dtype=np.float64)
    total_size = len(concentrations)

    for k in range(total_size):
        if k < num_input:
            # Input nodes - concentration remains unchanged
            next_concentration[k] = concentrations[k]
        else:
            # Calculate regulatory effects
            enhance = 0.0
            inhibit = 0.0

            for j in range(total_size):
                # Only consider non-output nodes for regulation
                if j < num_input or j >= (num_input + num_output):
                    enhance += concentrations[j] * enhance_match[j, k]
                    inhibit += concentrations[j] * inhibit_match[j, k]

            # Calculate concentration change
            diff = delta * (enhance - inhibit) / total_size
            new_concentration = concentrations[k] + diff

            # Ensure concentration stays within bounds
            if new_concentration > 1.0:
                next_concentration[k] = 1.0
            elif new_concentration < 0.0:
                next_concentration[k] = 0.0
            else:
                next_concentration[k] = new_concentration
    return next_concentration