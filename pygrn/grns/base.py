from pygrn import config
import numpy as np
import abc
import json


class GRN(abc.ABC):
    """Abstract class for GRNs

    Implements methods related to the protein encoding of a GRN, but requires
    subclasses to implement all methods related to dynamics.

    # where is the IDsize ? 
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

    idsize = 32 # TODO put it in config ? 

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self):
        """Resets concentrations and any other state variables"""
        pass

    @abc.abstractmethod
    def setup(self):
        """Calculates signatures and sets initial concentrations

        Requires that all genome variables have been set (inhibitors,
        enhancers, identifiers,num_input, num_output, num_regulatory)
        """
        pass

    @abc.abstractmethod
    def set_input(self, inputs):
        """Set the concentration of input GRNs to the provided array

        Requires that the input numpy array is bounded in [0, 1]
        """
        pass

    @abc.abstractmethod
    def step(self):
        """Runs the GRN concentration update"""
        pass

    @abc.abstractmethod
    def get_output(self):
        """Outputs a numpy array of concentrations of the output proteins"""
        pass

    @abc.abstractmethod
    def clone(self):
        """Creates a new GRN of the same type as the subclass

        Must safely pass all necessary genomic data and setup the new GRN
        """
        pass

    def __str__(self):
        return json.dumps({'num_input': int(self.num_input),
                           'num_output': int(self.num_output),
                           'num_regulatory': int(self.num_regulatory),
                           'ids': self.identifiers.tolist(),
                           'enh': self.enhancers.tolist(),
                           'inh': self.inhibitors.tolist(),
                           'beta': self.beta,
                           'delta': self.delta})

    def from_str(self, grn_str):
        """
            Read the GRN from json string 
        """
        g_dict = json.loads(grn_str)
        self.num_input = g_dict['num_input']
        self.num_output = g_dict['num_output']
        self.num_regulatory = g_dict['num_regulatory']
        self.identifiers = np.array(g_dict['ids'])
        self.enhancers = np.array(g_dict['enh'])
        self.inhibitors = np.array(g_dict['inh'])
        self.beta = g_dict['beta']
        self.delta = g_dict['delta']

    def random(self, num_input, num_output, num_regulatory):
        """Sets the GRNs genome data randomly based on input sizes"""
        grn_size = num_input + num_output + num_regulatory

        self.num_input = num_input
        self.num_output = num_output
        self.num_regulatory = num_regulatory

        self.inhibitors = np.random.random([grn_size])
        self.enhancers = np.random.random([grn_size])
        self.identifiers = np.random.random([grn_size])
        self.beta = (np.random.random() *
                     (config.BETA_MAX - config.BETA_MIN) +
                     config.BETA_MIN)
        self.delta = (np.random.random() *
                      (config.DELTA_MAX - config.DELTA_MIN) +
                      config.DELTA_MIN)
        return self

    def size(self):
        return len(self.identifiers)

    def protein_distance(self, other, k, j):
        """
            Compute the distance between proteins k of the parent genome and protein i of the other genome

        This distance is composed of three parts:
        - The difference in the identifiers (ID_COEF)
        - The difference in the inhibitors (INH_COEF)
        - The difference in the enhancers (ENH_COEF)

        The distance is computed as:
        TODO small questions, here in papers [1] they a*abs(tag_A - idB) and divide the whole things by p (IDsize in [1] github)
        distance = a * abs(identifiers[k] - identifiers[j]) +
                   b * abs(inhibitors[k] - inhibitors[j]) +
                   c * abs(enhancers[k] - enhancers[j])
        """
        return (abs(self.identifiers[k] - other.identifiers[j]) *
                config.ID_COEF +
                abs(self.inhibitors[k] - other.inhibitors[j]) *
                config.INH_COEF +
                abs(self.enhancers[k] - other.enhancers[j]) * config.ENH_COEF) / self.idsize

    def distance_to(self, other):
        """
            Returns the distance between the genome A and B
            The genome A is the parent object of genotype self, and B have the genotype other

            dist = summ_inputs(D(a, b) for a and B inputs protein in both Genomes ) + 
                   summ_outputs(D(a, b) for a and B outputs protein in both Genomes ) + 
                   summ_rA(min_Rb(D(a, b)) for a and B regulatory protein in both Genomes ) + Db + Dd/(max_size + 2) 
            [1]
            return dist
        """


        distance = 0.0

        if self.size() > other.size():
            gsmall = other
            glarge = self
        else:
            gsmall = self
            glarge = other

        # Compute Din and Dout, input protein are the first num_input + num_output elements
        for k in range(glarge.num_input + glarge.num_output):
            gDist = glarge.protein_distance(gsmall, k, k)
            distance += gDist

        # Compute Dreg
        # protein k in large
        for k in range(glarge.num_input + glarge.num_output, glarge.size()):
            if gsmall.num_regulatory == 0:
                distance += config.ID_COEF + config.INH_COEF + config.ENH_COEF # not mentionned in [1]
            else:
                # could be done with numpy no ?
                min_dist = np.inf 
                # protein j in small
                for j in range(gsmall.num_input + gsmall.num_output, gsmall.size()):
                    gdist = glarge.protein_distance(gsmall, k, j)
                    if gdist < min_dist:
                        min_dist = gdist
                distance += min_dist

        # Compare dynamics
        distance += (abs(glarge.beta - gsmall.beta) /
                     (config.BETA_MAX - config.BETA_MIN))
        distance += (abs(glarge.delta - gsmall.delta) /
                     (config.DELTA_MAX - config.DELTA_MIN))

        distance /= glarge.size() + 2
        return distance

    def get_dt(self):
        return self.dt
