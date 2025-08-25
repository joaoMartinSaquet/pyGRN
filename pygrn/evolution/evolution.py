from pygrn.evolution import Population
from pygrn.grns import ClassicGRN
import numpy as np
import os
import pathlib
from datetime import datetime
from uuid import uuid4
from loguru import logger
import pygrn.config as config

import joblib as jl    

class Evolution:

    def __init__(self, problem, 
        new_grn_function=lambda: ClassicGRN(),
        run_id=str(uuid4()), 
        grn_dir='grns', 
        log_dir='logs',
        num_workers=1, 
        log_grn = True, 
        log_fit = True):
        pathlib.Path(grn_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.grn_file = os.path.join(grn_dir, 'grns_' + run_id + '.log')
        self.log_file = os.path.join(log_dir, 'fits_' + run_id + '.log')
        self.problem = problem
        self.population = Population(new_grn_function, problem.nin,
                                     problem.nout)
        message = f"""
        GRNEAT HYPERPARAMETERS:
        POPULATION_SIZE = {config.POPULATION_SIZE}
        mutation_rate = {config.MUTATION_RATE}
        mutation_add_rate = {config.MUTATION_ADD_RATE}
        mutation_del_rate = {config.MUTATION_DEL_RATE}
        crossover_rate = {config.CROSSOVER_RATE}
        crossover_threshold = {config.CROSSOVER_THRESHOLD}
        tournament_size = {config.TOURNAMENT_SIZE}
        tournament_with_replacement = {config.TOURNAMENT_WITH_REPLACEMENT}
        start_regulatory_size = {config.START_REGULATORY_SIZE}
        min_species_size = {config.MIN_SPECIES_SIZE}
        start_species_threshold = {config.START_SPECIES_THRESHOLD}
        species_threshold_update = {config.SPECIES_THRESHOLD_UPDATE}
        min_species_threshold = {config.MIN_SPECIES_THRESHOLD}
        max_species_threshold = {config.MAX_SPECIES_THRESHOLD}
        id_coef = {config.ID_COEF}
        enh_coef = {config.ENH_COEF}
        inh_coef = {config.INH_COEF}
        max_selection_tries = {config.MAX_SELECTION_TRIES}
        beta_min = {config.BETA_MIN}
        beta_max = {config.BETA_MAX}
        delta_min = {config.DELTA_MIN}
        delta_max = {config.DELTA_MAX}
"""
        logger.info(message)

        self.generation = 0
        self.best_fit_history = []

        self.num_workers = num_workers

        self.log_fit = log_fit
        self.log_grn = log_grn

        self.mem = jl.Memory(location='cache_dir', verbose=0)

    def step(self):


        # clear cache memory 
        self.mem.clear(warn=False)
        self.population.evaluate(self.problem, self.num_workers)
        self.population.speciation()
        self.population.adjust_thresholds()

        # renew species
        self.population.set_offspring_count()
        self.population.make_offspring()
        self.report()
        self.problem.generation_function(self, self.generation)
        self.generation += 1

    def run(self, generations):
        for gen in range(generations):
            self.step()
        best_fit, best_ind_pop = self.population.get_best()
        return best_fit, best_ind_pop

    def report(self, ):

        
        for species_id in range(len(self.population.species)):
            sp = self.population.species[species_id]
            sp_best = sp.get_best_individual()
            # logger.debug(
            #     f'S,%s,%d,%d,%d,%f,%f,%d,%f,%f,%f' % (
            #     datetime.now().isoformat(),
            #     self.generation, species_id,
            #     len(sp.individuals),
            #     sp.sum_adjusted_fitness,
            #     sp_best.fitness,
            #     sp_best.grn.size(),
            #     sp.species_threshold,
            #     np.mean(sp.get_representative_distances()),
            #     np.mean([i.grn.size() for i in sp.individuals])  
            #     )

            # )
            if self.log_fit :
                with open(self.log_file, 'a') as f:
                    f.write('S,%s,%d,%d,%d,%f,%f,%d,%f,%f,%f\n' % (
                        datetime.now().isoformat(),
                        self.generation, 
                        species_id,
                        len(sp.individuals),
                        sp.sum_adjusted_fitness,
                        sp_best.fitness,
                        sp_best.grn.size(),
                        sp.species_threshold,
                        np.mean(sp.get_representative_distances()),
                        np.mean([i.grn.size() for i in sp.individuals])))
        best_fitness, best_ind = self.population.get_best()
        fit_mean, fit_std = self.population.get_stats()
        logger.info(f"Generation {self.generation}: best fit {best_fitness}, fit mean {fit_mean}, fit std {fit_std}")
        if self.log_fit:
            with open(self.log_file, 'a') as f:
                f.write('G,%s,%d,%d,%f,%d,%f,%f\n' % (
                    datetime.now().isoformat(),
                    self.generation, self.population.size(),
                    best_fitness, best_ind.grn.size(),
                    fit_mean, fit_std))
           
        if self.log_grn:
            with open(self.grn_file, 'a') as f:
                f.write(str(best_ind.grn) + '\n')

        self.best_fit_history.append(best_fitness)
