import numpy as np
import logging
from src.evolution import Mutation, ParentSelection, Recombination, EA
from src.target_function import ackley


def evaluate_black_box(mutation, selection, recombination):
    """
    Simple wrapper for your EA implementation. With your below hpo method you won't have to worry about
    other parameters
    :param mutation:
    :param selection:
    :param recombination:
    :return:
    """
    ea = EA(ackley, 20, 2, selection_type=selection,
            total_number_of_function_evaluations=500, problem_bounds=[-10, 10],
            mutation_type=mutation, recombination_type=recombination,
            sigma=1., children_per_step=5, fraction_mutation=.5, recom_proba=.5)
    res = ea.optimize()
    return res.fitness


def determine_best_hypers():
    """
    TODO implement either grid or random search to determine the best hyperparameter setting of your EA implementation
    when overfitting to the ackley function. The only parameter values you have to consider are selection_type,
    mutation_type and recombination type. You can treat the EA as a black-box by optimizing the black-box-function
    above. Note: the order of your "configuration" has to be as stated below

    :return: best configuration as tuple e.g. (mutation, selection, recombination) and performance value
    """
    # TODO
    return grid_search()

def grid_search():
    
    hyper_param = [Recombination.NONE, Mutation.NONE,ParentSelection.NEUTRAL]
    perf_min = 1000000
    
    mutations = [Mutation.NONE, Mutation.UNIFORM, Mutation.GAUSSIAN]
    parent_selections = [ParentSelection.NEUTRAL, ParentSelection.FITNESS, ParentSelection.TOURNAMENT]
    recombinations = [Recombination.NONE, Recombination.UNIFORM, Recombination.INTERMEDIATE]
    
    

    for mutation in mutations:
        for parent_selction in parent_selections:
            for recombination in recombinations:
                perf = evaluate_black_box(mutation, parent_selction, recombination)
                if perf < perf_min:
                    hyper_param = [mutation, parent_selction, recombination]
                    perf_min = perf
                    
    return hyper_param, perf_min