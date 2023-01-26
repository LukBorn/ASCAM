# All credit for the DISC algortihm belongs to the authors of
# White, D.S., Goldschen-Ohm, M.P., Goldsmith, R.H., Chanda, B. Top-down machine learning approach for high-throughput single-molecule analysis. Elife 2020, 9
# The code in this module is based on their paper and partly on their
# matlab implementation at commit af19eae on https://github.com/ChandaLab/DISC/

import numpy as np

from .divisive_segmentation import divisive_segmentation
from .agglomerative_clustering import (merge_by_ward_distance,
                                       agglomorative_clustering_fit)
from .infomation_criteria import compare_IC
from .viterbi import (viterbi_path, viterbi_path_from_data,
                      compute_emission_matrix,
                      empirical_transition_matrix)

def run_DISC(data, confidence_level=0.05, min_seg_length=3,
             min_cluster_size=3, IC_div_seg="BIC", IC_HAC="BIC"):
    """
    Input:
        data - observations to be idealized
        confidence_level - alpha value for t-test in changepoint detection
        min_seg_length - minimum segment length in changepoint detection
        min_cluster_size - minimum cluster size for k-mean clustering
        IC_div_seg - information criterion for divisive segmentation
        IC_HAC - information criterion for agglomorative clustering
    Returns:
        an idealization of the data to the most likely sequence as found
        by the DISC algortihm
    """

    # 1) Start with divisive segmentation.
    data_fit, _ = divisive_segmentation(
                                        data,
                                        confidence_level=confidence_level,
                                        min_seg_length=min_seg_length,
                                        min_cluster_size=min_cluster_size,
                                        information_criterion=IC_div_seg
                                        )

    # 2) Perform Hierarchical Agglomerative Clustering (HAC) by
    # consecutively merging those clusters with the smallest Ward distance
    # between them. Then select the fit with the optimal number of clusters
    # as measured by the given Information Criterion (IC)
    data_fit = agglomorative_clustering_fit(data, data_fit, IC=IC_HAC)
    all_data_fits = merge_by_ward_distance(data_fit)
    best_fit_ind = compare_IC(data, all_data_fits, IC=IC_HAC)
    data_fit = all_data_fits[:, best_fit_ind]
    # At this point we could in the future include logic to return a
    # specific number of states (≤nuber of states found by divisive
    # segmentation, or multiple fits.

    # 3) Run Viterbi algorithm to find best fit to the given number of
    # states and their amplitudes.
    # Use empirical distribution of states as initial distribution for the
    # Viterbi algorithm. Compute best fit normal distribution for
    # emission matrix approximation.
    return viterbi_path_from_data(data, data_fit)
