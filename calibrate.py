# coding=utf-8

""" Functions for calibrating the neutrality test by calculating a threshold for a given false positive
    rate (default is 0.02) against the Wright-Fisher model."""

import numpy as np
import sys
from scipy.special import binom
from scipy.stats import multinomial, expon
from scipy.stats import dirichlet


def get_ERM_matrix(n):
    ERM_matrix = np.zeros((n - 1, n - 1))
    for m in range(n - 1):
        for k in range(n - 1):
            ERM_matrix[m, k] = (k + 2) * binom(n - m - 2, k) / binom(n - 1, k + 1)
    return ERM_matrix


def mul(theta):
    def multinom(p):
        return multinomial.rvs(theta, p)

    return multinom


def generate_sfs_array(n, seg_sites, reps=10000):
    """Sample SFS values for Wright-Fisher model for given sample size n and conditioned on the
    number of segregating sites."""
    erm = get_ERM_matrix(n)
    kvec = np.arange(2, n + 1, dtype=int)
    variates = expon.rvs(scale=1 / binom(kvec, 2), size=(reps, n - 1))
    total_branch_lengths = variates @ kvec
    rel_branch_lengths = np.diag(1 / total_branch_lengths) @ variates
    qvars = (erm @ rel_branch_lengths.T).T
    sfs_array = np.apply_along_axis(mul(seg_sites), 1, qvars)
    return sfs_array


def test_neutrality3(n, reps):
    """Return function to calculate RLNT."""
    j_n = np.diag(1 / np.arange(2, n + 1))
    erm = get_ERM_matrix(n)
    avge_mx = erm.dot(j_n)
    kvec = np.arange(2, n + 1, dtype=int)
    variates = expon.rvs(scale=1 / binom(kvec, 2), size=(reps, n - 1))
    total_branch_lengths = variates @ kvec
    rel_branch_lengths = np.diag(1 / total_branch_lengths) @ variates
    qvars = (erm @ rel_branch_lengths.T).T

    def test_neutrality_short(sfs):
        """Calculate the odds ratio for neutral / not neutral."""
        h1 = np.mean(multinomial.pmf(sfs, np.sum(sfs), qvars))
        sample = dirichlet.rvs(np.ones(n - 1), size=reps)
        sample = avge_mx @ sample.T
        pmfs = multinomial.pmf(sfs, np.sum(sfs), sample.T)
        h2 = np.mean(pmfs)
        if h1 == 0 or h2 == 0:
            print(sfs, 'h1 = ', h1, 'h2 = ', h2)
            if h1 != 0:
                h2 = sys.float_info.min
        return h1 / h2

    return test_neutrality_short


def compute_threshold(n, seg_sites, reps=10000, threshold=0.02):
    """Calculate threshold value of RLNT below which we reject the neutral hypothesis."""
    sfs_array = generate_sfs_array(n, seg_sites, reps)
    results = np.apply_along_axis(test_neutrality3(n, reps), 1, sfs_array)
    results = np.sort(results)
    results = results[~np.isnan(results)]
    return results[int(len(results) * threshold)]