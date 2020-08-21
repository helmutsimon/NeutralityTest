# coding=utf-8

""" Calibrate the neutrality test by calculating a threshold for a given false positive
    rate (default is 0.02) against the Wright-Fisher model."""

import os, sys
import numpy as np
import pandas as pd
from selectiontest import selectiontest
import mpmath
from joblib import Parallel, delayed
from time import time
import click
from scitrack import CachingLogger, get_file_hexdigest


LOGGER = CachingLogger(create_dir=True)


def power_vector(p, x):
    result = mpmath.mp.mpf(0)
    for xi, pi in zip(x, p):
        if xi == 0:
            logpwr = 0
        else:
            logpwr = mpmath.fmul(xi, mpmath.log(pi))

        result = mpmath.fadd(result, logpwr)
    return mpmath.exp(result)


def power_array(x, p):
    result = mpmath.mp.mpf(0)
    for prow in p:
        pv = power_vector(prow, x)
        result = mpmath.fadd(result, pv)
    return result


def test_neutrality(sfs, variates0=None, variates1=None, reps=10000):
    """
    Calculate :math:`\\rho`, the log odds ratio of the data for the distribution given by variates0 over
    the distribution given by variates1.

    Parameters
    ----------
    sfs: list
        Site frequency spectrum, e.g. [1, 3, 0, 2, 1]
    variates0: numpy array
        Array of variates from null hypothesis distribution. Default uses Wright-Fisher model.
    variates1: numpy array
        Array of variates from alternative distribution. Default uses \`uniform\' model.
    reps: int
        Number of variates to generate if default is used.

    Returns
    -------
    numpy.float64
        :math:`\\rho` (value of log odds ratio). Values can include inf, -inf or nan if one or both probabilities
        are zero due to underflow error.

    """
    n = len(sfs) + 1
    segsites = sum(sfs)
    if variates0 is None:
        variates0 = selectiontest.sample_wf_distribution(n, reps)
    if variates1 is None:
        variates1 = selectiontest.sample_uniform_distribution(n, reps)
    h0 = power_array(sfs, variates0)
    h1 = power_array(sfs, variates1)
    result = mpmath.fsub(mpmath.log10(h1), mpmath.log10(h0))
    return float(result)


def mul(seg_sites):
    def multinom(p):
        return np.random.multinomial(seg_sites, p)

    return multinom


def generate_sfs_array(n, seg_sites, reps=10000):
    """
    Sample SFS values for Wright-Fisher model for given sample size n and conditioned on the
    number of segregating sites.

    """
    variates = selectiontest.sample_wf_distribution(n, reps)
    sfs_array = np.apply_along_axis(mul(seg_sites), 1, variates)
    return sfs_array


def compute_threshold(n, seg_sites, njobs, reps=10000, fpr=0.02):
    """
    Calculate threshold value of :math:`\\rho` corresponding to a given false positive rate (FPR).
    For values of :math:`\\rho` above the threshold we reject the
    null (by default neutral) hypothesis.

    Parameters
    ----------
    n: int
        Sample size
    seg_sites: int
        Number of segregating sites in sample.
    reps: int
        Number of variates to generate if default is used.
    fpr: float
        Selected FPR tolerance.

    Returns
    -------
    numpy.float64
        Threshold value for log odds ratio

    """

    variates0 = selectiontest.sample_wf_distribution(n, 10000)
    variates1 = selectiontest.sample_uniform_distribution(n, 10000)
    sfs_array = generate_sfs_array(n, seg_sites, reps)
    results = Parallel(n_jobs=njobs)(delayed(test_neutrality)(sfs, variates0, variates1, reps=reps) \
                                 for sfs in sfs_array)
    results = results[~np.isnan(results)]
    results = np.sort(results)
    return results[int(len(results) * (1 - fpr))]

@click.command()
@click.argument('job_no')
@click.argument('seg_sites', type=int)
@click.argument('sample_size_values', nargs=-1, type=int)
@click.option('-f', '--fpr', default=0.02, help="False positive rate. Default = 0.02")
@click.option('-r', '--reps', default=10000, help="Number of repetitions")
@click.option('-j', '--njobs', default=10, help="Number of repetitions")
@click.option('-p', '--dps', default=50, help="Number of repetitions")
@click.option('-d', '--dirx', default='data', type=click.Path(),
              help='Directory name for data and log files. Default is data')
def main(job_no, seg_sites, sample_size_values, fpr, reps, njobs, dps, dirx):
    np.set_printoptions(precision=3)                #
    if not os.path.exists(dirx):
        os.makedirs(dirx)
    LOGGER.log_file_path = dirx + "/" + str(os.path.basename(__file__)) + '_' + job_no + ".log"
    LOGGER.log_args()
    LOGGER.log_message(get_file_hexdigest(__file__), label="Hex digest of script.".ljust(17))
    try:
        LOGGER.log_message(str(os.environ['CONDA_DEFAULT_ENV']), label="Conda environment.".ljust(17))
    except KeyError:
        pass
    label = "Imported module".ljust(30)
    LOGGER.log_message('Name = ' + np.__name__ + ', version = ' + np.__version__, label=label)
    LOGGER.log_message('Name = ' + selectiontest.__name__ + ', version = ' + selectiontest.__version__, label=label)
    LOGGER.log_message('Name = ' + mpmath.__name__ + ', version = ' + mpmath.__version__, label=label)

    start_time = time()
    mpmath.mp.dps = dps
    thresholds = list()
    for n in sample_size_values:
        thr = compute_threshold(n, seg_sites, njobs, reps=reps, fpr=fpr)  # don't need last 2 params
        duration = time() - start_time
        print("%.2f" % (duration / 60.), "%3d" % seg_sites, "%4d" % n, "%.3f" % thr)
        sys.stdout.flush()
        thresholds.append(thr)
    results = pd.DataFrame(thresholds, index = sample_size_values, columns=[seg_sites])
    fname = dirx + "/calibration_" + job_no + ".csv"
    results.to_csv(fname)
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))


if __name__ == "__main__":
    main()