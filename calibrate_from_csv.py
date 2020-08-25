# coding=utf-8

""" Calibrate the neutrality test by calculating a threshold for a given false positive
    rate (default is 0.02) against the Wright-Fisher model.
    This method reads WF variates line by line rom csv file in order that large numbers opf variates can be used."""

import os, sys
import numpy as np
import pandas as pd
from selectiontest import selectiontest
from joblib import Parallel, delayed
from time import time
import click
from scitrack import CachingLogger, get_file_hexdigest


LOGGER = CachingLogger(create_dir=True)


def get_WF_likelihood(sfs, infile):
    count = 0
    mask = sfs > 0
    seg_sites = sum(sfs)
    lhood_sum = 0
    for probs in np.loadtxt(infile, delimiter=','):   #genfromtxt
        count += 1
        b = probs[mask] > 0
        if np.all(b > 0):
            lhood_sum += selectiontest.multinomial_pmf(sfs, seg_sites, probs)
    return lhood_sum / count


def calculate_rho(sfs, seg_sites, fname, variates1):
    infile = open(fname)
    h0 = get_WF_likelihood(sfs, infile)
    h1 = np.mean(selectiontest.multinomial_pmf(sfs, seg_sites, variates1))
    rho = np.log10(h1) - np.log10(h0)
    return rho

def compute_threshold(n, seg_sites, fname, njobs, sreps=10000, fpr=0.02):
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
    fname: str
        Name of .csv file containing Wright-Fisher variates (no path or suffix).
    njobs: int
        Number of parallel joblib processes.
    sreps: int
        Number of SFS configs and of uniform variates to generate if default is used.
    fpr: float
        Selected FPR tolerance.

    Returns
    -------
    numpy.float64
        Threshold value for log odds ratio

    """

    variates1 = selectiontest.sample_uniform_distribution(n, sreps)
    sfs_array = selectiontest.generate_sfs_array(n, seg_sites, sreps)
    print('sfs simulation complete')
    sys.stdout.flush()
    results = Parallel(n_jobs=njobs)(delayed(calculate_rho)(sfs, seg_sites, fname, variates1) for sfs in sfs_array)
    # results = list()
    # for sfs in sfs_array:
    #     infile = open(fname)
    #     h0 = get_WF_likelihood(sfs, infile)
    #     h1 = np.mean(selectiontest.multinomial_pmf(sfs, seg_sites, variates1))
    #     rho = np.log10(h1) - np.log10(h0)
    #     results.append(rho)
    print('selectiontest complete')
    sys.stdout.flush()
    results = np.array(results)
    print(np.sum(np.isneginf(results)))
    print(np.sum(np.isinf(results)))
    print(np.sum(np.isnan(results)))
    sys.stdout.flush()
    results = results[~np.isnan(results)]
    results = np.sort(results)
    return results[int(len(results) * (1 - fpr))]

@click.command()
@click.argument('job_no')
@click.argument('fname')
@click.argument('sample_size', type=int)
@click.argument('seg_sites_values', nargs=-1, type=int)
@click.option('-f', '--fpr', default=0.02, help="False positive rate. Default = 0.02")
@click.option('-sr', '--sreps', default=10000, help="Number of repetitions to generate sfs and uniform samples.")
@click.option('-j', '--njobs', default=10, help="Number of parallel jobs.")
@click.option('-d', '--dirx', default='data', type=click.Path(),
              help='Directory name for data and log files. Default is data')
def main(job_no, fname, sample_size, seg_sites_values, fpr, sreps, njobs, dirx):
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
    LOGGER.log_message('Name = ' + pd.__name__ + ', version = ' + pd.__version__, label=label)
    LOGGER.log_message('Name = ' + selectiontest.__name__ + ', version = ' + selectiontest.__version__, label=label)

    start_time = time()
    thresholds = list()
    fname = dirx + '/' + fname
    for sn in seg_sites_values:
        thr = compute_threshold(sample_size, sn, fname, njobs, sreps=sreps,fpr=fpr)
        duration = time() - start_time
        print("%.2f" % (duration / 60.), "%4d" % sample_size, "%3d" % sn, "%.3f" % thr)
        sys.stdout.flush()
        thresholds.append(thr)
    results = pd.Series(thresholds, index = seg_sites_values, name=sample_size)
    fname = dirx + "/calibration_" + job_no + ".csv"
    results.to_csv(fname)
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))


if __name__ == "__main__":
    main()