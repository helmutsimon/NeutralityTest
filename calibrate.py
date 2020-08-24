# coding=utf-8

""" Calibrate the neutrality test by calculating a threshold for a given false positive
    rate (default is 0.02) against the Wright-Fisher model."""

import os, sys
import numpy as np
import pandas as pd
from collections import Counter
from scipy.special import binom
from selectiontest import selectiontest
from joblib import Parallel, delayed
from time import time
import click
from scitrack import CachingLogger, get_file_hexdigest


LOGGER = CachingLogger(create_dir=True)


def sample_matrices3(n, size, seed):
    np.random.seed(seed)
    counts, mxs = Counter(), Counter()
    for j in range(size):
        f = list()
        for i in range(1, n):
            f.append(np.random.choice(i))
        f = f[::-1]
        mx = selectiontest.derive_tree_matrix(f)
        hashmx = mx.tobytes()
        mxs[hashmx] = mx
        counts[hashmx] += 1
    return mxs, counts


def sample_wf_distribution(n, reps, seed):
    """
    Calculate variates for the probability distribution Q under Wright Fisher model.

    Parameters
    ----------
    n: int
        Sample size
    reps: int
        Number of variates to generate if default is used.

    Returns
    -------
    numpy.ndarray
         Array of variates (reps, n-1)

    """
    matrices, counts = sample_matrices3(n, reps, seed)
    kvec = np.arange(2, n + 1, dtype=int)
    branch_lengths = np.random.exponential(scale=1 / binom(kvec, 2), size=(reps, n - 1))
    total_branch_lengths = branch_lengths @ kvec
    rel_branch_lengths = list()
    for row, total_length in zip(branch_lengths, total_branch_lengths):
        rel_row = row / total_length
        rel_branch_lengths.append(rel_row)
    rel_branch_lengths = np.array(rel_branch_lengths)
    variates = list()
    rbl_count = 0
    for key in matrices.keys():
        for j in range(counts[key]):
            mx = matrices[key]
            variate = (mx.T).dot(rel_branch_lengths[rbl_count])
            #variate = list(variate)       #
            rbl_count += 1
            err = 1 - np.sum(variate)
            variate[np.argmax(variate)] += err
            variates.append(variate)
    return np.array(variates)


def sample_WF_pllel(n, size, njobs):
    seeds = np.random.choice(2 * njobs, njobs, replace=False)
    results = Parallel(n_jobs=njobs)(delayed(sample_wf_distribution)(n, size, seed) for seed in seeds)
    return np.vstack(results)


def compute_threshold(n, seg_sites, njobs, sreps=10000, wreps=10000, fpr=0.02):
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
    njobs: int
        Number of parallel joblib processes.
    sreps: int
        Number of SFS configs and of uniform variates to generate if default is used.
    wreps: int
        Number of Wright-Fisher variates to generate if default is used.
    fpr: float
        Selected FPR tolerance.

    Returns
    -------
    numpy.float64
        Threshold value for log odds ratio

    """

    #variates0 = selectiontest.sample_wf_distribution(n, wreps)
    variates0 = sample_WF_pllel(n, wreps, njobs)
    sys.stdout.flush()
    variates1 = selectiontest.sample_uniform_distribution(n, sreps)
    sfs_array = selectiontest.generate_sfs_array(n, seg_sites, sreps)
    print('sfs simulation complete')
    sys.stdout.flush()
    num_wf_vars = variates0.shape[0]
    results = list()
    for sfs in sfs_array:
        a = sfs > 0
        b = variates0[:, a] > 0
        c = np.all(b > 0, axis=1)
        compat_vars = variates0[c, :]
        if compat_vars.shape[0] == 0:
            h0 = 0
        else:
            h0 = np.sum(selectiontest.multinomial_pmf(sfs, seg_sites, compat_vars)) / num_wf_vars
        h1 = np.mean(selectiontest.multinomial_pmf(sfs, seg_sites, variates1))
        rho = np.log10(h1) - np.log10(h0)
        results.append(rho)
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
@click.argument('seg_sites_values', nargs=4, type=int)
@click.argument('sample_size_values', nargs=-1, type=int)
@click.option('-f', '--fpr', default=0.02, help="False positive rate. Default = 0.02")
@click.option('-sr', '--sreps', default=10000, help="Number of repetitions to generate sfs and uniform samples.")
@click.option('-wr', '--wreps', default=10000, help="Number of repetitions for WF samples used in selectiontest.")
@click.option('-j', '--njobs', default=10, help="Number of parallel jobs.")
@click.option('-d', '--dirx', default='data', type=click.Path(),
              help='Directory name for data and log files. Default is data')
def main(job_no, seg_sites_values, sample_size_values, fpr, sreps, wreps, njobs, dirx):
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
    thresholds, rows = list(), list()
    for sn in seg_sites_values:
        if sn == 0:
            break
        for n in sample_size_values:
            print(sn, n)
            thr = compute_threshold(n, sn, njobs, sreps=sreps, wreps=wreps,fpr=fpr)  # don't need last 2 params
            duration = time() - start_time
            print("%.2f" % (duration / 60.), "%4d" % n, "%3d" % sn, "%.3f" % thr)
            sys.stdout.flush()
            thresholds.append(thr)
        rows.append(thresholds)
    results = pd.DataFrame(rows, index = seg_sites_values, columns=sample_size_values)
    fname = dirx + "/calibration_" + job_no + ".csv"
    results.to_csv(fname)
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))


if __name__ == "__main__":
    main()