# coding=utf-8

""" Calibrate the neutrality test by calculating a threshold for a given false positive
    rate (default is 0.02) against the Wright-Fisher model.

    This scripts uses an order 2 Taylor series approximation toi calculate likelihood under the Wrifgt-Fisher model."""

import os, sys
import numpy as np
import pandas as pd
from selectiontest import selectiontest
from time import time
import click
from scitrack import CachingLogger, get_file_hexdigest


LOGGER = CachingLogger(create_dir=True)


def hessian(x, p):
    n = len(x) + 1
    m = sum(x)
    result = np.zeros((n - 2, n - 2))
    pmf = selectiontest.multinomial_pmf(x, m, p)
    for i in range(n - 2):
        for j in range(n - 2):
            result[i,j] = pmf * (x[i]/p[i] - x[n - 2]/p[n - 2]) * (x[j]/p[j] - x[n - 2]/p[n - 2])
    for i in range(n - 2):
        result[i, i] += -pmf * x[i] / (p[i] * p[i])
    return result


def approx_likelihood(sfs, ev=None, covar=None, reps=None):
    n = len(sfs) + 1
    if ev is None:
        assert reps is not None, "If expected value is not supplied, reps must be supplied."
        variates = np.empty((reps, n - 2), dtype=float)
        for i, q in enumerate(selectiontest.sample_wf_distribution(n, reps)):
            variates[i] = q
        ev = np.mean(variates, axis=0)
        covar = np.cov(variates[:,:-1], rowvar=False)
    else:
        assert covar is not None, "If expected value is supplied, covariance matrix must be supplied also."
    m = sum(sfs)
    term0 = selectiontest.multinomial_pmf(sfs, m, ev)
    hess = hessian(sfs, ev)
    term2 = np.sum(hess * covar) / 2
    return term0 + term2


def approx_threshold(n, seg_sites, sreps=10000, wreps=10000, fpr=0.02):
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
    variates0 = np.empty((wreps, n - 1), dtype=float)
    for i, q in enumerate(selectiontest.sample_wf_distribution(n, wreps)):
        variates0[i] = q
    ev = np.mean(variates0, axis=0)
    covar = np.cov(variates0[:,:-1], rowvar=False)
    variates1 = selectiontest.sample_uniform_distribution(n, sreps)
    results = list()
    errcount = 0
    for sfs in selectiontest.generate_sfs_array(n, seg_sites, sreps):
        h0 = approx_likelihood(sfs, ev=ev, covar=covar)
        if h0 <= 0:
            errcount += 1
            h0 = selectiontest.multinomial_pmf(sfs, seg_sites, ev)
        h1 = np.mean(selectiontest.multinomial_pmf(sfs, seg_sites, variates1))
        rho = np.log10(h1) - np.log10(h0)
        results.append(rho)
    results = np.array(results)
    print("Count -inf: ", np.sum(np.isneginf(results)))
    print("Count  inf: ", np.sum(np.isinf(results)))
    print("Count  nan: ", np.sum(np.isnan(results)))
    print("Count  -ve: ", errcount)
    results = results[~np.isnan(results)]
    results = np.sort(results)
    return results[int(len(results) * (1 - fpr))]


@click.command()
@click.argument('job_no')
@click.argument('seg_sites_values', nargs=4, type=int)
@click.argument('sample_size_values', nargs=-1, type=int)
@click.option('-f', '--fpr', default=0.02, help="False positive rate. Default = 0.02")
@click.option('-sr', '--sreps', default=10000, help="Number of repetitions to generate sfs and uniform samples.")
@click.option('-wr', '--wreps', default=100000, help="Number of repetitions for WF samples used in selectiontest.")
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
    rows = list()
    for sn in seg_sites_values:
        thresholds = list()
        if sn == 0:
            break
        for n in sample_size_values:
            print(sn, n)
            thr = approx_threshold(n, sn, sreps=sreps, wreps=wreps, fpr=fpr)
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