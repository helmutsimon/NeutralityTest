# coding=utf-8

""" Calibrate the neutrality test by calculating a threshold for a given false positive
    rate (default is 0.02) against the Wright-Fisher model."""

import os, sys
import numpy as np
import pandas as pd
from selectiontest import selectiontest
from time import time
import click
from scitrack import CachingLogger, get_file_hexdigest


LOGGER = CachingLogger(create_dir=True)


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
    seg_sites_values = [x for x in seg_sites_values if x > 0]
    rows = list()
    for sn in seg_sites_values:
        thresholds = list()
        for n in sample_size_values:
            print(sn, n)
            thr = selectiontest.compute_threshold(n, sn, sreps=sreps, wreps=wreps, fpr=fpr)
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