# coding=utf-8


"""
Calculates :math:`\\rho` and Tajima's D for neutral and non-neutral data simulated with MSMS.
Simulations are conditioned either on :math:`\\theta` or on the number of segregating sites.
The data is intended to be used to plot roc curves (see relevant Jupyter notebook).

The script invokes a java script and must be run in a directory containing msms.jar file.

A sample run statement is:

nohup python3 /Users/helmutsimon/repos/NeutralityTest/roc_simulation.py 2s00a 10 0 -t 10 -sho 2 -sf 1e-2  > rs_2s00a.txt &

"""

import numpy as np
import os, sys
import gzip, pickle
from selectiontest import selectiontest
from time import time
from collections import Counter
import msprime
import subprocess
import click
import pandas as pd
import scipy
from scitrack import CachingLogger, get_file_hexdigest

LOGGER = CachingLogger(create_dir=True)


def compute_sfs(variant_array):
    """
    Compute the site frequency spectrum from a variant array output by MSMS.

    """
    n = variant_array.shape[1]
    occurrences = np.sum(variant_array, axis=1)
    sfs = Counter(occurrences)
    sfs = [sfs[i] for i in range(1, n)]
    return np.array(sfs)


def read_demography_file(filename, pop_size):
    """
    Read demographic history file and translate into MSMS commands. The file uses absolute
    population size and exponential growth rate per generation. These are scaled by 4N_0 (pop_size)
    for MSMS. The input file format used is compatible with msprime and msprime functions are
    used to print a description of the demographic history.

    """
    if filename is None:
        return list()
    if isinstance(filename, str):
        if filename[-4:] != '.csv':
            filename = filename + '.csv'
            infile = open(filename, 'r')
            LOGGER.input_file(infile.name)
            infile.close()
        demo_parameters = pd.read_csv(filename)
    elif isinstance(filename, pd.DataFrame):
        demo_parameters = filename
    else:
        raise ValueError('Events_file parameter wrong type: ' + str(type(filename)))
    dem_cmds = list()
    demographic_events = list()
    for index, row in demo_parameters.iterrows():
        t = str(row['time'] / (4 * pop_size))
        size_change = str(row['size'] / pop_size)
        growth_rate = str(row['rate'] * 4 * pop_size)
        ppc = msprime.PopulationParametersChange(time=row['time'],
                    initial_size=row['size'], growth_rate=row['rate'])
        demographic_events.append(ppc)
        event = 'time: ' + str(row['time']) + ', size: ' + str(row['size']) +\
                ', rate: ' + str(row['rate'])
        #print(event)
        LOGGER.log_message(event, label='Demographic change event'.ljust(50))
        cmds = ["-eN", t, size_change, "-eG", t, growth_rate]
        dem_cmds = dem_cmds + cmds
    return dem_cmds, demographic_events


def run_simulations(reps, pop_size, n, theta, seg_sites, growth_rate, SAA, SaA, SF, events_file, recombination_rate):
    """
    Run simulations using MSMS.

    """
    # Construct MSMS parameters for demographic history and print a summary using msprime.
    if events_file is None:
        events_commands = list()
    else:
        events_commands, demographic_events = read_demography_file(events_file, pop_size)
        population_configurations = [msprime.PopulationConfiguration(initial_size=pop_size,
                    sample_size=n, growth_rate=growth_rate)]
        demographic_hist = msprime.DemographyDebugger(demographic_events=demographic_events,
                    population_configurations=population_configurations)
        demographic_hist.print_history()
        sys.stdout.flush()

    # Build the MSMS command line, conditioning either on theta or seg_sites and including selection if required.
    msms_cmd = ["java", "-jar", "msms.jar", str(n), str(reps), "-N", str(pop_size)] + events_commands
    if growth_rate:
        growth_rate_scaled = growth_rate * 4 * pop_size
        msms_cmd = msms_cmd + [ "-G", str(growth_rate_scaled)]
    if seg_sites is not None and theta is not None:
        raise ValueError('seg_sites and theta parameters cannot both be valid.')
    if seg_sites is not None:
        msms_cmd = msms_cmd + ["-s", str(seg_sites)]
    if theta is not None:
        msms_cmd = msms_cmd + ["-t", str(theta)]
    if SAA is not None:
        if events_file is not None or growth_rate:
            raise ValueError('Cannnot combine demographic change and selection.')
        if SaA is None:
            SaA = SAA / 2
        msms_cmd = msms_cmd + ["-SAA", str(SAA), "-SaA", str(SaA), "-SF", str(SF)]
    if recombination_rate:
        msms_cmd = msms_cmd + ["-r", str(recombination_rate)]
    print(msms_cmd)
    # Run MSMS command and return output.
    msms_out = subprocess.check_output(msms_cmd)
    return(msms_out)


def process_simulation_output(msms_out, variates0, variates1, reps):
    """
    Process MSMS output line by by line (each line corresponding to a single simulation).
    We use a common set of variates for all calls to test_neutrality.

    """
    sfs_list, trs, taj_D = list(), list(), list()
    not_enough, err_nan = 0, 0
    ms_lines = msms_out.splitlines()
    variant_array = list()
    for line in ms_lines[6:]:
        line = line.decode('unicode_escape')
        if line[:2] in ['//', 'se', 'po']:
            variant_array = list()
            continue
        if line != "":
            variant_array.append([*line])
        else:
            variant_array = np.array(variant_array, dtype=int).T
            if variant_array.shape[0] < 4:
                not_enough += 1
                continue
            sfs = compute_sfs(variant_array)
            sfs_list.append(sfs)
            tr = selectiontest.test_neutrality(sfs, variates0=variates0, variates1=variates1, reps=reps)
            if np.isnan(tr):
                err_nan += 1
            else:
                trs.append(tr)
            taj_D.append(selectiontest.calculate_D(sfs))
            variant_array = list()
    LOGGER.log_message(str(not_enough), label='# Discarded for insufficient segregating sites: ')
    LOGGER.log_message(str(err_nan),    label='# Discarded for returning nan value for rho :   ')
    return trs, taj_D, sfs_list


@click.command()
@click.argument('job_no')
@click.argument('n', type=int)
@click.argument('growth_rate', type=float)
@click.option('-seg', 'seg_sites', type=int, default=None)
@click.option('-t', 'theta', type=float, default=None)
@click.option('-N', '--pop_size', type=float, default=100000)
@click.option('-rep', '--reps', type=int, default=10000)
@click.option('-sho', '--sho', type=float, default=None)
@click.option('-she', '--she', type=float, default=None)
@click.option('-sf', '--sf', type=float, default=None)
@click.option('-e', '--events_file', default=None)
@click.option('-r', '--recomb_rate', default=0, type=float)
@click.option('-d', '--dirx', default='data', type=click.Path(),
              help='Directory name for data and log files. Default is data')
def main(reps, job_no, pop_size, n, seg_sites, theta, growth_rate, sho, she, sf, events_file, recomb_rate, dirx):
    start_time = time()
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
    LOGGER.log_message('Name = ' + scipy.__name__ + ', version = ' + scipy.__version__, label=label)
    LOGGER.log_message('Name = ' + msprime.__name__ + ', version = ' + msprime.__version__, label=label)
    LOGGER.log_message('Name = ' + selectiontest.__name__ + ', version = ' + selectiontest.__version__, label=label)
    results = dict()

    print('Neutral population')
    msms_out = run_simulations(reps, pop_size, n, theta, seg_sites, 0, None, None, None, None, recomb_rate)
    variates0 = np.empty((reps, n - 1), dtype=float)
    for i, q in enumerate(selectiontest.sample_wf_distribution(n, reps)):
        variates0[i] = q
    variates1 = selectiontest.sample_uniform_distribution(n, reps)
    trs, taj_D, sfs_list = process_simulation_output(msms_out, variates0, variates1, reps)
    results['rho_false'] = trs
    results['taj_false'] = taj_D
    sfs_list = np.array(sfs_list)
    outfile_name = 'data/sfs_neutral_' + job_no + '.pklz'
    with gzip.open(outfile_name, 'wb') as outfile:
        pickle.dump(sfs_list, outfile)
    outfile = open(outfile_name, 'r')
    LOGGER.output_file(outfile.name)
    LOGGER.log_message(str(np.mean(sfs_list, axis=0)), label='Mean sfs constant population'.ljust(50))

    print('Non-neutral population')
    msms_out = run_simulations(reps, pop_size, n, theta, seg_sites, growth_rate, sho, she, sf, events_file, recomb_rate)
    variates0 = np.empty((reps, n - 1), dtype=float)
    for i, q in enumerate(selectiontest.sample_wf_distribution(n, reps)):
        variates0[i] = q
    variates1 = selectiontest.sample_uniform_distribution(n, reps)
    trs, taj_D, sfs_list = process_simulation_output(msms_out, variates0, variates1, reps)
    results['rho_true'] = trs
    results['taj_true'] = taj_D
    sfs_list = np.array(sfs_list)
    outfile_name = dirx + '/sfs_non_neutral_' + job_no + '.pklz'
    with gzip.open(outfile_name, 'wb') as outfile:
        pickle.dump(sfs_list, outfile)
    outfile = open(outfile_name, 'r')
    LOGGER.output_file(outfile.name)
    LOGGER.log_message(str(np.mean(sfs_list, axis=0)), label='Mean sfs non-neutral population'.ljust(50))

    fname = dirx + '/roc_data_' + job_no + '.pklz'
    with gzip.open(fname, 'wb') as outfile:
        pickle.dump(results, outfile)
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))


if __name__ == "__main__":
    main()
