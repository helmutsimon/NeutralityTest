# coding=utf-8


""" Calculates neutrality odds and Tajima's D for neutral and non-neutral data simulated
    with MSMS. Simulations are conditioned on the number of segregating sites.
    The data can be used in roc curves.

    Parameters are job no., population size, sample size, sequence length, (nucleotide) mutation rate,
    population growth rate, demographic events file name (optional), recombination rate (optional),
    print demographic history flag (optional).

    A sample run statement is:

    nohup python3 /Users/helmutsimon/repos/NeutralityTest/simulate_roc_data.py cd01 10 0.0001 10 > srd_cd01.txt &
    """

import numpy as np
import os, sys
import gzip, pickle
from time import time
from collections import Counter
import msprime
import subprocess
import click
import pandas as pd
import scipy
from scipy.special import binom
from scipy.stats import multinomial, expon
from scipy.stats import dirichlet
from scitrack import CachingLogger, get_file_hexdigest

LOGGER = CachingLogger(create_dir=True)


def get_ERM_matrix(n):
    ERM_matrix = np.zeros((n - 1, n - 1))
    for m in range(n - 1):
        for k in range(n - 1):
            ERM_matrix[m, k] = (k + 2) * binom(n - m - 2, k) / binom(n - 1, k + 1)
    return ERM_matrix


def compute_sfs(variant_array):
    """Compute the site frequency spectrum from a variant array output by MSMS."""
    n = variant_array.shape[1]
    occurrences = np.sum(variant_array, axis=1)
    sfs = Counter(occurrences)
    sfs = [sfs[i] for i in range(1, n)]
    return np.array(sfs)


def test_neutrality(sfs, variates, projdir=None, tree_law='yule', mfilename=None,
                    concentration=1, reps=50000):
    """Calculate the odds ratio for neutral / not neutral."""
    n = len(sfs) + 1
    j_n = np.diag(1 / np.arange(2, n + 1))
    erm = get_ERM_matrix(n)
    if tree_law == 'yule':
        avge_mx = erm.dot(j_n)
    elif tree_law == 'uniform':
        if mfilename is None:
            mfilename = projdir + "/bayescoalescentest/data/matrix_list_" + str(n) + '.pklz'
        with gzip.open(mfilename, 'rb') as tree_matrices:
            tree_matrices = pickle.load(tree_matrices)
        tree_matrices = [m.T for m in tree_matrices]
        n_trees = len(tree_matrices)
        avge_mx = sum(tree_matrices) / n_trees
        avge_mx = avge_mx.dot(j_n)
    else:
        raise ValueError('Invalid tree law ' + tree_law)
    kvec = np.arange(2, n + 1, dtype=int)
    total_branch_lengths = variates @ kvec
    rel_branch_lengths = list()
    for row, total_length in zip(variates, total_branch_lengths):
        rel_row = row / total_length
        rel_branch_lengths.append(rel_row)
    rel_branch_lengths = np.array(rel_branch_lengths)
    #rel_branch_lengths = np.diag(1 / total_branch_lengths) @ variates
    qvars = (erm @ rel_branch_lengths.T).T
    h1 = np.mean(multinomial.pmf(sfs, np.sum(sfs), qvars))
    sample = dirichlet.rvs(np.ones(n - 1) / concentration, size=reps)
    sample = avge_mx @ sample.T
    pmfs = multinomial.pmf(sfs, np.sum(sfs), sample.T)
    h2 = np.mean(pmfs)
    if h1 == 0 or h2 == 0:
        print(sfs, 'h1 = ', h1, 'h2 = ', h2)
        if h1 != 0:
            h2 = sys.float_info.min
    return h1 / h2


def pi_calc(sfs):
    """Calculate the mean number of pairwaise differences from a site frequency spectrum."""
    sfs = np.array(sfs)
    n = len(sfs) + 1
    g1 = np.arange(1, n)
    g2 = n - g1
    g3 = g1 * g2 * sfs
    pi = np.sum(g3) * 2 / (n * (n - 1))
    return pi


def calculate_D(sfs):
    """Calculate Tajima's D from a site frequency spectrum."""
    seg_sites = np.sum(sfs)
    pi = pi_calc(sfs)
    n = len(sfs) + 1
    n_seq = np.arange(1, n)
    a1 = (1 / n_seq).sum()
    a2 = ((1 / n_seq) ** 2).sum()
    b1 = (n + 1) / (3 * (n - 1))
    b2 = (2 * (n ** 2 + n + 3)) / (9 * n * (n - 1))
    c1 = b1 - (1 / a1)
    c2 = b2 - ((n + 2) / (a1 * n)) + (a2 / a1 ** 2)
    e1 = c1 / a1
    e2 = c2 / (a1 ** 2 + a2)
    tajD = (pi - (seg_sites / a1)) / np.sqrt(e1 * seg_sites + e2 * seg_sites * (seg_sites - 1))
    return tajD


def read_demography_file(filename, pop_size):
    """Read demographic history file and translate into MSMS commands. The file uses absolute
    population size and exponential growth rate per generation. These are scaled by 4N_0 (pop_size)
    for MSMS. The input file format used is compatible with msprime and the relevant msprime function
    is used to generate a description of the demographic history."""
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
        time = str(row['time'] / (4 * pop_size))
        size_change = str(row['size'] / pop_size)
        growth_rate = str(row['rate'] * 4 * pop_size)
        ppc = msprime.PopulationParametersChange(time=row['time'],
                    initial_size=row['size'], growth_rate=row['rate'])
        demographic_events.append(ppc)
        event = 'time: ' + str(row['time']) + ', size: ' + str(row['size']) +\
                ', rate: ' + str(row['rate'])
        #print(event)
        LOGGER.log_message(event, label='Demographic change event'.ljust(50))
        cmds = ["-eN", time, size_change, "-eG", time, growth_rate]
        dem_cmds = dem_cmds + cmds
    return dem_cmds, demographic_events


def run_simulations(reps, pop_size, n, recombination_rate, seg_sites,
        growth_rate, SAA, SaA, SF, events_file, tree_law, mfilename, concentration, projdir, print_history):
    if SAA is not None:
        if SaA is None:
            SaA = SAA / 2
        if events_file is not None:
            raise ValueError('Cannot combine demographic change events with selection.')
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
    sfs_list, trs, taj_D = list(), list(), list()
    kvec = np.arange(2, n + 1, dtype=int)
    wf_variates = expon.rvs(scale=1 / binom(kvec, 2), size=(reps, n - 1))
    growth_rate_scaled = growth_rate * 4 * pop_size
    msms_cmd = ["java", "-jar", "msms.jar", str(n), str(reps), "-s",  str(seg_sites),  "-N",
                    str(pop_size), "-G", str(growth_rate_scaled)] + events_commands
    if SAA is not None:
        msms_cmd = msms_cmd + ["-SAA", str(SAA), "-SaA", str(SaA), "-SF", str(SF)]
    if recombination_rate != 0:
        msms_cmd = msms_cmd + ["-r", str(recombination_rate)]
    print(msms_cmd)
    msms_out = subprocess.check_output(msms_cmd)
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
            sfs = compute_sfs(variant_array)
            sfs_list.append(sfs)
            tr = test_neutrality(sfs, wf_variates, projdir, tree_law, mfilename, concentration, reps)
            if np.isnan(tr):
                continue
            assert not np.isinf(tr), 'tr inf ' + str(tr)
            trs.append(tr)
            taj_D.append(calculate_D(sfs))
            variant_array = list()
    return trs, taj_D, sfs_list


@click.command()
@click.argument('job_no')
@click.argument('n', type=int)
@click.argument('growth_rate', type=float)
@click.argument('seg_sites', type=int)
@click.option('-p', '--pop_size', type=float, default=100000)
@click.option('-len', '--length', type=float, default=1000)
@click.option('-mu', '--mutation_rate', default='2.5e-8')
@click.option('-rep', '--reps', type=int, default=10000)
@click.option('-sho', '--sho', type=float, default=None)
@click.option('-she', '--she', type=float, default=None)
@click.option('-sf', '--sf', type=float, default=None)
@click.option('-e', '--events_file', default=None)
@click.option('-r', '--recombination_rate', default=0, type=float)
@click.option('-p', '--print_history', default=True, type=bool)
@click.option('-law', '--tree_law', default='yule')
@click.option('-m', '--mfilename', default=None)
@click.option('-c', '--concentration', type=float, default=1)
@click.option('-d', '--dir', default='data', type=click.Path(),
              help='Directory name for data and log files. Default is data')
def main(reps, job_no, pop_size, n, length, mutation_rate, seg_sites, growth_rate, sho, she, sf,
         events_file, recombination_rate, print_history, tree_law, mfilename, concentration, dir):
    start_time = time()
    abspath = os.path.abspath(__file__)
    projdir = "/".join(abspath.split("/")[:-2])
    np.set_printoptions(precision=3)                #
    if not os.path.exists(dir):
        os.makedirs(dir)
    LOGGER.log_file_path = dir + "/" + str(os.path.basename(__file__)) + '_' + job_no + ".log"
    LOGGER.log_args()
    LOGGER.log_message(get_file_hexdigest(__file__), label="Hex digest of script.".ljust(17))
    try:
        LOGGER.log_message(str(os.environ['CONDA_DEFAULT_ENV']), label="Conda environment.".ljust(17))
    except KeyError:
        pass
    LOGGER.log_message('Name = ' + np.__name__ + ', version = ' + np.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + pd.__name__ + ', version = ' + pd.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + scipy.__name__ + ', version = ' + scipy.__version__,
                       label="Imported module".ljust(30))
    pop_size = int(pop_size)
    length = int(length)
    results_false = pd.DataFrame()
    results_true = pd.DataFrame()
    print('Neutral population')
    trs, taj_D, sfs_list = run_simulations(reps, pop_size, n, recombination_rate, seg_sites, 0,
                None, None, None, None, tree_law, mfilename, concentration, projdir, print_history)
    results_false['bayes'] = trs
    results_false['taj'] = taj_D
    sfs_list = np.array(sfs_list)
    LOGGER.log_message(str(np.mean(sfs_list, axis=0)), label='Mean sfs constant population'.ljust(50))
    print('Non-neutral population')
    trs, taj_D, sfs_list = run_simulations(reps, pop_size, n, recombination_rate, seg_sites, growth_rate,
                sho, she, sf, events_file, tree_law, mfilename, concentration, projdir, print_history)
    results_true['bayes'] = trs
    results_true['taj'] = taj_D
    sfs_list = np.array(sfs_list)
    LOGGER.log_message(str(np.mean(sfs_list, axis=0)), label='Mean sfs non-neutral population'.ljust(50))
    fname = 'data/roc_data_false_' + job_no + '.csv'
    results_false.to_csv(fname)
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    fname = 'data/roc_data_true_' + job_no + '.csv'
    results_true.to_csv(fname)
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))


if __name__ == "__main__":
    main()

