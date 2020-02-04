# coding=utf-8


""" Calculates neutrality odds and Tajima's D for neutral and non-neutral data simulated
    with msprime. The data can be used in roc curves.

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
import subprocess
import click
import pandas as pd
import msprime
import scipy
from scipy.special import binom
from scipy.stats import multinomial, expon
from scipy.stats import dirichlet
from scitrack import CachingLogger, get_file_hexdigest

abspath = os.path.abspath(__file__)
projdir = "/".join(abspath.split("/")[:-2])
sys.path.append(projdir + '/bayescoalescentest')

from shared import msprime_functions

LOGGER = CachingLogger(create_dir=True)


def get_ERM_matrix(n):
    ERM_matrix = np.zeros((n - 1, n - 1))
    for m in range(n - 1):
        for k in range(n - 1):
            ERM_matrix[m, k] = (k + 2) * binom(n - m - 2, k) / binom(n - 1, k + 1)
    return ERM_matrix


def test_neutrality(sfs, variates, projdir, tree_law, mfilename, concentration, reps=50000):
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
    #print(tree_law)
    #print(avge_mx)
    kvec = np.arange(2, n + 1, dtype=int)
    total_branch_lengths = variates @ kvec
    rel_branch_lengths = np.diag(1 / total_branch_lengths) @ variates
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


def test_neutrality2(sfs, random_state=None, reps=50000):
    """Calculate the odds ratio for neutral / not neutral."""
    n = len(sfs) + 1
    j_n = np.diag(1 / np.arange(2, n + 1))
    erm = get_ERM_matrix(n)
    avge_mx = erm.dot(j_n)
    kvec = np.arange(2, n + 1, dtype=int)
    variates = expon.rvs(scale=1 / binom(kvec, 2), size=(reps, n - 1), random_state=random_state)
    total_branch_lengths = variates.dot(kvec)
    rel_branch_lengths = np.diag(1 / total_branch_lengths).dot(variates)
    qvars = (erm.dot(rel_branch_lengths.T)).T
    h1 = np.mean(multinomial.pmf(sfs, np.sum(sfs), qvars))
    sample = dirichlet.rvs(np.ones(n - 1), size=reps, random_state=random_state)
    #sample = avge_mx @ sample.T
    sample = avge_mx.dot(sample.T)
    pmfs = multinomial.pmf(sfs, np.sum(sfs), sample.T)
    h2 = np.mean(pmfs)
    if h1 == 0 or h2 == 0:
        print(sfs, 'h1 = ', h1, 'h2 = ', h2)
        if h1 != 0:
            h2 = sys.float_info.min
    return h1 / h2


def folded_sfs(va1):
    n = va1.shape[0]
    ones = np.ones(va1.shape[1])
    rnum = str(va1.shape[0])
    sfs_list = list()
    for i in range(1, 2 ** n + 1):
        va = va1.copy()
        #print(i)
        fmt = '{0:0' + rnum + 'b}'
        lbin = fmt.format(i)
        binrep = [x for x in lbin]
        for j, value in enumerate(binrep):
            if value == '1':
                va.loc[[j]] = (va.loc[[j]] != ones).astype(int)
        sfs = msprime_functions.compute_sfs(va)
        sfs_list.append(sfs)
    return sfs_list


def test_folded(variant_array, reps=50000):
    sfs_list = folded_sfs(pd.DataFrame(variant_array))
    results = [test_neutrality2(sfs, reps) for sfs in sfs_list]
    return np.mean(results)


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


def relative_branch_lengths(coalescence_times):
    """Calculate relative branch lengths from coalescence_times."""
    n = len(coalescence_times) + 1
    shifted = np.concatenate((np.zeros(1), coalescence_times))[:-1]
    branch_lengths = np.flipud(coalescence_times - shifted)
    kvec = np.arange(2, n + 1, dtype=int)
    total_branch_lengths = branch_lengths.dot(kvec)
    # rel_branch_lengths = np.diag(1 / total_branch_lengths) @ branch_lengths
    rel_branch_lengths = branch_lengths / total_branch_lengths
    return rel_branch_lengths


def run_msprime_simulations(reps, pop_size, n, length, recombination_rate, mutation_rate, seg_sites,
        growth_rate, SAA, SaA, SF, events_file, tree_law, mfilename, concentration, print_history, folded=False):
    """Simulate a (large) population and take a random sample from it. We return the details of this sample tree
    (sfs, branch lengths and matrix) as well as averages from other population samples to provide data for empirical
    prior. In the first statement we generate a tree for the entire population size."""

    demographic_events = msprime_functions.read_demographic_history(events_file, print_history)
    population_configurations = [msprime.PopulationConfiguration(growth_rate=growth_rate,
                                                                 initial_size=pop_size, sample_size=n)]
    demographic_hist = msprime.DemographyDebugger(demographic_events=demographic_events,
                                                  population_configurations=population_configurations)
    if print_history:
        demographic_hist.print_history()
    tree_sequences = msprime.simulate(Ne=pop_size, population_configurations=population_configurations,
                                      demographic_events=demographic_events, length=length, recombination_rate=0, \
                                      mutation_rate=mutation_rate, num_replicates=reps)
    sfs_list, trs, taj_D = list(), list(), list()
    kvec = np.arange(2, n + 1, dtype=int)
    wf_variates = expon.rvs(scale=1 / binom(kvec, 2), size=(reps, n - 1))
    for tree_sequence in tree_sequences:
        tree = next(tree_sequence.trees())
        root = tree.get_root()
        leaves = list(tree.leaves(root))
        matrix, coalescence_times = msprime_functions.analyse_tree(n, tree, leaves)
        rel_branch_lengths = relative_branch_lengths(coalescence_times)
        erm = get_ERM_matrix(n)
        qvar = (erm @ rel_branch_lengths.T).T
        sfs = multinomial.rvs(seg_sites, qvar)
        sfs_list.append(sfs)
        taj_D.append(calculate_D(sfs))
        sfs_list.append(sfs)
        trs.append(test_neutrality(sfs, wf_variates, projdir, tree_law, mfilename, concentration, reps))
    return trs, taj_D, sfs_list


def run_msms_simulations(reps, pop_size, n, length, recombination_rate, mutation_rate, seg_sites,
        growth_rate, SAA, SaA, SF, events_file, tree_law, mfilename, concentration, print_history, folded=False):
    if events_file is not None:
        raise ValueError('Events file supplied for msms simulation.')
    if growth_rate != 0:
        raise ValueError('growth_rate must be zero for msms simulation.')
    sfs_list, trs, taj_D = list(), list(), list()
    kvec = np.arange(2, n + 1, dtype=int)
    wf_variates = expon.rvs(scale=1 / binom(kvec, 2), size=(reps, n - 1))
    if SAA is None:
        msms_cmd = ["java", "-jar", "msms.jar", str(n), str(reps), "-s",  str(seg_sites),  "-N", str(pop_size)]
    else:
        msms_cmd = ["java", "-jar", "msms.jar", str(n), str(reps), "-s",  str(seg_sites), "-SAA", str(SAA),
                    "-SaA", str(SaA), "-SF", str(SF), "-N", str(pop_size)]
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
            sfs = msprime_functions.compute_sfs(variant_array)
            #print(sfs)
            sfs_list.append(sfs)
            if folded == True:
                tr = test_folded(variant_array, reps=reps)
            else:
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
@click.option('-f', '--folded', default=False, type=bool)
@click.option('-m', '--mfilename', default=None)
@click.option('-c', '--concentration', type=float, default=1)
@click.option('-d', '--dir', default='data', type=click.Path(),
              help='Directory name for data and log files. Default is data')
def main(reps, job_no, pop_size, n, length, mutation_rate, seg_sites, growth_rate, sho, she, sf,
         events_file, recombination_rate, print_history, tree_law, folded, mfilename, concentration, dir):
    start_time = time()
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
    LOGGER.log_message('Name = ' + msprime.__name__ + ', version = ' + msprime.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + scipy.__name__ + ', version = ' + scipy.__version__,
                       label="Imported module".ljust(30))
    pop_size = int(pop_size)
    length = int(length)
    run_simulations = run_msprime_simulations
    if '/' in mutation_rate:
        mutation_rates = mutation_rate.split('/')
        mrate0 = float(mutation_rates[0])
        mrate1 = float(mutation_rates[1])
    else:
        mrate0 = float(mutation_rate)
        mrate1 = float(mutation_rate)
    if sho is not None:
        run_simulations = run_msms_simulations
        if she is None:
            she = sho / 2
        if events_file is not None:
            raise ValueError('Cannot have demographic change events with selection.')
    results_false = pd.DataFrame()
    results_true = pd.DataFrame()
    print('Neutral population')
    trs, taj_D, sfs_list = run_simulations(reps, pop_size, n, length, recombination_rate,
                mrate0, seg_sites, 0, None, None, None, None, tree_law, mfilename, concentration,
                print_history)
    results_false['bayes'] = trs
    results_false['taj'] = taj_D
    sfs_list = np.array(sfs_list)
    LOGGER.log_message(str(np.mean(sfs_list, axis=0)), label='Mean sfs constant population'.ljust(50))
    print('Non-neutral population')
    trs, taj_D, sfs_list = run_simulations(reps, pop_size, n, length, recombination_rate,
        mrate1, seg_sites, growth_rate, sho, she, sf, events_file, tree_law, mfilename, concentration,
        print_history, folded)
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

