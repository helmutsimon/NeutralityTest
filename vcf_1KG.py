# coding=utf-8


import numpy as np
import sys
import pysam
from vcf import Reader        # https://pypi.org/project/PyVCF/
from pyliftover import LiftOver
from collections import Counter
from scipy.special import binom
from scipy.stats import multinomial, expon
from scipy.stats import dirichlet
import pandas as pd


#def gt_total_mutations(record, pop):
#    #record.INFO['AC'] does the same thing.
#    result = 0
#    for proband in record.samples:
#        gt = proband.gt_alleles
#        result += (int(gt[0]) + int(gt[1]))
#    return result


def pi_calc_Counter(n, sfs):
    """Calculate mean number of pairwise differences from a site frequency spectrum in
    form of Counter object."""
    del sfs[0]
    temp = dict()
    for key in sfs.keys():
        temp[key] = key * (n - key) * sfs[key]
    pi = sum(temp.values()) * 2 / (n * (n - 1))
    #print(pi)
    return pi


def calculate_D_Counter(n, sfs):
    """Calculate Tajima's D from a site frequency spectrum in the form of a Counter object.
    Ensure that the Counter object does not have a key=0."""
    seg_sites = sum(sfs.values())
    pi = pi_calc_Counter(n, sfs)
    #n = len(sfs) + 1
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


def get_ERM_matrix(n):
    ERM_matrix = np.zeros((n - 1, n - 1))
    for m in range(n - 1):
        for k in range(n - 1):
            ERM_matrix[m, k] = (k + 2) * binom(n - m - 2, k) / binom(n - 1, k + 1)
    return ERM_matrix


def test_neutrality_Counter(n, sfs_c, random_state=None, variates=None, reps=50000):
    """Calculate odds ratio for neutral/not neutral from site frequency spectrum in form
    of a Counter object."""
    j_n = np.diag(1 / np.arange(2, n + 1))
    erm = get_ERM_matrix(n)
    avge_mx = erm.dot(j_n)
    kvec = np.arange(2, n + 1, dtype=int)
    if variates is None:
        variates = expon.rvs(scale=1 / binom(kvec, 2), size=(reps, n - 1), random_state=random_state)
    total_branch_lengths = variates.dot(kvec)
    rel_branch_lengths = np.diag(1 / total_branch_lengths).dot(variates)
    qvars = (erm.dot(rel_branch_lengths.T)).T
    sfs = np.zeros(n - 1, int)
    for i in sfs_c.keys():
        sfs[i - 1] = sfs_c[i]
    h1 = np.mean(multinomial.pmf(sfs, np.sum(sfs), qvars))
    sample = dirichlet.rvs(np.ones(n - 1), size=reps, random_state=random_state)
    sample = avge_mx.dot(sample.T)
    pmfs = multinomial.pmf(sfs, np.sum(sfs), sample.T)
    h2 = np.mean(pmfs)
    if h1 == 0 or h2 == 0:
        print(sfs, 'h1 = ', h1, 'h2 = ', h2)
        if h1 != 0:
            h2 = sys.float_info.min
    return h1 / h2


def get_sfs(vcf_file, panel, chrom, start, end, select_chr=True):
    """Get SFS from sequence for given population. The panel file is used to select probands."""
    n = panel.shape[0]
    if not select_chr:
        n = 2 * n
    snps = vcf_file.fetch(str(chrom), start, end)
    count, anc_count = 0, 0
    allele_counts = list()
    non_seg_snps = list()
    for record in snps:
        allele_count = 0
        if record.is_snp:
            count += 1
            #Test the ancestral is one of the alleles
            if record.INFO['AA'][0] not in [record.REF, record.ALT]:
                continue
            anc_count += 1
            for proband in record.samples:
                if proband.sample in panel.index:
                    gt = proband.gt_alleles
                    if select_chr:
                        allele_count += int(gt[0])
                    else:
                        allele_count += int(gt[0]) + int(gt[1])
                #print(record.ID.ljust(11), record.POS, record.REF, record.INFO['AA'][0], record.ALT[0],
                 #         "%3d" % allele_count, "%.6f" % (allele_count / 5008), record.INFO['AF'])
            if allele_count < n:    #Some SNPs may not segregate in some subpopulations.
                allele_counts.append(allele_count)
            else:
                non_seg_snps.append(record.ID)
    #print('Total SNPs               =', count)
    #print('SNPs with valid ancestor =', anc_count)
    print('Sample size              =', n)
    sfs_c = Counter(allele_counts)
    del sfs_c[0]
    print('Seg. sites in population =', sum(sfs_c.values()))
    sfs = np.zeros(n - 1, int)
    for i in sfs_c.keys():
        sfs[i - 1] = sfs_c[i]
    return sfs, n, non_seg_snps


def create_heatmap_table(results, panel_all, statistic):
    results['rlnt'] = np.log10(results['rlnt'])
    panel2 = panel_all[['pop', 'super_pop']]
    pop_tab = panel2.drop_duplicates(subset=['pop', 'super_pop'])
    heat_table = results.pivot(index='pop', columns='segstart', values=statistic)
    heat_table = heat_table.merge(pop_tab, on='pop')
    heat_table = heat_table.sort_values(['super_pop'])
    heat_table = heat_table.drop(columns='super_pop')
    heat_table.set_index('pop', inplace=True)
    return heat_table


def count_haplotypes(vcf_file, panel, pop, chrom, start, end):
    """Analyse vcf data and return a table of haplotypes and their counts for a given population
    and genome segment."""
    snps = vcf_file.fetch(str(chrom), start, end)
    columns = [snp.ID for snp in snps]
    columns = list(set(columns))
    count = 0
    snps = vcf_file.fetch(str(chrom), start, end)
    panel = panel[panel['pop'] == pop]
    index = [str(x) for x in panel.index]
    result = np.zeros((len(index), len(columns)), dtype=int)
    result = pd.DataFrame(result, index=index, columns=columns)
    for record in snps:
        if record.is_snp:
            for proband in record.samples:
                if proband.sample in [str(x) for x in panel.index]:
                    gt = proband.gt_alleles
                    if gt[0] != '0':
                        count += 1
                        result.loc[proband.sample, record.ID] = 1
    assert count == result.sum().sum(), "Allele count error."
    return result.groupby(columns).size().reset_index(name='Count')

