# coding=utf-8

"""
Functions used in analysis of 1KG vcf data.

"""
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from matplotlib.backends.backend_pdf import PdfPages


def get_sfs(vcf_file, panel, chrom, start, end, select_chr=True):
    """
    Get SFS from vcf data for given population. The panel file is used to select probands.

    """
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
            # Test the ancestral is one of the alleles
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
    sfs_c = Counter(allele_counts)
    del sfs_c[0]
    sfs = np.zeros(n - 1, int)
    for i in sfs_c.keys():
        sfs[i - 1] = sfs_c[i]
    return sfs, n, non_seg_snps


def create_heatmap_table(results, panel_all, statistic):
    """
    Reformat data from format produced internally by analyse_region_by_population.py for seaborn heatmap.

    """
    panel2 = panel_all[['pop', 'super_pop']]
    pop_tab = panel2.drop_duplicates(subset=['pop', 'super_pop'])
    heat_table = results.pivot(index='pop', columns='segstart', values=statistic)
    heat_table = heat_table.merge(pop_tab, on='pop')
    heat_table = heat_table.sort_values(['super_pop'])
    heat_table = heat_table.drop(columns='super_pop')
    heat_table.set_index('pop', inplace=True)
    return heat_table


def print_heatmap_pdf(pdfname, heat_table, colors, label, vmin, vmax, fname):
    """
    Print heatmap as pdf.

    """
    with PdfPages(pdfname) as pdf:
        fig = plt.figure(figsize=(20, 7))
        sns.set_style("whitegrid")
        cmap = matplotlib.colors.ListedColormap(colors)  # Create a new colormap with colors
        ax = sns.heatmap(heat_table, cmap=cmap, cbar_kws={'label': label}, vmin=vmin, vmax=vmax)
        ax.set_xlabel('Segment (GRCh37 coordinates)')
        ax.set_ylabel('Population')
        d = pdf.infodict()
        d['Title'] = 'Selection heatmap for chromosome 2q11.1 ' + label
        d['Author'] = 'H. Simon'
        d['Subject'] = 'Datafile: ' + fname
        d['CreationDate'] = datetime.datetime.today()
    return fig


def count_haplotypes(vcf_file, panel, pop, chrom, start, end):
    """
    Analyse vcf data and return a table of haplotypes and their counts for a given population
    and genome segment.

    """
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

