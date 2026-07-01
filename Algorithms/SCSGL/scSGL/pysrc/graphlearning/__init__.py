# scSGL - a python package for fene regulatory network inference using graph signal processing based
# signed graph learning
# Copyright (C) 2021 Abdullah Karaaslanli <evdilak@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pandas as pd

from scipy import sparse
from scipy.spatial.distance import squareform

from . import unsigned
from . import signed
from ..associations import correlation, dotprod, proprho, zikendall

ASSOCIATIONS = {"dotprod": dotprod.calc,
                "correlation": correlation.calc,
                "proprho": proprho.calc,
                "zikendall": zikendall.calc}

def _find_bs_upper_bound(k, d, density):
    for i in range(1, 100):
        w = unsigned.learn(k, d, i)
        densities_est = np.count_nonzero(w)/len(w)
        if densities_est > density:
            return i

def _binary_search(k, d, density_pos, density_neg):
    apos_done = False
    aneg_done = False

    apos_min = 0
    aneg_min = 0

    # TODO: Find better upper bound
    if density_pos > 0:
        apos_max = _find_bs_upper_bound(k, d, density_pos)
    else:
        apos_max = 0
        apos_done = True
        wpos = np.zeros(len(k))

    if density_neg > 0:
        aneg_max = _find_bs_upper_bound(k, d, density_neg)
    else:
        aneg_max = 0
        aneg_done = True
        wneg = np.zeros(len(k))

    densities_pos = np.zeros(50)
    densities_neg = np.zeros(50)
    for i in range(50):
        apos = (apos_min + apos_max)/2
        aneg = (aneg_min + aneg_max)/2

        if aneg == 0: # Learn only positive edges
            wpos = unsigned.learn(k, d, apos)
            densities_pos[i] = np.count_nonzero(wpos)/len(wpos)
        elif apos == 0: # Learn only negative edges
            wneg = unsigned.learn(-k, d, aneg)
            densities_neg[i] = np.count_nonzero(wneg)/len(wneg)
        else: 
            wpos, wneg= signed.learn(k, d, apos, aneg, lpos_init="zeros", 
                lneg_init="zeros")
            densities_pos[i] = np.count_nonzero(wpos)/len(wpos)
            densities_neg[i] = np.count_nonzero(wneg)/len(wneg)

        # print("Current densities: {:.2f}    {:.2f}".format(densities_pos[i], densities_neg[i]))

        # Check if desired density is obtained for positive part
        if not apos_done:
            if np.abs(density_pos - densities_pos[i]) < 1e-2:
                apos_done = True
            elif density_pos < densities_pos[i]:
                apos_max = apos
            elif density_pos > densities_pos[i]:
                apos_min = apos

        # Check if desired density is obtained for negative part
        if not aneg_done:
            if np.abs(density_neg - densities_neg[i]) < 1e-2:
                aneg_done = True
            elif density_neg < densities_neg[i]:
                aneg_max = aneg
            elif density_neg > densities_neg[i]:
                aneg_min = aneg

        # If desired densities are obtained, break
        if (apos_done and aneg_done):
            break

        # If binary search stuck, break
        if i>2:
            if np.abs(densities_pos[i] - densities_pos[i-1]) < 1e-3 and \
            np.abs(densities_pos[i] - densities_pos[i-2]) < 1e-3 and \
            np.abs(densities_neg[i] - densities_neg[i-1]) < 1e-3 and \
            np.abs(densities_neg[i] - densities_neg[i-2]) < 1e-3:
                break

    return wpos, -wneg   

def _sparse_corrcoef(counts):
    counts = counts.tocsr()
    n_samples = counts.shape[1]
    if n_samples == 0:
        raise ValueError("Sparse correlation requires at least one sample.")

    row_sums = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    row_sq_sums = np.asarray(counts.multiply(counts).sum(axis=1)).ravel().astype(np.float64)
    norms = row_sq_sums - (row_sums * row_sums / float(n_samples))
    norms[norms < 0] = 0
    norms = np.sqrt(norms)

    dot = (counts @ counts.T).toarray().astype(np.float64, copy=False)
    centered_dot = dot - (row_sums[:, None] * row_sums[None, :] / float(n_samples))
    denom = norms[:, None] * norms[None, :]
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = centered_dot / denom
    corr[~np.isfinite(corr)] = 0
    np.fill_diagonal(corr, 1)
    return corr


def _association_matrix(X, assoc):
    if sparse.issparse(X):
        if assoc == "correlation":
            return _sparse_corrcoef(X)
        if assoc == "dotprod":
            return (X @ X.T).toarray()
        return ASSOCIATIONS[assoc](X.toarray())
    return ASSOCIATIONS[assoc](X)


def learn_signed_graph(X, pos_density, neg_density, assoc="dotprod", gene_names = None, 
                       verbose=False):
    # TODO: Docstring
    # TODO: Input check

    if assoc not in ASSOCIATIONS:
        raise ValueError(f"Unknown association type: {assoc}")

    if gene_names is None:
        gene_names = np.arange(1, X.shape[0]+1)
    else:
        gene_names = np.asarray(gene_names)

    # Check if there is any genes that has no expression at all
    if sparse.issparse(X):
        nnzeros = np.asarray(X.getnnz(axis=1) != 0).ravel()
    else:
        nnzeros = np.count_nonzero(X, axis=1) != 0
    X_nnzeros = X[nnzeros, :]

    # Calculate association matrix
    K = _association_matrix(X_nnzeros, assoc)
    k = K[np.triu_indices_from(K, k=1)]
    if len(k) == 0:
        return pd.DataFrame({"Gene1": [], "Gene2": [], "EdgeWeight": []})

    max_abs_k = np.max(np.abs(k)) if len(k) else 0
    if max_abs_k > 0:
        k /= max_abs_k
        d = np.diag(K)/max_abs_k
    else:
        d = np.diag(K)

    # Learn graph with desired density
    if verbose:
        print("Estimating a graph whose positive and negative edges densities are",
              "{:.3f} and {:.3f}...".format(pos_density, neg_density))

    wpos, wneg = _binary_search(k, d, pos_density, neg_density)

    pos_density_est = np.count_nonzero(wpos)/len(wpos)
    neg_density_est = np.count_nonzero(wneg)/len(wneg)

    if verbose:
        print("Graph is found. Its positive and negative edge densities are {:.3f} and {:.3f}"\
            .format(pos_density_est, neg_density_est))

    return convert_df(gene_names[nnzeros], wpos, wneg)

def convert_df(gene_names, lpos, lneg):
    gene_names = np.asarray(gene_names)
    L = squareform(np.squeeze(lpos + lneg))
    rows, cols = np.nonzero(L)
    directed = rows != cols
    rows = rows[directed]
    cols = cols[directed]

    return pd.DataFrame({
        "Gene1": gene_names[rows],
        "Gene2": gene_names[cols],
        "EdgeWeight": L[rows, cols],
    })
