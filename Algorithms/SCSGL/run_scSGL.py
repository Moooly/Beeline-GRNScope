# Please refer to https://github.com/SPLab-aviyente/scSGL/blob/main/notebooks/demo.ipynb

import argparse
import pandas as pd #to load read GSD dataset
import numpy as np
import sys
from scipy import sparse
sys.path.append('scSGL') #to add a path to search for the requested module

from pysrc.graphlearning import learn_signed_graph


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run scSGL algorithm.') 
    parser.add_argument('--expression_file', 
        help='Path to ExpressionData file')
    parser.add_argument('--ground_truth_net_file', 
        help='Path to groundTruthNetwork file')
    parser.add_argument('--pos_density', default='0.45', #to control the density of positive part of the learned signed graph
        help='Positive density')
    parser.add_argument('--neg_density', default='0.45', #to control the density of negative part of the learned signed graph
        help='Negative density')
    parser.add_argument('--assoc', default='correlation', #to infer a signed graph with correlation kernel
        help='Association type')
    parser.add_argument('--out_file',
        help='Path to output file')
    parser.add_argument('--max_regulators_per_target', type=int, default=0,
        help='Keep only this many strongest regulators per target. Use 0 to write all edges.')
    parser.add_argument('--matrix_format', choices=['auto', 'sparse', 'dense'], default='auto',
        help='Expression matrix backend. auto keeps sparse for low-density matrices.')
    parser.add_argument('--sparse_density_threshold', type=float, default=0.35,
        help='In auto mode, keep sparse when nonzero density is at or below this value.')
    parser.add_argument('--csv_chunk_size', type=int, default=1000,
        help='Rows per chunk while reading expression CSV.')

    return parser

def parse_arguments():
    parser = get_parser()
    opts = parser.parse_args()

    return opts

def read_expression_matrix(path, matrix_format='auto', sparse_density_threshold=0.35, csv_chunk_size=1000):
    matrix_format = str(matrix_format or 'auto').lower()
    chunks = pd.read_csv(path, sep=None, engine='python', index_col=0, chunksize=max(1, csv_chunk_size))
    gene_names = []
    sparse_chunks = []
    dense_chunks = [] if matrix_format == 'dense' else None
    nonzero_count = 0
    cell_count = None

    for chunk in chunks:
        gene_names.extend(chunk.index.astype(str).tolist())
        values = chunk.to_numpy(dtype=np.float32, copy=False)
        if cell_count is None:
            cell_count = values.shape[1]
        elif cell_count != values.shape[1]:
            raise ValueError(f"{path} has inconsistent cell counts across chunks.")

        if matrix_format == 'dense':
            dense_chunks.append(values.copy())
            nonzero_count += int((values != 0).sum())
            continue

        sparse_chunk = sparse.csr_matrix(values)
        sparse_chunks.append(sparse_chunk)
        nonzero_count += int(sparse_chunk.nnz)

    if not gene_names:
        raise ValueError(f"{path} contains no expression genes.")

    gene_count = len(gene_names)
    total_values = gene_count * (cell_count or 0)
    density = nonzero_count / total_values if total_values else 0.0

    if matrix_format == 'dense':
        matrix = dense_chunks[0] if len(dense_chunks) == 1 else np.vstack(dense_chunks)
        print(f"Loaded dense scSGL expression matrix {matrix.shape}; density {density:.4f}", flush=True)
        return matrix.astype(np.float32, copy=False), np.array(gene_names)

    matrix = sparse_chunks[0].tocsr() if len(sparse_chunks) == 1 else sparse.vstack(sparse_chunks, format='csr', dtype=np.float32)
    if matrix_format == 'sparse' or density <= sparse_density_threshold:
        print(f"Loaded sparse scSGL expression matrix {matrix.shape}; nnz {matrix.nnz}; density {density:.4f}", flush=True)
        return matrix, np.array(gene_names)

    dense_matrix = matrix.toarray().astype(np.float32, copy=False)
    print(f"Loaded dense scSGL expression matrix {dense_matrix.shape}; density {density:.4f}", flush=True)
    return dense_matrix, np.array(gene_names)

def main(args):
    #python run_scSGL.py --expression_file scSGL/data/inputs/GSD/ExpressionData.csv --ground_truth_net_file scSGL/data/inputs/GSD/GroundTruthNetwork.csv --out_file outFile.txt

    opts = parse_arguments()
    expression_matrix, gene_names = read_expression_matrix(
        opts.expression_file,
        matrix_format=opts.matrix_format,
        sparse_density_threshold=opts.sparse_density_threshold,
        csv_chunk_size=opts.csv_chunk_size,
    )

    #Learn signed graph with the parameters
    G = learn_signed_graph(expression_matrix, pos_density=float(opts.pos_density), neg_density=float(opts.neg_density),
                                assoc=opts.assoc, gene_names=gene_names)
    #G is a dataframe with each row indicating an edge between two genes.
    #Each edge is also associated with a weight, which is either positive or negative depending on the sign of the edge.
    if opts.max_regulators_per_target and opts.max_regulators_per_target > 0:
        G = G.copy()
        G["EdgeWeight"] = pd.to_numeric(G["EdgeWeight"], errors="coerce")
        G = G.dropna(subset=["Gene1", "Gene2", "EdgeWeight"])
        G["_abs_weight"] = G["EdgeWeight"].abs()
        G = (
            G.sort_values(["Gene2", "_abs_weight", "Gene1"], ascending=[True, False, True])
             .groupby("Gene2", sort=False)
             .head(opts.max_regulators_per_target)
             .drop(columns=["_abs_weight"])
        )

    G.to_csv(opts.out_file, index = False, sep = '\t')  #to write the output file

if __name__ == "__main__":
    main(sys.argv)
