from optparse import OptionParser
import csv
import math
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from arboreto.algo import diy
from arboreto.core import EARLY_STOP_WINDOW_LENGTH, SGBM_KWARGS
from distributed import Client

DEFAULT_GENIE3_TREES = 500
DEFAULT_GRNBOOST2_TREES = 5000
GENIE3_RF_KWARGS = {
    'n_jobs': 1,
    'n_estimators': DEFAULT_GENIE3_TREES,
    'max_features': 'sqrt',
}

def parseArgs(args):
    parser = OptionParser()

    parser.add_option('', '--algo', type = 'str',
                      help='Algorithm to run. Can either by GENIE3 or GRNBoost2')

    parser.add_option('', '--inFile', type='str',
                      help='Path to input tab-separated expression SamplesxGenes file')

    parser.add_option('', '--outFile', type = 'str',
                      help='File where the output network is stored')

    parser.add_option('', '--nWorkers', type='int', default=None,
                      help='Number of local Dask workers. Defaults to CPU count.')

    parser.add_option('', '--threadsPerWorker', type='int', default=1,
                      help='Number of threads per local Dask worker.')

    parser.add_option('', '--genie3Trees', type='int', default=DEFAULT_GENIE3_TREES,
                      help='Number of random forest trees to use for GENIE3.')

    parser.add_option('', '--grnboost2Trees', type='int', default=DEFAULT_GRNBOOST2_TREES,
                      help='Maximum number of boosting trees to use for GRNBoost2.')

    parser.add_option('', '--runRoot', type='str', default=None,
                      help='Batch mode root containing run_id/algorithm/working_dir folders.')

    parser.add_option('', '--runIds', type='str', default=None,
                      help='Comma-separated run IDs to execute in batch mode.')

    parser.add_option('', '--algorithmId', type='str', default=None,
                      help='Algorithm output directory name used in batch mode.')
    parser.add_option('', '--maxRegulatorsPerTarget', type='int', default=None,
                      help='Keep only this many strongest regulators per target in the output.')
    parser.add_option('', '--matrixFormat', type='choice', default='auto',
                      choices=('auto', 'sparse', 'dense'),
                      help='Expression matrix backend: auto, sparse, or dense. Defaults to auto.')
    parser.add_option('', '--sparseDensityThreshold', type='float', default=0.35,
                      help='In auto mode, keep sparse when nonzero density is at or below this value.')
    parser.add_option('', '--csvChunkSize', type='int', default=1000,
                      help='Rows per chunk while reading expression TSV into memory.')
    parser.add_option('', '--inputOrientation', type='choice', default='samplesByGenes',
                      choices=('samplesByGenes', 'genesByCells'),
                      help='Input matrix orientation. Defaults to samplesByGenes.')

    (opts, args) = parser.parse_args(args)

    return opts, args

def update_candidate(target_candidates, source, target, score, max_regulators_per_target):
    try:
        score = float(score)
    except (TypeError, ValueError):
        return
    if not source or not target or not math.isfinite(score):
        return

    candidates = target_candidates.setdefault(str(target), {})
    source = str(source)
    current_score = candidates.get(source)
    if current_score is not None:
        if abs(score) > abs(current_score):
            candidates[source] = score
        return

    if len(candidates) < max_regulators_per_target:
        candidates[source] = score
        return

    weakest_source, weakest_score = min(
        candidates.items(),
        key=lambda item: (abs(float(item[1])), str(item[0])),
    )
    if abs(score) > abs(float(weakest_score)):
        del candidates[weakest_source]
        candidates[source] = score


def write_network(network, out_file, max_regulators_per_target):
    if max_regulators_per_target is not None and max_regulators_per_target > 0:
        target_candidates = {}
        for edge in network.itertuples(index=False):
            update_candidate(
                target_candidates,
                getattr(edge, 'TF'),
                getattr(edge, 'target'),
                getattr(edge, 'importance'),
                max_regulators_per_target,
            )

        rows = []
        for target, candidates in target_candidates.items():
            for source, score in candidates.items():
                rows.append((source, target, score))
        rows.sort(key=lambda row: (-abs(float(row[2])), str(row[0]), str(row[1])))

        with open(out_file, 'w', newline='') as output_handle:
            writer = csv.writer(output_handle, delimiter='\t')
            writer.writerow(['TF', 'target', 'importance'])
            writer.writerows(rows)
        return

    network.to_csv(out_file, index = False, sep = '\t')


def read_samples_by_genes_matrix(in_file, matrix_format, csv_chunk_size):
    header = pd.read_csv(in_file, sep='\t', nrows=0).columns.tolist()
    if len(header) < 2:
        raise ValueError(
            f"{in_file} must be a tab-separated matrix with sample IDs in the first column "
            "and gene names in the remaining columns."
        )

    gene_names = header[1:]
    expression_dtypes = {column: 'float32' for column in gene_names}
    chunks = pd.read_csv(
        in_file,
        sep='\t',
        index_col=0,
        header=0,
        dtype=expression_dtypes,
        na_filter=False,
        chunksize=csv_chunk_size,
    )
    sparse_chunks = []
    row_count = 0
    nonzero_count = 0
    dense_chunks = [] if matrix_format == 'dense' else None

    for chunk in chunks:
        values = chunk.to_numpy(dtype='float32', copy=False)
        row_count += values.shape[0]

        if matrix_format == 'dense':
            dense_chunks.append(values.copy())
            nonzero_count += int((values != 0).sum())
            continue

        sparse_chunk = sparse.csr_matrix(values)
        sparse_chunks.append(sparse_chunk)
        nonzero_count += int(sparse_chunk.nnz)

    if row_count == 0:
        raise ValueError(f"{in_file} contains no expression rows.")

    return sparse_chunks, dense_chunks, gene_names, row_count, len(gene_names), nonzero_count


def read_genes_by_cells_matrix(in_file, matrix_format, csv_chunk_size):
    chunks = pd.read_csv(
        in_file,
        sep=None,
        engine='python',
        index_col=0,
        header=0,
        chunksize=csv_chunk_size,
    )
    sparse_chunks = []
    dense_chunks = [] if matrix_format == 'dense' else None
    gene_names = []
    gene_count = 0
    sample_count = None
    nonzero_count = 0

    for chunk in chunks:
        gene_names.extend(chunk.index.astype(str).tolist())
        values = chunk.to_numpy(dtype='float32', copy=False)
        if sample_count is None:
            sample_count = values.shape[1]
        elif sample_count != values.shape[1]:
            raise ValueError(f"{in_file} has inconsistent cell counts across chunks.")
        gene_count += values.shape[0]

        if matrix_format == 'dense':
            dense_chunks.append(values.copy())
            nonzero_count += int((values != 0).sum())
            continue

        sparse_chunk = sparse.csr_matrix(values)
        sparse_chunks.append(sparse_chunk)
        nonzero_count += int(sparse_chunk.nnz)

    if gene_count == 0 or not gene_names:
        raise ValueError(f"{in_file} contains no expression genes.")
    if sample_count is None or sample_count == 0:
        raise ValueError(f"{in_file} contains no expression cells.")

    if matrix_format == 'dense':
        genes_by_cells = dense_chunks[0] if len(dense_chunks) == 1 else np.vstack(dense_chunks)
        dense_chunks = [genes_by_cells.T.astype('float32', copy=False)]
        sparse_chunks = []
    else:
        genes_by_cells = sparse.vstack(sparse_chunks, format='csr', dtype='float32')
        sparse_chunks = [genes_by_cells.T.tocsc()]
        dense_chunks = None

    return sparse_chunks, dense_chunks, gene_names, sample_count, gene_count, nonzero_count


def read_expression_matrix(
    in_file,
    matrix_format='auto',
    sparse_density_threshold=0.35,
    csv_chunk_size=1000,
    input_orientation='samplesByGenes',
):
    matrix_format = str(matrix_format or 'auto').lower()
    csv_chunk_size = max(1, int(csv_chunk_size or 1000))
    input_orientation = str(input_orientation or 'samplesByGenes')

    try:
        if input_orientation == 'genesByCells':
            sparse_chunks, dense_chunks, gene_names, row_count, gene_count, nonzero_count = (
                read_genes_by_cells_matrix(in_file, matrix_format, csv_chunk_size)
            )
        else:
            sparse_chunks, dense_chunks, gene_names, row_count, gene_count, nonzero_count = (
                read_samples_by_genes_matrix(in_file, matrix_format, csv_chunk_size)
            )
    except ValueError as exc:
        raise ValueError(
            f"{in_file} contains a non-numeric or malformed expression matrix."
        ) from exc

    total_values = row_count * len(gene_names)
    density = nonzero_count / total_values if total_values else 0.0

    if matrix_format == 'dense':
        matrix = dense_chunks[0] if len(dense_chunks) == 1 else np.vstack(dense_chunks)
        print(
            f"Loaded dense expression matrix shape {matrix.shape}; "
            f"nonzero density {density:.4f}",
            flush=True,
        )
        return matrix.astype('float32', copy=False), gene_names, density, 'dense'

    matrix = sparse_chunks[0].tocsc() if len(sparse_chunks) == 1 else sparse.vstack(sparse_chunks, format='csc', dtype='float32')
    use_sparse = matrix_format == 'sparse' or density <= float(sparse_density_threshold)
    if use_sparse:
        print(
            f"Loaded sparse expression matrix shape {matrix.shape}; "
            f"nnz {matrix.nnz}; nonzero density {density:.4f}",
            flush=True,
        )
        return matrix, gene_names, density, 'sparse'

    dense_matrix = matrix.toarray().astype('float32', copy=False)
    print(
        f"Loaded dense expression matrix shape {dense_matrix.shape}; "
        f"nonzero density {density:.4f} exceeds sparse threshold {sparse_density_threshold}",
        flush=True,
    )
    return dense_matrix, gene_names, density, 'dense'


def run_inference(
    algo,
    in_file,
    out_file,
    client,
    genie3_trees,
    grnboost2_trees,
    max_regulators_per_target,
    matrix_format,
    sparse_density_threshold,
    csv_chunk_size,
    input_orientation,
):
    expression_matrix, gene_names, density, resolved_matrix_format = read_expression_matrix(
        in_file,
        matrix_format=matrix_format,
        sparse_density_threshold=sparse_density_threshold,
        csv_chunk_size=csv_chunk_size,
        input_orientation=input_orientation,
    )
    print(
        f"{algo} using {resolved_matrix_format} expression shape "
        f"{expression_matrix.shape} from {in_file} (density {density:.4f})",
        flush=True,
    )

    normalized_algo = str(algo).upper()
    if normalized_algo == 'GENIE3':
        rf_kwargs = dict(GENIE3_RF_KWARGS)
        rf_kwargs['n_estimators'] = genie3_trees
        print(f"GENIE3 using n_estimators={rf_kwargs['n_estimators']}", flush=True)
        network = diy(
            expression_data=expression_matrix,
            regressor_type='RF',
            regressor_kwargs=rf_kwargs,
            client_or_address=client,
            gene_names=gene_names,
        )
    elif normalized_algo == 'GRNBOOST2':
        gbm_kwargs = dict(SGBM_KWARGS)
        gbm_kwargs['n_estimators'] = grnboost2_trees
        print(f"GRNBoost2 using n_estimators={gbm_kwargs['n_estimators']}", flush=True)
        network = diy(
            expression_data=expression_matrix,
            regressor_type='GBM',
            regressor_kwargs=gbm_kwargs,
            client_or_address=client,
            gene_names=gene_names,
            early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
        )
    else:
        raise ValueError("Wrong algorithm name. Should either be GENIE3 or GRNBoost2.")

    write_network(network, out_file, max_regulators_per_target)

def main(args):
    opts, args = parseArgs(args)

    n_workers = opts.nWorkers or os.cpu_count() or 1
    client = Client(
        processes=False,
        n_workers=n_workers,
        threads_per_worker=opts.threadsPerWorker,
    )

    try:
        if opts.runRoot:
            if not opts.runIds or not opts.algorithmId:
                raise ValueError("--runRoot batch mode requires --runIds and --algorithmId.")

            run_root = Path(opts.runRoot)
            run_ids = [run_id.strip() for run_id in opts.runIds.split(',') if run_id.strip()]
            for run_id in run_ids:
                working_dir = run_root / run_id / opts.algorithmId / 'working_dir'
                run_inference(
                    opts.algo,
                    working_dir / 'ExpressionData.csv',
                    working_dir / 'outFile.txt',
                    client,
                    opts.genie3Trees,
                    opts.grnboost2Trees,
                    opts.maxRegulatorsPerTarget,
                    opts.matrixFormat,
                    opts.sparseDensityThreshold,
                    opts.csvChunkSize,
                    opts.inputOrientation,
                )
        else:
            if not opts.inFile or not opts.outFile:
                raise ValueError("Single-run mode requires --inFile and --outFile.")
            run_inference(
                opts.algo,
                opts.inFile,
                opts.outFile,
                client,
                opts.genie3Trees,
                opts.grnboost2Trees,
                opts.maxRegulatorsPerTarget,
                opts.matrixFormat,
                opts.sparseDensityThreshold,
                opts.csvChunkSize,
                opts.inputOrientation,
            )
    finally:
        client.close()
                        
if __name__ == "__main__":
    main(sys.argv)
