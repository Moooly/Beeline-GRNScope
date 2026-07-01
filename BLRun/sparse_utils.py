import csv
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import sparse
except ImportError:
    sparse = None


def read_expression_sparse(expression_path, chunksize=1000):
    if sparse is None:
        raise ImportError(
            "SciPy is required for sparse expression staging. "
            "Update the BEELINE conda environment from utils/environment.yml."
        )

    expression_path = Path(expression_path)
    chunks = pd.read_csv(
        expression_path,
        sep=None,
        engine='python',
        header=0,
        index_col=0,
        chunksize=max(1, int(chunksize)),
    )
    genes = []
    sparse_chunks = []
    columns = None
    nonzero_count = 0

    for chunk in chunks:
        if columns is None:
            columns = chunk.columns.astype(str).tolist()
        elif columns != chunk.columns.astype(str).tolist():
            raise ValueError(f"{expression_path} has inconsistent columns across chunks.")

        genes.extend(chunk.index.astype(str).tolist())
        values = np.asarray(chunk.values, dtype=np.float32)
        sparse_chunk = sparse.csr_matrix(values)
        sparse_chunks.append(sparse_chunk)
        nonzero_count += int(sparse_chunk.nnz)

    if columns is None or not sparse_chunks:
        raise ValueError(f"Expression matrix is empty: {expression_path}")

    matrix = (
        sparse_chunks[0].tocsr()
        if len(sparse_chunks) == 1
        else sparse.vstack(sparse_chunks, format='csr', dtype=np.float32)
    )
    total_values = matrix.shape[0] * matrix.shape[1]
    density = nonzero_count / total_values if total_values else 0.0
    return matrix, genes, columns, density


def expression_metadata(expression_path):
    with open(expression_path, 'r', newline='') as handle:
        sample = handle.read(65536)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',\t;')
        except csv.Error:
            first_line = sample.splitlines()[0] if sample.splitlines() else ''
            dialect = csv.excel_tab if '\t' in first_line else csv.excel
        reader = csv.reader(handle, dialect)
        try:
            header = next(reader)
        except StopIteration:
            return [], []
        cells = [str(value) for value in header[1:]]
        genes = [str(row[0]) for row in reader if row and str(row[0]).strip()]
        return genes, cells


def cell_positions(all_cells, selected_cells):
    position_by_cell = {str(cell): index for index, cell in enumerate(all_cells)}
    missing = [str(cell) for cell in selected_cells if str(cell) not in position_by_cell]
    if missing:
        preview = ', '.join(missing[:5])
        raise KeyError(f"Expression matrix is missing {len(missing)} pseudotime cells: {preview}")
    return [position_by_cell[str(cell)] for cell in selected_cells]


def write_gene_by_cell_matrix(
    expression_matrix,
    genes,
    cell_indices,
    out_path,
    *,
    delimiter=',',
    include_header=True,
    include_gene_column=True,
    header_gene_label='GENES',
    cell_names=None,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submatrix = expression_matrix[:, cell_indices].tocsr()
    cell_names = [] if cell_names is None else [str(cell) for cell in cell_names]

    with open(out_path, 'w', newline='') as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        if include_header:
            if include_gene_column:
                writer.writerow([header_gene_label, *cell_names])
            else:
                writer.writerow(cell_names)

        for gene, row_index in zip(genes, range(submatrix.shape[0])):
            values = submatrix.getrow(row_index).toarray().ravel()
            if include_gene_column:
                writer.writerow([gene, *values])
            else:
                writer.writerow(values)


def write_cell_by_gene_matrix(
    expression_matrix,
    genes,
    cell_indices,
    out_path,
    *,
    delimiter=',',
    include_header=True,
    append_columns=None,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cell_by_gene = expression_matrix[:, cell_indices].T.tocsr()
    append_columns = append_columns or []
    for name, values in append_columns:
        if len(values) != cell_by_gene.shape[0]:
            raise ValueError(
                f"Column {name} has {len(values)} values, expected {cell_by_gene.shape[0]}."
            )

    with open(out_path, 'w', newline='') as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        if include_header:
            writer.writerow([*genes, *[name for name, _ in append_columns]])

        append_values = [list(values) for _, values in append_columns]
        for row_index in range(cell_by_gene.shape[0]):
            values = cell_by_gene.getrow(row_index).toarray().ravel()
            writer.writerow([*values, *[column[row_index] for column in append_values]])
