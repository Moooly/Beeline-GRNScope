import csv
import os

import numpy as np
import pandas as pd
try:
    from scipy import sparse
except ImportError:  # pragma: no cover - sparse mode requires scipy at runtime.
    sparse = None

from BLRun.runner import Runner


class PearsonRunner(Runner):
    """Concrete runner for pairwise Pearson correlation GRN inference.
    Runs entirely within the BEELINE conda environment; no Docker image is used.
    The image field in the config should be set to 'local'."""

    EDGE_WRITE_CHUNK_SIZE = 200_000
    DEFAULT_CORRELATION_BLOCK_SIZE = 256
    DEFAULT_SPARSE_DENSITY_THRESHOLD = 0.35

    def generateInputs(self):
        '''
        Verifies that the expression data file exists in the input directory.
        No file copying is required because Pearson runs locally without Docker.

        :param self.input_dir: Path — directory containing input files
        :param self.exprData: str — expression data filename
        :raises FileNotFoundError: if the expression data file is missing
        '''
        if not (self.input_dir / self.exprData).exists():
            raise FileNotFoundError(
                f"Expression data file not found: {self.input_dir / self.exprData}")

    def run(self):
        '''
        Computes pairwise Pearson correlation between all gene pairs.
        Writes the ranked edge list directly to output_dir/rankedEdges.csv.

        :param self.input_dir: Path — directory containing expression data
        :param self.exprData: str — CSV filename; rows = genes, columns = cells
        :output output_dir/rankedEdges.csv: tab-separated ranked edge list
        '''
        expression_path = self.input_dir / self.exprData
        matrix_format = self._resolve_matrix_format()
        if matrix_format == 'dense':
            values, genes = self._read_dense_expression_values(expression_path)
            self._write_ranked_edges_from_expression_values(
                values,
                genes,
                self.output_dir / 'rankedEdges.csv',
            )
            return

        values, genes, density = self._read_sparse_expression_values(expression_path)
        use_sparse = (
            matrix_format == 'sparse'
            or density <= self._resolve_sparse_density_threshold()
        )
        if use_sparse:
            self._write_ranked_edges_from_sparse_expression_values(
                values,
                genes,
                self.output_dir / 'rankedEdges.csv',
            )
            return

        values = values.toarray().astype(np.float32, copy=False)

        self._write_ranked_edges_from_expression_values(
            values,
            genes,
            self.output_dir / 'rankedEdges.csv',
        )

    def _resolve_matrix_format(self) -> str:
        raw_value = (
            self.params.get('matrixFormat')
            or self.params.get('sparseMatrix')
            or os.environ.get('GRNSCOPE_PEARSON_MATRIX_FORMAT')
            or os.environ.get('GRNSCOPE_SPARSE_MATRIX_FORMAT')
            or 'auto'
        )
        if isinstance(raw_value, bool):
            return 'sparse' if raw_value else 'dense'
        value = str(raw_value).strip().lower()
        if value in {'true', 'yes', 'on', '1'}:
            return 'sparse'
        if value in {'false', 'no', 'off', '0'}:
            return 'dense'
        if value in {'auto', 'sparse', 'dense'}:
            return value
        return 'auto'

    def _resolve_sparse_density_threshold(self) -> float:
        raw_value = (
            self.params.get('sparseDensityThreshold')
            or os.environ.get('GRNSCOPE_PEARSON_SPARSE_DENSITY_THRESHOLD')
            or os.environ.get('GRNSCOPE_SPARSE_DENSITY_THRESHOLD')
            or self.DEFAULT_SPARSE_DENSITY_THRESHOLD
        )
        try:
            return max(0.0, min(1.0, float(raw_value)))
        except (TypeError, ValueError):
            return self.DEFAULT_SPARSE_DENSITY_THRESHOLD

    def _resolve_csv_chunk_size(self) -> int:
        raw_value = (
            self.params.get('csvChunkSize')
            or os.environ.get('GRNSCOPE_PEARSON_CSV_CHUNK_SIZE')
            or os.environ.get('GRNSCOPE_CSV_CHUNK_SIZE')
            or 1000
        )
        try:
            return max(1, int(raw_value))
        except (TypeError, ValueError):
            return 1000

    def _read_dense_expression_values(self, expression_path):
        ExpressionData = pd.read_csv(expression_path, header=0, index_col=0)
        if not isinstance(ExpressionData, pd.DataFrame):
            raise TypeError(f"ExpressionData must be a DataFrame, got {type(ExpressionData)}")
        genes = np.asarray(ExpressionData.index)
        values = np.asarray(ExpressionData.values, dtype=np.float32)
        return values, genes

    def _read_sparse_expression_values(self, expression_path):
        if sparse is None:
            raise ImportError(
                "PEARSON sparse mode requires scipy. Recreate the BEELINE env "
                "from utils/environment.yml or set matrixFormat: ['dense']."
            )

        chunks = pd.read_csv(
            expression_path,
            header=0,
            index_col=0,
            chunksize=self._resolve_csv_chunk_size(),
        )
        genes = []
        sparse_chunks = []
        nonzero_count = 0
        cell_count = None

        for chunk in chunks:
            genes.extend(chunk.index.astype(str).tolist())
            values = np.asarray(chunk.values, dtype=np.float32)
            if cell_count is None:
                cell_count = values.shape[1]
            elif cell_count != values.shape[1]:
                raise ValueError(f"{expression_path} has inconsistent cell counts across chunks.")
            sparse_chunk = sparse.csr_matrix(values)
            sparse_chunks.append(sparse_chunk)
            nonzero_count += int(sparse_chunk.nnz)

        if not sparse_chunks:
            raise ValueError(f"Expression data file is empty: {expression_path}")

        matrix = (
            sparse_chunks[0].tocsr()
            if len(sparse_chunks) == 1
            else sparse.vstack(sparse_chunks, format='csr', dtype=np.float32)
        )
        total_values = matrix.shape[0] * matrix.shape[1]
        density = nonzero_count / total_values if total_values else 0.0
        return matrix, np.asarray(genes), density

    def parseOutput(self):
        '''
        Legacy parser for an existing working_dir/outFile.txt correlation matrix.
        Normal Pearson runs write output_dir/rankedEdges.csv directly in run().

        :param self.working_dir: Path — directory containing outFile.txt
        :output output_dir/rankedEdges.csv: tab-separated edge list with columns
            Gene1 (str), Gene2 (str), EdgeWeight (float, signed Pearson r)
        '''
        ranked_edges = self.output_dir / 'rankedEdges.csv'
        if ranked_edges.exists():
            return

        outFile = self.working_dir / 'outFile.txt'
        if not outFile.exists():
            print(str(outFile) + ' does not exist, skipping...')
            return

        # Read square correlation matrix (genes x genes)
        CorrDF = pd.read_csv(outFile, sep='\t', header=0, index_col=0)
        if not isinstance(CorrDF, pd.DataFrame):
            raise TypeError(f"CorrDF must be a DataFrame, got {type(CorrDF)}")

        self._write_ranked_edges_from_corr_values(
            np.asarray(CorrDF.values, dtype=np.float64),
            np.asarray(CorrDF.index),
            self.output_dir / 'rankedEdges.csv',
            top_k=self._resolve_top_k(len(CorrDF.index)),
        )

    def _resolve_top_k(self, gene_count: int) -> int:
        '''
        Resolve how many source genes to keep for each target gene.

        The default keeps PEARSON practical for large matrices. Set topK,
        pearsonTopK, maxEdgesPerTarget, or GRNSCOPE_PEARSON_TOP_K to 0/all
        only when you really want the full all-vs-all edge table.
        '''
        raw_value = self.params.get('pearsonTopK') or os.environ.get('GRNSCOPE_PEARSON_TOP_K')
        if isinstance(raw_value, str) and raw_value.strip().lower() in {'0', 'all', 'none', 'false', 'off'}:
            return max(0, gene_count - 1)
        if raw_value is None:
            top_k = self._resolve_max_edges_per_target(gene_count)
            return max(0, gene_count - 1) if top_k is None else min(top_k, max(0, gene_count - 1))
        try:
            top_k = int(raw_value)
        except (TypeError, ValueError):
            top_k = self._resolve_max_edges_per_target(gene_count)
            return max(0, gene_count - 1) if top_k is None else min(top_k, max(0, gene_count - 1))
        if top_k <= 0:
            return max(0, gene_count - 1)
        return min(top_k, max(0, gene_count - 1))

    @staticmethod
    def _standardize_expression_rows(values: np.ndarray) -> np.ndarray:
        '''
        Center and L2-normalize each gene vector so dot products are Pearson r.
        '''
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"Expression values must be 2D, got shape {values.shape}")
        if values.shape[1] == 0:
            return np.zeros(values.shape, dtype=np.float32)

        values = values.copy()
        values -= values.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(values, axis=1)
        valid = norms > 0
        values[valid] /= norms[valid, None]
        values[~valid] = 0.0
        return values

    def _resolve_corr_block_size(self) -> int:
        try:
            return max(
                1,
                int(os.environ.get(
                    'GRNSCOPE_PEARSON_CORRELATION_BLOCK_SIZE',
                    str(self.DEFAULT_CORRELATION_BLOCK_SIZE),
                )),
            )
        except (TypeError, ValueError):
            return self.DEFAULT_CORRELATION_BLOCK_SIZE

    def _write_ranked_edges_from_expression_values(
        self,
        expression_values: np.ndarray,
        genes: np.ndarray,
        out_path,
    ) -> None:
        genes = np.asarray(genes)
        gene_count = len(genes)
        top_k = self._resolve_top_k(gene_count)

        if gene_count < 2:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['Gene1', 'Gene2', 'EdgeWeight'])
            return

        standardized_values = self._standardize_expression_rows(expression_values)
        block_size = self._resolve_corr_block_size()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['Gene1', 'Gene2', 'EdgeWeight'])
            rows = []

            for block_start in range(0, gene_count, block_size):
                block_end = min(block_start + block_size, gene_count)
                corr_block = standardized_values @ standardized_values[block_start:block_end].T

                for block_offset, target_index in enumerate(range(block_start, block_end)):
                    target_scores = corr_block[:, block_offset]
                    abs_scores = np.abs(target_scores).astype(np.float32, copy=True)
                    abs_scores[target_index] = -np.inf
                    abs_scores[~np.isfinite(abs_scores)] = -np.inf

                    candidate_count = min(top_k, gene_count - 1)
                    if candidate_count <= 0:
                        continue
                    if candidate_count < gene_count - 1:
                        candidate_indices = np.argpartition(abs_scores, -candidate_count)[-candidate_count:]
                    else:
                        candidate_indices = np.arange(gene_count)
                        candidate_indices = candidate_indices[candidate_indices != target_index]
                    candidate_indices = candidate_indices[
                        np.argsort(abs_scores[candidate_indices])[::-1]
                    ]

                    for source_index in candidate_indices:
                        score = target_scores[source_index]
                        if not np.isfinite(score):
                            continue
                        rows.append((genes[source_index], genes[target_index], float(score)))
                        if len(rows) >= self.EDGE_WRITE_CHUNK_SIZE:
                            writer.writerows(rows)
                            rows.clear()

            if rows:
                writer.writerows(rows)

    def _write_ranked_edges_from_sparse_expression_values(
        self,
        expression_values,
        genes: np.ndarray,
        out_path,
    ) -> None:
        if sparse is None:
            raise ImportError("Sparse Pearson calculation requires scipy.")

        expression_values = expression_values.tocsr().astype(np.float32, copy=False)
        genes = np.asarray(genes)
        gene_count, cell_count = expression_values.shape
        top_k = self._resolve_top_k(gene_count)

        if gene_count < 2:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['Gene1', 'Gene2', 'EdgeWeight'])
            return

        if cell_count == 0:
            raise ValueError("Expression matrix must contain at least one cell.")

        row_sums = np.asarray(expression_values.sum(axis=1)).ravel().astype(np.float64)
        row_sq_sums = np.asarray(expression_values.multiply(expression_values).sum(axis=1)).ravel().astype(np.float64)
        norms = row_sq_sums - (row_sums * row_sums / float(cell_count))
        norms[norms < 0] = 0
        norms = np.sqrt(norms)

        block_size = self._resolve_corr_block_size()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['Gene1', 'Gene2', 'EdgeWeight'])
            rows = []

            for block_start in range(0, gene_count, block_size):
                block_end = min(block_start + block_size, gene_count)
                target_block = expression_values[block_start:block_end, :]
                dot_block = (expression_values @ target_block.T).toarray().astype(np.float64, copy=False)
                centered_dot_block = dot_block - (
                    row_sums[:, None] * row_sums[block_start:block_end][None, :] / float(cell_count)
                )
                denom = norms[:, None] * norms[block_start:block_end][None, :]
                with np.errstate(invalid='ignore', divide='ignore'):
                    corr_block = centered_dot_block / denom

                for block_offset, target_index in enumerate(range(block_start, block_end)):
                    target_scores = corr_block[:, block_offset]
                    abs_scores = np.abs(target_scores).astype(np.float64, copy=True)
                    abs_scores[target_index] = -np.inf
                    abs_scores[~np.isfinite(abs_scores)] = -np.inf

                    candidate_count = min(top_k, gene_count - 1)
                    if candidate_count <= 0:
                        continue
                    if candidate_count < gene_count - 1:
                        candidate_indices = np.argpartition(abs_scores, -candidate_count)[-candidate_count:]
                    else:
                        candidate_indices = np.arange(gene_count)
                        candidate_indices = candidate_indices[candidate_indices != target_index]
                    candidate_indices = candidate_indices[
                        np.argsort(abs_scores[candidate_indices])[::-1]
                    ]

                    for source_index in candidate_indices:
                        score = target_scores[source_index]
                        if not np.isfinite(score):
                            continue
                        rows.append((genes[source_index], genes[target_index], float(score)))
                        if len(rows) >= self.EDGE_WRITE_CHUNK_SIZE:
                            writer.writerows(rows)
                            rows.clear()

            if rows:
                writer.writerows(rows)

    @staticmethod
    def _compute_corr_values(expression_data: pd.DataFrame) -> np.ndarray:
        '''
        Legacy full-matrix Pearson calculation used only by parseOutput().
        '''
        if not isinstance(expression_data, pd.DataFrame):
            raise TypeError(
                f"expression_data must be a DataFrame, got {type(expression_data)}"
            )
        values = np.asarray(expression_data.values, dtype=np.float64)
        if values.shape[0] < 2:
            return np.eye(values.shape[0], dtype=np.float64)
        with np.errstate(invalid='ignore', divide='ignore'):
            corr_values = np.corrcoef(values)
        if corr_values.ndim == 0:
            return np.eye(values.shape[0], dtype=np.float64)
        return corr_values

    @classmethod
    def _write_ranked_edges_from_corr_values(
        cls,
        corr_values: np.ndarray,
        genes: np.ndarray,
        out_path,
        top_k=None,
    ) -> None:
        '''
        Write a correlation matrix as a directed ranked edge list.

        For each target gene, only the strongest top_k source genes are kept.
        This prevents large datasets from producing all-vs-all edge tables
        with millions of rows that overwhelm GRNScope during aggregation.
        '''
        corr_values = np.asarray(corr_values)
        genes = np.asarray(genes)

        if corr_values.ndim != 2 or corr_values.shape[0] != corr_values.shape[1]:
            raise ValueError(f"corr_values must be square, got shape {corr_values.shape}")
        if len(genes) != corr_values.shape[0]:
            raise ValueError(
                f"genes length {len(genes)} does not match corr shape {corr_values.shape}"
            )

        gene_count = len(genes)
        if gene_count < 2:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['Gene1', 'Gene2', 'EdgeWeight'])
            return

        if top_k is None:
            top_k = cls._adaptive_max_edges_per_target(gene_count)
        top_k = min(max(1, int(top_k)), gene_count - 1)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['Gene1', 'Gene2', 'EdgeWeight'])
            rows = []

            for target_index, target_gene in enumerate(genes):
                target_scores = np.asarray(corr_values[:, target_index])
                abs_scores = np.abs(target_scores).astype(np.float64, copy=True)
                abs_scores[target_index] = -np.inf
                abs_scores[~np.isfinite(abs_scores)] = -np.inf

                if top_k < gene_count - 1:
                    candidate_indices = np.argpartition(abs_scores, -top_k)[-top_k:]
                else:
                    candidate_indices = np.arange(gene_count)
                    candidate_indices = candidate_indices[candidate_indices != target_index]

                candidate_indices = candidate_indices[
                    np.argsort(abs_scores[candidate_indices])[::-1]
                ]

                for source_index in candidate_indices:
                    score = target_scores[source_index]
                    if not np.isfinite(score):
                        continue
                    rows.append((genes[source_index], target_gene, float(score)))
                    if len(rows) >= cls.EDGE_WRITE_CHUNK_SIZE:
                        writer.writerows(rows)
                        rows.clear()

            if rows:
                writer.writerows(rows)
