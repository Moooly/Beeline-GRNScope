import csv
import os

import numpy as np
import pandas as pd

from BLRun.runner import Runner


class PearsonRunner(Runner):
    """Concrete runner for pairwise Pearson correlation GRN inference.
    Runs entirely within the BEELINE conda environment; no Docker image is used.
    The image field in the config should be set to 'local'."""

    EDGE_WRITE_CHUNK_SIZE = 200_000
    DEFAULT_TOP_K_PER_TARGET = 10

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
        # Read expression data: rows = genes, columns = cells
        ExpressionData = pd.read_csv(
            self.input_dir / self.exprData, header=0, index_col=0)
        if not isinstance(ExpressionData, pd.DataFrame):
            raise TypeError(f"ExpressionData must be a DataFrame, got {type(ExpressionData)}")

        corr_values = self._compute_corr_values(ExpressionData)
        genes = np.asarray(ExpressionData.index)
        top_k = self._resolve_top_k(len(genes))
        del ExpressionData

        self._write_ranked_edges_from_corr_values(
            corr_values,
            genes,
            self.output_dir / 'rankedEdges.csv',
            top_k=top_k,
        )

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
        raw_value = (
            self.params.get('topK')
            or self.params.get('pearsonTopK')
            or self.params.get('maxEdgesPerTarget')
            or os.environ.get('GRNSCOPE_PEARSON_TOP_K')
            or self.DEFAULT_TOP_K_PER_TARGET
        )
        if isinstance(raw_value, str) and raw_value.strip().lower() in {'0', 'all', 'none', 'false', 'off'}:
            return max(0, gene_count - 1)
        try:
            top_k = int(raw_value)
        except (TypeError, ValueError):
            top_k = self.DEFAULT_TOP_K_PER_TARGET
        if top_k <= 0:
            return max(0, gene_count - 1)
        return min(top_k, max(0, gene_count - 1))

    @staticmethod
    def _compute_corr_values(expression_data: pd.DataFrame) -> np.ndarray:
        '''
        Compute a gene x gene Pearson correlation matrix.

        For complete numeric data this uses numpy directly, which avoids
        pandas correlation overhead. If missing values are present, fall back
        to pandas so pairwise-complete correlation behavior is preserved.
        '''
        if not isinstance(expression_data, pd.DataFrame):
            raise TypeError(
                f"expression_data must be a DataFrame, got {type(expression_data)}"
            )

        values = np.asarray(expression_data.values, dtype=np.float64)
        if values.shape[0] < 2:
            return np.eye(values.shape[0], dtype=np.float64)

        if np.isnan(values).any():
            return np.asarray(
                expression_data.T.corr(method='pearson').values,
                dtype=np.float64,
            )

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
            top_k = cls.DEFAULT_TOP_K_PER_TARGET
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
