import numpy as np
import pandas as pd

from BLRun.runner import Runner


class PearsonRunner(Runner):
    """Concrete runner for pairwise Pearson correlation GRN inference.
    Runs entirely within the BEELINE conda environment; no Docker image is used.
    The image field in the config should be set to 'local'."""

    EDGE_WRITE_CHUNK_SIZE = 200_000

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

        self._write_ranked_edges_from_corr_values(
            corr_values,
            np.asarray(ExpressionData.index),
            self.output_dir / 'rankedEdges.csv',
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
        )

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
    ) -> None:
        '''
        Write a symmetric correlation matrix as a directed ranked edge list.

        The upper triangle is sorted once, then each undirected pair is
        mirrored into both directed orientations. Output is written in chunks
        to avoid materializing all directed edge rows in memory at once.
        '''
        corr_values = np.asarray(corr_values)
        genes = np.asarray(genes)

        if corr_values.ndim != 2 or corr_values.shape[0] != corr_values.shape[1]:
            raise ValueError(f"corr_values must be square, got shape {corr_values.shape}")
        if len(genes) != corr_values.shape[0]:
            raise ValueError(
                f"genes length {len(genes)} does not match corr shape {corr_values.shape}"
            )

        upper_i, upper_j = np.triu_indices(len(genes), k=1)
        weights = corr_values[upper_i, upper_j]
        sort_values = np.abs(weights).copy()
        sort_values[np.isnan(sort_values)] = -np.inf
        order = np.argsort(sort_values)[::-1]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            f.write('Gene1\tGene2\tEdgeWeight\n')

        for start in range(0, len(order), cls.EDGE_WRITE_CHUNK_SIZE):
            chunk_order = order[start:start + cls.EDGE_WRITE_CHUNK_SIZE]
            source = genes[upper_i[chunk_order]]
            target = genes[upper_j[chunk_order]]
            chunk_weights = weights[chunk_order]
            chunk_df = cls._build_directed_edge_chunk(source, target, chunk_weights)
            chunk_df.to_csv(out_path, sep='\t', index=False, header=False, mode='a')

    @staticmethod
    def _build_directed_edge_chunk(
        source: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray,
    ) -> pd.DataFrame:
        '''
        Build a directed edge chunk by mirroring each source-target pair.
        '''
        directed_count = len(weights) * 2
        gene1 = np.empty(directed_count, dtype=object)
        gene2 = np.empty(directed_count, dtype=object)
        edge_weight = np.empty(directed_count, dtype=weights.dtype)

        gene1[0::2] = source
        gene2[0::2] = target
        edge_weight[0::2] = weights

        gene1[1::2] = target
        gene2[1::2] = source
        edge_weight[1::2] = weights

        return pd.DataFrame({
            'Gene1': gene1,
            'Gene2': gene2,
            'EdgeWeight': edge_weight,
        })
