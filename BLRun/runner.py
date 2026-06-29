from abc import ABC, abstractmethod
import csv
import math
import os
from pathlib import Path
import subprocess
from typing import Optional

import numpy as np
import pandas as pd

class Runner(ABC):
    """
    Abstract base_input class for BEELINE GRN inference algorithm runners.

    Subclasses must implement generateInputs, run, and parseOutput.
    Attributes set here reflect the fields accessed by runner implementations.
    """

    def __init__(self, root: Path, config: dict):
        """
        Parameters
        ----------
        root : Path
            Root path from which all subpaths are resolved.
        config : dict
            Merged configuration for a single dataset + algorithm run.
            Expected structure:
              input:
                input_dir:   <str>  input directory (absolute, or relative to root)
              dataset:
                dataset_id:          <str>  dataset group label (path segment under input_dir)
                run_id:              <str>  run label (path segment under dataset_id)
                exprData:            <str>  expression data filename
                pseudoTimeData:      <str>  pseudotime data filename
                groundTruthNetwork:  <str>  ground truth network filename
              output_settings:
                output_dir: <str>  output directory (absolute, or relative to root)
              algo_name: <str>  name of the algorithm (appended to output_dir)
              params: <dict>  algorithm-specific parameters
        """
        inp = config['input']
        ds  = config['dataset']

        input_dir_path  = Path(inp['input_dir'])
        output_dir_path = Path(config['output_settings']['output_dir'])
        # experiment_id : str — optional; when set, an experiment_id segment is
        # inserted between output_dir and the dataset path so multiple experiment
        # runs can coexist under the same base output directory.
        experiment_id_prefix = config['output_settings'].get('experiment_id', '')

        base_input = input_dir_path if input_dir_path.is_absolute() else root / input_dir_path

        base_output = output_dir_path if output_dir_path.is_absolute() else root / output_dir_path
        if experiment_id_prefix:
            base_output = base_output / experiment_id_prefix
        base_output = base_output / ds['dataset_id'] / ds['run_id'] / config['algo_name']

        base_input = base_input.resolve()
        base_output = base_output.resolve()

        # input_dir: run-level input directory (expression data, pseudo-time).
        self.input_dir  = base_input / ds['dataset_id'] / ds['run_id']
        self.output_dir = base_output
        self.working_dir = base_output / "working_dir"

        # Erase working directory so stale inputs from prior runs are not reused.
        if self.working_dir.exists():
            for item in sorted(self.working_dir.rglob('*'), reverse=True):
                item.unlink() if (item.is_file() or item.is_symlink()) else item.rmdir()
            self.working_dir.rmdir()

        # Pre-create output_dir and working_dir so docker cannot claim them as root.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Precompute progress message for CLI output.
        self.algo_name = config['algo_name']
        self.running_message = (
            f"Running {config['algo_name']} | dataset: {ds['dataset_id']} | run: {ds['run_id']}"
        )

        self.exprData           = ds.get('exprData',           'ExpressionData.csv')
        self.pseudoTimeData     = ds.get('pseudoTimeData',     'PseudoTime.csv')
        self.groundTruthNetwork = ds.get('groundTruthNetwork', 'GroundTruthNetwork.csv')
        # ground_truth_file: full path to the dataset-level ground truth CSV.
        self.ground_truth_file  = base_input / ds['dataset_id'] / self.groundTruthNetwork
        
        # image: Docker image name used to run this algorithm (e.g. "grnbeeline/genie3:base").
        # Mandatory — every algorithm entry in the config must supply this field.
        if 'image' not in config or not config['image']:
            raise ValueError("Algorithm config must include a non-empty 'image' field.")
        if not isinstance(config['image'], str):
            raise TypeError(f"'image' must be a str, got {type(config['image'])}")
        self.image = config['image']

        # Unwrap single-element lists so runners receive scalar values.
        # YAML config files commonly wrap param values in brackets
        # (e.g. `pVal: [0.01]`), which YAML parses as a list.
        raw_params = config.get('params', {})
        self.params = {
            k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
            for k, v in raw_params.items()
        }

    @abstractmethod
    def generateInputs(self):
        """Prepare algorithm-specific input files from the dataset."""

    @abstractmethod
    def run(self):
        """Execute the inference algorithm."""

    @abstractmethod
    def parseOutput(self) -> None:
        """
        Parse raw algorithm output and write a ranked edge list to disk.

        Implementations should build a DataFrame with columns Gene1, Gene2,
        EdgeWeight and pass it to self._write_ranked_edges(). Returns early
        without writing if the expected output file is missing.
        """

    def _run_docker(self, cmd: str, append: bool = False) -> None:
        """
        Execute a shell command and write combined stdout/stderr to output.txt.

        Parameters
        ----------
        cmd : str
            Shell command to execute (passed to the shell verbatim).
        append : bool
            If True, append to an existing output.txt. Use for runners that
            invoke docker in a loop so all container output is collected in
            one file. Defaults to False (overwrite).
        """
        if not isinstance(cmd, str):
            raise TypeError(f"cmd must be str, got {type(cmd)}")
        if not isinstance(append, bool):
            raise TypeError(f"append must be bool, got {type(append)}")

        mode = 'a' if append else 'w'
        with open(self.output_dir / 'output.txt', mode) as f:
            proc = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                f.write(line)
                f.flush()
            proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(
                f"Docker command failed (exit {proc.returncode}). "
                f"See {self.output_dir / 'output.txt'} for details."
            )

    @staticmethod
    def _adaptive_max_edges_per_target(gene_count: Optional[int]) -> int:
        if gene_count is None or gene_count <= 0:
            return 50
        if gene_count <= 500:
            return 100
        if gene_count <= 3000:
            return 50
        if gene_count <= 8000:
            return 25
        return 10

    def _resolve_max_edges_per_target(self, gene_count: Optional[int] = None) -> Optional[int]:
        normalized_algo_name = str(getattr(self, 'algo_name', '')).upper()
        raw_value = (
            self.params.get('maxRegulatorsPerTarget')
            or self.params.get('maxEdgesPerTarget')
            or self.params.get('topK')
            or os.environ.get(f'GRNSCOPE_{normalized_algo_name}_MAX_EDGES_PER_TARGET')
            or os.environ.get('GRNSCOPE_RANKED_EDGES_PER_TARGET_LIMIT')
            or os.environ.get('GRNSCOPE_MAX_REGULATORS_PER_TARGET')
        )

        if isinstance(raw_value, str) and raw_value.strip().lower() in {
            '0', 'all', 'none', 'false', 'off'
        }:
            return None

        if raw_value is None:
            return self._adaptive_max_edges_per_target(gene_count)

        try:
            parsed_value = int(raw_value)
        except (TypeError, ValueError):
            return self._adaptive_max_edges_per_target(gene_count)

        if parsed_value <= 0:
            return None
        if gene_count is not None and gene_count > 0:
            return min(parsed_value, gene_count)
        return parsed_value

    @staticmethod
    def _update_candidate(
        target_candidates: dict,
        source: str,
        target: str,
        score: float,
        max_edges_per_target: Optional[int],
    ) -> None:
        if not source or not target:
            return
        try:
            score = float(score)
        except (TypeError, ValueError):
            return
        if not math.isfinite(score):
            return

        candidates = target_candidates.setdefault(str(target), {})
        source = str(source)
        current_score = candidates.get(source)
        if current_score is not None:
            if abs(score) > abs(current_score):
                candidates[source] = score
            return

        if max_edges_per_target is None or len(candidates) < max_edges_per_target:
            candidates[source] = score
            return

        weakest_source, weakest_score = min(
            candidates.items(),
            key=lambda item: (abs(float(item[1])), str(item[0])),
        )
        if abs(score) > abs(float(weakest_score)):
            del candidates[weakest_source]
            candidates[source] = score

    @staticmethod
    def _candidate_rows(target_candidates: dict):
        rows = []
        for target, candidates in target_candidates.items():
            for source, score in candidates.items():
                rows.append((source, target, score))
        rows.sort(key=lambda row: (-abs(float(row[2])), str(row[0]), str(row[1])))
        return rows

    def _write_candidate_edges(self, target_candidates: dict) -> None:
        if not self.output_dir.is_dir():
            raise FileNotFoundError(
                f"Output directory does not exist: {self.output_dir}")

        with open(self.output_dir / 'rankedEdges.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['Gene1', 'Gene2', 'EdgeWeight'])
            writer.writerows(self._candidate_rows(target_candidates))

    def _read_gene_names(self, expression_path: Path):
        with open(expression_path, 'r', newline='') as f:
            sample = f.read(65536)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=',\t;')
            except csv.Error:
                first_line = sample.splitlines()[0] if sample.splitlines() else ''
                dialect = csv.excel_tab if '\t' in first_line else csv.excel
            reader = csv.reader(f, dialect=dialect)
            try:
                next(reader)
            except StopIteration:
                return []
            return [row[0] for row in reader if row and str(row[0]).strip()]

    def _write_ranked_edges_from_matrix(
        self,
        matrix,
        genes,
        *,
        absolute_scores: bool = False,
        transform=None,
        skip_self: bool = False,
    ) -> None:
        target_candidates: dict = {}
        self._update_candidates_from_matrix(
            target_candidates,
            matrix,
            genes,
            absolute_scores=absolute_scores,
            transform=transform,
            skip_self=skip_self,
        )
        self._write_candidate_edges(target_candidates)

    def _update_candidates_from_matrix(
        self,
        target_candidates: dict,
        matrix,
        genes,
        *,
        absolute_scores: bool = False,
        transform=None,
        skip_self: bool = False,
    ) -> None:
        values = np.asarray(matrix)
        genes = list(genes)
        if values.ndim != 2:
            raise ValueError(f"Expected a 2D output matrix, got shape {values.shape}.")
        if values.shape[0] != len(genes) or values.shape[1] != len(genes):
            raise ValueError(
                f"Output matrix shape {values.shape} does not match {len(genes)} genes."
            )

        max_edges_per_target = self._resolve_max_edges_per_target(len(genes))
        if max_edges_per_target is not None:
            max_edges_per_target = max(1, min(max_edges_per_target, len(genes)))

        for target_index, target_gene in enumerate(genes):
            scores = values[:, target_index].astype(float, copy=True)
            if transform is not None:
                scores = transform(scores)
            if absolute_scores:
                scores = np.abs(scores)
            scores[~np.isfinite(scores)] = -np.inf
            if skip_self and target_index < len(scores):
                scores[target_index] = -np.inf

            finite_count = int(np.isfinite(scores).sum())
            if finite_count == 0:
                continue
            take_count = finite_count if max_edges_per_target is None else min(max_edges_per_target, finite_count)
            if take_count < len(scores):
                candidate_indices = np.argpartition(scores, -take_count)[-take_count:]
            else:
                candidate_indices = np.arange(len(scores))
            candidate_indices = candidate_indices[
                np.argsort(scores[candidate_indices])[::-1]
            ]

            for source_index in candidate_indices:
                self._update_candidate(
                    target_candidates,
                    str(genes[source_index]).strip(),
                    str(target_gene).strip(),
                    float(scores[source_index]),
                    max_edges_per_target,
                )

    def _write_ranked_edges_from_edge_files(
        self,
        edge_files,
        *,
        sep: str,
        header=0,
        names=None,
        source_col='Gene1',
        target_col='Gene2',
        score_col='EdgeWeight',
        score_abs: bool = False,
        swap_source_target: bool = False,
        chunksize: int = 200000,
    ) -> None:
        target_candidates: dict = {}
        max_edges_per_target = self._resolve_max_edges_per_target()

        for edge_file in edge_files:
            for chunk in pd.read_csv(
                edge_file,
                sep=sep,
                header=header,
                names=names,
                chunksize=chunksize,
            ):
                scores = pd.to_numeric(chunk[score_col], errors='coerce')
                for source, target, score in zip(
                    chunk[source_col],
                    chunk[target_col],
                    scores,
                ):
                    if pd.isna(score):
                        continue
                    if score_abs:
                        score = abs(float(score))
                    if swap_source_target:
                        source, target = target, source
                    self._update_candidate(
                        target_candidates,
                        str(source).strip(),
                        str(target).strip(),
                        float(score),
                        max_edges_per_target,
                    )

        self._write_candidate_edges(target_candidates)

    def _cap_ranked_edges_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if not {'Gene1', 'Gene2', 'EdgeWeight'}.issubset(df.columns):
            missing = {'Gene1', 'Gene2', 'EdgeWeight'} - set(df.columns)
            raise ValueError(f"ranked edge DataFrame is missing columns: {sorted(missing)}")

        edges = df[['Gene1', 'Gene2', 'EdgeWeight']].copy()
        edges['Gene1'] = edges['Gene1'].astype(str).str.strip()
        edges['Gene2'] = edges['Gene2'].astype(str).str.strip()
        edges['EdgeWeight'] = pd.to_numeric(edges['EdgeWeight'], errors='coerce')
        edges = edges[
            (edges['Gene1'] != '')
            & (edges['Gene2'] != '')
            & edges['EdgeWeight'].notna()
        ]

        max_edges_per_target = self._resolve_max_edges_per_target()
        if max_edges_per_target is None:
            return edges

        edges['_abs_score'] = edges['EdgeWeight'].abs()
        edges.sort_values(
            ['Gene2', '_abs_score', 'Gene1'],
            ascending=[True, False, True],
            inplace=True,
        )
        capped = edges.groupby('Gene2', sort=False).head(max_edges_per_target)
        return capped[['Gene1', 'Gene2', 'EdgeWeight']]

    def _write_ranked_edges(self, df: pd.DataFrame) -> None:
        """
        Write a ranked edge list to self.output_dir/rankedEdges.csv.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns Gene1, Gene2, EdgeWeight.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If self.output_dir does not exist at the time of writing.
        TypeError
            If df is not a pd.DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be pd.DataFrame, got {type(df)}")
        if not self.output_dir.is_dir():
            raise FileNotFoundError(
                f"Output directory does not exist: {self.output_dir}")
        self._cap_ranked_edges_dataframe(df).to_csv(
            self.output_dir / 'rankedEdges.csv',
            sep='\t',
            index=False,
        )
