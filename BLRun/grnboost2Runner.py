import os
import pandas as pd
from pathlib import Path

from BLRun.runner import Runner

ARBORETO_SCRIPT = Path(__file__).resolve().parent.parent / "Algorithms" / "ARBORETO" / "runArboreto.py"


def count_expression_genes(expression_file):
    try:
        with open(expression_file, "r", encoding="utf-8") as handle:
            header = handle.readline().rstrip("\r\n").split("\t")
    except OSError:
        return None
    return max(0, len(header) - 1)


def arboreto_dask_args():
    worker_count = os.environ.get("GRNSCOPE_ARBORETO_DASK_WORKERS")
    if not worker_count:
        try:
            concurrent_tasks = max(1, int(os.environ.get("GRNSCOPE_MAX_CONCURRENT_ALGORITHMS", "2")))
        except ValueError:
            concurrent_tasks = 2
        worker_count = str(max(1, (os.cpu_count() or 1) // concurrent_tasks))

    threads_per_worker = os.environ.get("GRNSCOPE_ARBORETO_THREADS_PER_WORKER", "1")
    return [f"--nWorkers={worker_count}", f"--threadsPerWorker={threads_per_worker}"]


def adaptive_grnboost2_tree_count(gene_count):
    if gene_count is None or gene_count <= 0:
        return 1000
    if gene_count <= 500:
        return 5000
    if gene_count <= 2000:
        return 2000
    if gene_count <= 8000:
        return 1000
    return 500


def resolve_grnboost2_tree_count(expression_file):
    configured_tree_count = os.environ.get("GRNSCOPE_GRNBOOST2_TREES")
    if configured_tree_count:
        try:
            return max(1, int(configured_tree_count))
        except ValueError:
            pass
    return adaptive_grnboost2_tree_count(count_expression_genes(expression_file))


def grnboost2_tree_args(expression_file):
    return [f"--grnboost2Trees={resolve_grnboost2_tree_count(expression_file)}"]


class GRNBoost2Runner(Runner):
    """Concrete runner for the GRNBoost2 GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for GRNBoost2.
        If the folder/files under self.input_dir exist,
        this function will not do anything.
        '''

        # Create ExpressionData.csv file in the created input directory
        GRNBOOST2_EXPRESSION_FILE = self.working_dir / "ExpressionData.csv"
        if not GRNBOOST2_EXPRESSION_FILE.exists():
            ExpressionData = pd.read_csv(self.input_dir / self.exprData,
                                         header = 0, index_col = 0)

            # Write .csv file
            ExpressionData.T.to_csv(GRNBOOST2_EXPRESSION_FILE,
                                 sep = '\t', header  = True, index = True)

    def run(self):
        '''
        Function to run GRNBOOST2 algorithm
        '''

        max_edges_per_target = self._resolve_max_edges_per_target()
        cap_arg = (
            [f"--maxRegulatorsPerTarget={max_edges_per_target}"]
            if max_edges_per_target is not None
            else []
        )
        cmdToRun = ' '.join(['docker run --rm',
                            f"-v {self.working_dir}:/usr/working_dir",
                            f"-v {ARBORETO_SCRIPT}:/runArboreto.py:ro",
                            '--expose=41269',
                            f'{self.image} /bin/sh -c \"time -v -o',
                            "/usr/working_dir/time.txt",
                            'python /runArboreto.py --algo=GRNBoost2',
                            '--inFile=/usr/working_dir/ExpressionData.csv',
                            '--outFile=/usr/working_dir/outFile.txt',
                            *arboreto_dask_args(),
                            *grnboost2_tree_args(self.working_dir / "ExpressionData.csv"),
                            *cap_arg, '\"'])

        self._run_docker(cmdToRun)

    def parseOutput(self):
        '''
        Function to parse outputs from GRNBOOST2.
        '''
        workDir = self.working_dir
        outFile = workDir / 'outFile.txt'

        # Quit if output file does not exist
        if not outFile.exists():
            print(str(outFile) + ' does not exist, skipping...')
            return

        self._write_ranked_edges_from_edge_files(
            [outFile],
            sep='\t',
            source_col='TF',
            target_col='target',
            score_col='importance',
        )
