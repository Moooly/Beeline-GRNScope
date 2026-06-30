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


def adaptive_genie3_tree_count(gene_count):
    if gene_count is None or gene_count <= 0:
        return 100
    if gene_count <= 500:
        return 500
    if gene_count <= 2000:
        return 250
    if gene_count <= 8000:
        return 100
    return 50


def resolve_genie3_tree_count(expression_file):
    configured_tree_count = os.environ.get("GRNSCOPE_GENIE3_TREES")
    if configured_tree_count:
        try:
            return max(1, int(configured_tree_count))
        except ValueError:
            pass
    return adaptive_genie3_tree_count(count_expression_genes(expression_file))


def genie3_tree_args(expression_file):
    return [f"--genie3Trees={resolve_genie3_tree_count(expression_file)}"]


class GENIE3Runner(Runner):
    """Concrete runner for the GENIE3 GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for GENIE3.
        If the folder/files under self.input_dir exist,
        this function will not do anything.
        '''

        # Create ExpressionData.csv file in the created input directory
        GENIE3_EXPRESSION_FILE = self.working_dir / "ExpressionData.csv"
        if not GENIE3_EXPRESSION_FILE.exists():
            # input data
            ExpressionData = pd.read_csv(self.input_dir / self.exprData,
                                         header = 0, index_col = 0)

            # Write .csv file — arboreto expects cells as rows, genes as columns
            ExpressionData.T.to_csv(GENIE3_EXPRESSION_FILE,
                                 sep = '\t', header  = True, index = True)

    def run(self):
        '''
        Function to run GENIE3 algorithm
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
                            'python /runArboreto.py --algo=GENIE3',
                            '--inFile=/usr/working_dir/ExpressionData.csv',
                            '--outFile=/usr/working_dir/outFile.txt',
                            *arboreto_dask_args(),
                            *genie3_tree_args(self.working_dir / "ExpressionData.csv"),
                            *cap_arg, '\"'])

        self._run_docker(cmdToRun)

    @classmethod
    def run_batch(cls, runners):
        '''
        Run multiple GENIE3 confidence runs in one Docker/Dask session.
        '''

        if not runners:
            return

        first_runner = runners[0]
        output_root = first_runner.output_dir.parent.parent
        run_ids = ','.join(runner.output_dir.parent.name for runner in runners)
        max_edges_per_target = first_runner._resolve_max_edges_per_target()
        cap_arg = (
            [f"--maxRegulatorsPerTarget={max_edges_per_target}"]
            if max_edges_per_target is not None
            else []
        )

        cmdToRun = ' '.join(['docker run --rm',
                            f"-v {output_root}:/usr/arboreto_runs",
                            f"-v {ARBORETO_SCRIPT}:/runArboreto.py:ro",
                            '--expose=41269',
                            f'{first_runner.image} /bin/sh -c \"time -v -o',
                            "/usr/arboreto_runs/arboreto_batch_time.txt",
                            'python /runArboreto.py --algo=GENIE3',
                            '--runRoot=/usr/arboreto_runs',
                            f'--runIds={run_ids}',
                            '--algorithmId=GENIE3',
                            *arboreto_dask_args(),
                            *genie3_tree_args(first_runner.working_dir / "ExpressionData.csv"),
                            *cap_arg, '\"'])

        first_runner._run_docker(cmdToRun)

    def parseOutput(self):
        '''
        Function to parse outputs from GENIE3.
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
