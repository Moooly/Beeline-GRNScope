import pandas as pd
import shlex

from BLRun.runner import Runner
from BLRun.sparse_utils import read_expression_sparse, write_cell_by_gene_matrix


class JUMP3Runner(Runner):
    """Concrete runner for the JUMP3 GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for JUMP3.
        If the folder/files under self.input_dir exist,
        this function will not do anything.
        '''

        # Create ExpressionData.csv file in the created input directory
        JUMP3_EXPRESSION_FILE = self.working_dir / "ExpressionData.csv"
        if not JUMP3_EXPRESSION_FILE.exists():
            ExpressionData, genes, cells, density = read_expression_sparse(
                self.input_dir / self.exprData,
                chunksize=self.params.get('csvChunkSize', 1000),
            )
            PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                                 header = 0, index_col = 0)
            PTData.index = PTData.index.map(str)
            # Acc. to JUMP3:
            # In input argument Time, the first time point of each time series must be 0.
            # Also has to be an integer!
            aligned_pt = PTData.reindex(cells)
            missing_pt = aligned_pt.index[aligned_pt['PseudoTime'].isnull()].tolist()
            if missing_pt:
                preview = ', '.join(missing_pt[:5])
                raise KeyError(f"JUMP3 pseudotime is missing {len(missing_pt)} expression cells: {preview}")
            time_values = (aligned_pt['PseudoTime'] - PTData['PseudoTime'].min()).tolist()
            if 'Experiment' in PTData:
                experiment_values = aligned_pt['Experiment'].tolist()
            else:
                # generate it from cell number Ex_y, where x is experiment number
                #newExpressionData['Experiment'] = [int(x.split('_')[0].strip('E')) for x in PTData.index.astype(str)]
                experiment_values = [1] * len(cells)

            write_cell_by_gene_matrix(
                ExpressionData,
                genes,
                list(range(len(cells))),
                JUMP3_EXPRESSION_FILE,
                delimiter=',',
                include_header=True,
                append_columns=[
                    ('Time', time_values),
                    ('Experiment', experiment_values),
                ],
            )

    def run(self):
        '''
        Function to run JUMP3 algorithm
        '''

        work_mount = shlex.quote(f"{self.working_dir}:/usr/working_dir")
        cmdToRun = ' '.join(['docker run --rm',
                            f"-v {work_mount}",
                            f'{self.image} /bin/sh -c \"time -v -o',
                            "/usr/working_dir/time.txt",
                            './runJump3',
                            "/usr/working_dir/ExpressionData.csv", "/usr/working_dir/outFile.txt", '\"'])

        self._run_docker(cmdToRun)

    def parseOutput(self):
        '''
        Function to parse outputs from JUMP3.
        '''
        workDir = self.working_dir
        outFile = workDir / 'outFile.txt'

        # Quit if output file does not exist
        if not outFile.exists():
            print(str(outFile) + ' does not exist, skipping...')
            return

        # Read output
        OutDF = pd.read_csv(outFile, sep = ',')

        gene_list = self._read_gene_names(self.input_dir / self.exprData)
        self._write_ranked_edges_from_matrix(
            OutDF.values,
            gene_list,
            absolute_scores=True,
        )
