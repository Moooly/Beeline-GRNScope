import pandas as pd

from BLRun.runner import Runner


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
                            '--expose=41269',
                            f'{self.image} /bin/sh -c \"time -v -o',
                            "/usr/working_dir/time.txt",
                            'python runArboreto.py --algo=GRNBoost2',
                            '--inFile=/usr/working_dir/ExpressionData.csv',
                            '--outFile=/usr/working_dir/outFile.txt',
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
