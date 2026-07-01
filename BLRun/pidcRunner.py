import shlex
import shutil

from BLRun.runner import Runner


class PIDCRunner(Runner):
    """Concrete runner for the PIDC GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for PIDC.
        If the folder/files under self.input_dir exist,
        this function will not do anything.
        '''

        # Create ExpressionData.csv file in the created input directory
        PIDC_EXPRESSION_FILE = self.working_dir / "ExpressionData.csv"
        if not PIDC_EXPRESSION_FILE.exists():
            shutil.copy2(self.input_dir / self.exprData, PIDC_EXPRESSION_FILE)

    def run(self):
        '''
        Function to run PIDC algorithm
        '''

        work_mount = shlex.quote(f"{self.working_dir}:/usr/working_dir")
        matrix_format = str(self.params.get('matrixFormat') or self.params.get('sparseMatrix') or 'auto')
        if matrix_format.lower() in {'true', 'yes', 'on', '1'}:
            matrix_format = 'sparse'
        elif matrix_format.lower() in {'false', 'no', 'off', '0'}:
            matrix_format = 'dense'

        cmdToRun = ' '.join(['docker run --rm',
                            f"-v {work_mount}",
                            f'{self.image} /bin/sh -c \"time -v -o',
                            "/usr/working_dir/time.txt",
                            'julia runPIDC.jl',
                            "/usr/working_dir/ExpressionData.csv",
                            "/usr/working_dir/outFile.txt",
                            matrix_format, '\"'])

        self._run_docker(cmdToRun)

    def parseOutput(self):
        '''
        Function to parse outputs from PIDC.
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
            header=None,
            names=['Gene1', 'Gene2', 'EdgeWeight'],
        )
