import shlex
import shutil

from BLRun.runner import Runner


class SCSGLRunner(Runner):
    """Concrete runner for the scSGL GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for scSGL.
        If the folder/files under self.input_dir exist,
        this function will not do anything.
        '''

        # Create ExpressionData.csv file in the created input directory
        SCSGL_EXPRESSION_FILE = self.working_dir / "ExpressionData.csv"
        if not SCSGL_EXPRESSION_FILE.exists():
            shutil.copy2(self.input_dir / self.exprData, SCSGL_EXPRESSION_FILE)

        SCSGL_GROUND_TRUTH_FILE = self.working_dir / "GroundTruthNetwork.csv"
        if not SCSGL_GROUND_TRUTH_FILE.exists():
            shutil.copy2(self.ground_truth_file, SCSGL_GROUND_TRUTH_FILE)

    def run(self):
        '''
        Function to run SCSGL algorithm
        '''

        pos_density = str(self.params['pos_density'])
        neg_density = str(self.params['neg_density'])
        assoc = str(self.params['assoc'])
        max_edges_per_target = str(self._resolve_max_edges_per_target() or 0)
        matrix_format = str(self.params.get('matrixFormat') or self.params.get('sparseMatrix') or 'auto')
        if matrix_format.lower() in {'true', 'yes', 'on', '1'}:
            matrix_format = 'sparse'
        elif matrix_format.lower() in {'false', 'no', 'off', '0'}:
            matrix_format = 'dense'
        sparse_density_threshold = str(self.params.get('sparseDensityThreshold', 0.35))
        csv_chunk_size = str(self.params.get('csvChunkSize', 1000))
        work_mount = shlex.quote(f"{self.working_dir}:/usr/working_dir")

        cmdToRun = ' '.join(['docker run --rm',
                            f"-v {work_mount}",
                            '--expose=41269',
                            f'{self.image} /bin/sh -c \"time -v -o',
                            "/usr/working_dir/time.txt", 'python run_scSGL.py',
                            '--expression_file=/usr/working_dir/ExpressionData.csv',
                            '--ground_truth_net_file=/usr/working_dir/GroundTruthNetwork.csv',
                            '--out_file=/usr/working_dir/outFile.txt',
                            '--pos_density='+pos_density, '--neg_density='+neg_density, '--assoc='+assoc,
                            '--matrix_format='+matrix_format,
                            '--sparse_density_threshold='+sparse_density_threshold,
                            '--csv_chunk_size='+csv_chunk_size,
                            '--max_regulators_per_target='+max_edges_per_target,
                            '\"'])

        self._run_docker(cmdToRun)

    def parseOutput(self):
        '''
        Function to parse outputs from SCSGL.
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
            source_col='Gene1',
            target_col='Gene2',
            score_col='EdgeWeight',
        )
