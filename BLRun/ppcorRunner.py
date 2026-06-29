import pandas as pd

from BLRun.runner import Runner


class PPCORRunner(Runner):
    """Concrete runner for the PPCOR GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for PPCOR.
        If the folder/files under self.input_dir exist,
        this function will not do anything.
        '''

        # Create ExpressionData.csv file in the created input directory
        PPCOR_EXPRESSION_FILE = self.working_dir / "ExpressionData.csv"
        if not PPCOR_EXPRESSION_FILE.exists():
            ExpressionData = pd.read_csv(self.input_dir / self.exprData,
                                         header = 0, index_col = 0)

            newExpressionData = ExpressionData.copy()

            # Write .csv file
            newExpressionData.to_csv(PPCOR_EXPRESSION_FILE,
                                 sep = ',', header  = True, index = True)

    def run(self):
        '''
        Function to run PPCOR algorithm
        '''

        cmdToRun = ' '.join(['docker run --rm',
                            f"-v {self.working_dir}:/usr/working_dir",
                            f'{self.image} /bin/sh -c \"time -v -o',
                            "/usr/working_dir/time.txt",
                            'Rscript runPPCOR.R',
                            "/usr/working_dir/ExpressionData.csv",
                            "/usr/working_dir/outFile.txt",
                            str(self._resolve_max_edges_per_target() or 0),
                            str(self.params['pVal']), '\"'])

        # Run command
        self._run_docker(cmdToRun)

    def parseOutput(self):
        '''
        Function to parse outputs from PPCOR.
        '''
        workDir = self.working_dir
        outFile = workDir / 'outFile.txt'

        # Quit if output file does not exist
        if not outFile.exists():
            print(str(outFile) + ' does not exist, skipping...')
            return

        p_value_cutoff = float(self.params['pVal'])
        max_edges_per_target = self._resolve_max_edges_per_target()
        significant_candidates = {}
        fallback_candidates = {}

        for chunk in pd.read_csv(outFile, sep = '\t', header = 0, chunksize = 200000):
            cor_values = pd.to_numeric(chunk['corVal'], errors='coerce')
            p_values = pd.to_numeric(chunk['pValue'], errors='coerce')
            for gene1, gene2, cor_value, p_value in zip(
                chunk['Gene1'],
                chunk['Gene2'],
                cor_values,
                p_values,
            ):
                if pd.isna(cor_value):
                    continue
                self._update_candidate(
                    fallback_candidates,
                    str(gene1).strip(),
                    str(gene2).strip(),
                    float(cor_value),
                    max_edges_per_target,
                )
                if pd.isna(p_value) or float(p_value) > p_value_cutoff:
                    continue
                self._update_candidate(
                    significant_candidates,
                    str(gene1).strip(),
                    str(gene2).strip(),
                    float(cor_value),
                    max_edges_per_target,
                )

        if significant_candidates:
            self._write_candidate_edges(significant_candidates)
            return

        for target, candidates in fallback_candidates.items():
            for source in list(candidates.keys()):
                candidates[source] = 0.0
        self._write_candidate_edges(fallback_candidates)
