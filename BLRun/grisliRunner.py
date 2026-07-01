import os
import pandas as pd
import shlex

from BLRun.runner import Runner
from BLRun.sparse_utils import (
    cell_positions,
    read_expression_sparse,
    write_gene_by_cell_matrix,
)


class GRISLIRunner(Runner):
    """Concrete runner for the GRISLI GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for GRISLI.
        If the folder/files under self.input_dir exist,
        this function will not do anything.
        '''

        ExpressionData, genes, cells, density = read_expression_sparse(
            self.input_dir / self.exprData,
            chunksize=self.params.get('csvChunkSize', 1000),
        )
        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        for idx in range(len(colNames)):
            (self.working_dir / str(idx)).mkdir(exist_ok = True)

            # Select cells belonging to each pseudotime trajectory
            colName = colNames[idx]
            index = PTData[colName].index[PTData[colName].notnull()]

            exprName = str(idx)+"/ExpressionData.tsv"
            selected_cells = index.astype(str).tolist()
            write_gene_by_cell_matrix(
                ExpressionData,
                genes,
                cell_positions(cells, selected_cells),
                self.working_dir / exprName,
                delimiter='\t',
                include_header=False,
                include_gene_column=False,
            )

            cellName = str(idx)+"/PseudoTime.tsv"
            ptDF = PTData.loc[index,[colName]]
            ptDF.to_csv(self.working_dir / cellName,
                                     sep = '\t', header  = False, index = False)

    def run(self):
        '''
        Function to run GRISLI algorithm
        '''

        L = str(self.params['L'])
        R = str(self.params['R'])
        alphaMin = str(self.params['alphaMin'])

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        work_mount = shlex.quote(f"{self.working_dir}:/usr/working_dir")
        for idx in range(len(colNames)):
            os.makedirs(str(self.working_dir / str(idx)), exist_ok = True)

            cmdToRun = ' '.join(['docker run --rm',
                                f"-v {work_mount}",
                                f'{self.image} /bin/sh -c \"time -v -o',
                                "/usr/working_dir/time" + str(idx) + ".txt",
                                './GRISLI',
                                "/usr/working_dir/" + str(idx) + "/",
                                "/usr/working_dir/" + str(idx) + "/outFile.txt",
                                L, R, alphaMin, '\"'])

            self._run_docker(cmdToRun, append=(idx > 0))

    def parseOutput(self):
        '''
        Function to parse outputs from GRISLI.
        '''
        workDir = self.working_dir

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)
        colNames = PTData.columns
        gene_list = self._read_gene_names(self.input_dir / self.exprData)
        target_candidates = {}

        for indx in range(len(colNames)):
            # Read output
            outFile = str(indx)+'/outFile.txt'
            if not (workDir / outFile).exists():
                # Quit if output file does not exist
                print(str(workDir / outFile) + ' does not exist, skipping...')
                return
            OutDF = pd.read_csv(workDir / outFile, sep = ',', header = None)
            max_rank_score = len(gene_list) * len(gene_list)
            self._update_candidates_from_matrix(
                target_candidates,
                OutDF.values,
                gene_list,
                transform=lambda values, offset=max_rank_score: offset - values,
            )

        self._write_candidate_edges(target_candidates)
