import os
import pandas as pd
import shlex

from BLRun.runner import Runner
from BLRun.sparse_utils import (
    cell_positions,
    read_expression_sparse,
    write_gene_by_cell_matrix,
)


class SCODERunner(Runner):
    """Concrete runner for the SCODE GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for SCODE.
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
            # Create output subdirectory in advance to prevent docker from
            # creating it with root-exclusive permissions
            (self.working_dir / str(idx)).mkdir(exist_ok=True)

            # Select cells belonging to each pseudotime trajectory
            colName = colNames[idx]
            index = PTData[colName].index[PTData[colName].notnull()]
            exprName = "ExpressionData"+str(idx)+".csv"
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
            cellName = "PseudoTime"+str(idx)+".csv"
            ptDF = PTData.loc[index,[colName]]
            # SCODE expects a column labeled PseudoTime.
            ptDF.rename(columns = {colName:'PseudoTime'}, inplace = True)
            # output file
            ptDF.to_csv(self.working_dir / cellName,
                                     sep = '\t', header  = False)

    def run(self):
        '''
        Function to run SCODE algorithm
        '''

        z = str(self.params['z'])
        nIter = str(self.params['nIter'])
        nRep = str(self.params['nRep'])

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        genes = self._read_gene_names(self.input_dir / self.exprData)
        work_mount = shlex.quote(f"{self.working_dir}:/usr/working_dir")

        for idx in range(len(colNames)):
            colName = colNames[idx]
            nCells = str(int(PTData[colName].notnull().sum()))
            nGenes = str(len(genes))

            cmdToRun = ' '.join(['docker run --rm',
                                f'--user {os.getuid()}:{os.getgid()}',
                                '-e HOME=/tmp',
                                f"-v {work_mount}",
                                f'{self.image} /bin/sh -c \"time -v -o',
                                "/usr/working_dir/time" + str(idx) + ".txt",
                                'ruby run_R.rb',
                                "/usr/working_dir/ExpressionData" + str(idx) + ".csv",
                                "/usr/working_dir/PseudoTime" + str(idx) + ".csv",
                                "/usr/working_dir/" + str(idx),
                                nGenes, z, nCells, nIter, nRep, '\"'])

            self._run_docker(cmdToRun, append=(idx > 0))

    def parseOutput(self):
        '''
        Function to parse outputs from SCODE.
        '''
        workDir = self.working_dir

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)
        colNames = PTData.columns
        gene_list = self._read_gene_names(self.input_dir / self.exprData)
        target_candidates = {}

        for indx in range(len(colNames)):
            # Read output
            outFile = str(indx)+'/meanA.txt'
            if not (workDir / outFile).exists():
                # Quit if output file does not exist
                print(str(workDir / outFile) + ' does not exist, skipping...')
                return
            OutDF = pd.read_csv(workDir / outFile, sep = '\t', header = None)

            self._update_candidates_from_matrix(
                target_candidates,
                OutDF.values,
                gene_list,
                absolute_scores=True,
            )

        self._write_candidate_edges(target_candidates)
