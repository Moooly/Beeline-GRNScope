import pandas as pd
import shlex

from BLRun.runner import Runner
from BLRun.sparse_utils import (
    cell_positions,
    read_expression_sparse,
    write_gene_by_cell_matrix,
)


class GRNVBEMRunner(Runner):
    """Concrete runner for the GRN-VBEM GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for GRNVBEM.
        It will create the input folder at self.working_dir if it
        does not exist already. The input folder will contain an ExpressionData.csv with
        cells ordered according to the pseudotime along the columns, and genes along
        the rows. If the files already exist, this function will overwrite it.
        '''

        ExpressionData, genes, cells, density = read_expression_sparse(
            self.input_dir / self.exprData,
            chunksize=self.params.get('csvChunkSize', 1000),
        )
        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        for idx in range(len(colNames)):
            # Select cells belonging to each pseudotime trajectory
            colName = colNames[idx]
            index = PTData[colName].index[PTData[colName].notnull()]
            exprName = "ExpressionData"+str(idx)+".csv"

            subPT = PTData.loc[index,:]
            # Order columns by PseudoTime
            selected_cells = subPT.sort_values([colName]).index.astype(str).tolist()
            write_gene_by_cell_matrix(
                ExpressionData,
                genes,
                cell_positions(cells, selected_cells),
                self.working_dir / exprName,
                delimiter=',',
                include_header=True,
                include_gene_column=True,
                header_gene_label='GENES',
                cell_names=selected_cells,
            )

    def run(self):
        '''
        Function to run GRN-VBEM algorithm
        '''

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        work_mount = shlex.quote(f"{self.working_dir}:/usr/working_dir")
        for idx in range(len(colNames)):
            cmdToRun = ' '.join(['docker run --rm',
                                f"-v {work_mount}",
                                f'{self.image} /bin/sh -c \"time -v -o',
                                "/usr/working_dir/time" + str(idx) + ".txt",
                                './GRNVBEM',
                                "/usr/working_dir/ExpressionData" + str(idx) + ".csv",
                                "/usr/working_dir/outFile" + str(idx) + ".txt", '\"'])

            self._run_docker(cmdToRun, append=(idx > 0))

    def parseOutput(self):
        '''
        Function to parse outputs from GRNVBEM.
        '''
        workDir = self.working_dir

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        edge_files = []

        for indx in range(len(colNames)):
            outFileName = 'outFile'+str(indx)+'.txt'
            # Quit if output file does not exist
            if not (workDir / outFileName).exists():
                print(str(workDir / outFileName) + ' does not exist, skipping...')
                return
            edge_files.append(workDir / outFileName)

        self._write_ranked_edges_from_edge_files(
            edge_files,
            sep='\t',
            source_col='Parent',
            target_col='Child',
            score_col='Probability',
        )
