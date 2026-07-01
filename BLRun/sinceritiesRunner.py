import pandas as pd
import shlex

from BLRun.runner import Runner
from BLRun.sparse_utils import (
    cell_positions,
    read_expression_sparse,
    write_cell_by_gene_matrix,
)


class SINCERITIESRunner(Runner):
    """Concrete runner for the SINCERITIES GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for SINCERITIES.
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
            # Select cells belonging to each pseudotime trajectory
            colName = colNames[idx]
            index = PTData[colName].index[PTData[colName].notnull()]
            exprName = "ExpressionData"+str(idx)+".csv"
            selected_cells = index.astype(str).tolist()
            # Perform quantile binning as recommeded in the paper
            # http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html#pandas.qcut
            nBins = int(self.params['nBins'])
            tQuantiles = pd.qcut(PTData.loc[index,colName], q = nBins, duplicates ='drop')
            mid = [(a.left + a.right)/2 for a in tQuantiles]

            write_cell_by_gene_matrix(
                ExpressionData,
                genes,
                cell_positions(cells, selected_cells),
                self.working_dir / exprName,
                delimiter=',',
                include_header=True,
                append_columns=[('Time', mid)],
            )

    def run(self):
        '''
        Function to run SINCERITIES algorithm
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
                                'Rscript MAIN.R',
                                "/usr/working_dir/ExpressionData" + str(idx) + ".csv",
                                "/usr/working_dir/outFile" + str(idx) + ".txt", '\"'])

            self._run_docker(cmdToRun, append=(idx > 0))

    def parseOutput(self):
        '''
        Function to parse outputs from SINCERITIES.
        '''
        workDir = self.working_dir

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)
        colNames = PTData.columns
        edge_files = []
        for idx in range(len(colNames)):
            # Read output
            outFile = 'outFile'+str(idx)+'.txt'
            if not (workDir / outFile).exists():
                # Quit if output file does not exist
                print(str(workDir / outFile) + ' does not exist, skipping...')
                return
            edge_files.append(workDir / outFile)

        self._write_ranked_edges_from_edge_files(
            edge_files,
            sep=',',
            source_col='TargetGENES',
            target_col='SourceGENES',
            score_col='Interaction',
        )
