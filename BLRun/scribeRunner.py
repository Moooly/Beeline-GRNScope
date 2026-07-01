import pandas as pd
import shlex

from BLRun.runner import Runner
from BLRun.sparse_utils import (
    cell_positions,
    read_expression_sparse,
    write_gene_by_cell_matrix,
)


class SCRIBERunner(Runner):
    """Concrete runner for the SCRIBE GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for SCRIBE.
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
            write_gene_by_cell_matrix(
                ExpressionData,
                genes,
                cell_positions(cells, selected_cells),
                self.working_dir / exprName,
                delimiter=',',
                include_header=True,
                include_gene_column=True,
                header_gene_label='',
                cell_names=selected_cells,
            )
            cellName = "pseudoTimeData"+str(idx)+".csv"
            ptDF = PTData.loc[index,[colName]]
            # Scribe expects a column labeled Time.
            ptDF.rename(columns = {colName:'Time'}, inplace = True)

            ptDF.to_csv(self.working_dir / cellName,
                                     sep = ',', header  = True, index = True)

        SCRIBE_GENE_FILE = self.working_dir / "GeneData.csv"
        if not SCRIBE_GENE_FILE.exists():
            # required column!!
            geneDict = {}
            geneDict['gene_short_name'] = [gene.replace('x_', '') for gene in genes]

            geneDF = pd.DataFrame(geneDict, index = genes)
            geneDF.to_csv(SCRIBE_GENE_FILE,
                          sep = ',', header = True)

    def run(self):
        '''
        Function to run SCRIBE algorithm.
        To see all the inputs runScribe.R script takes, run:
        docker run scribe:base /bin/sh -c "Rscript runScribe.R -h"
        '''

        # required inputs
        delay = str(self.params['delay'])
        method = str(self.params['method'])
        low = str(self.params['lowerDetectionLimit'])
        fam = str(self.params['expressionFamily'])
        max_edges_per_target = str(self._resolve_max_edges_per_target() or 0)

        # Build the command to run Scribe
        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)
        colNames = PTData.columns
        work_mount = shlex.quote(f"{self.working_dir}:/usr/working_dir")

        for idx in range(len(colNames)):
            # Specify file names for inputs and outputs
            exprName = "ExpressionData"+str(idx)+".csv"
            cellName = "pseudoTimeData"+str(idx)+".csv"
            outFile = "outFile"+str(idx)+".csv"
            timeFile = 'time'+str(idx)+".txt"

            cmdToRun = ' '.join(['docker run --rm',
                           f"-v {work_mount}",
                           f'{self.image} /bin/sh -c \"time -v -o',
                           "/usr/working_dir/" + timeFile, 'Rscript runScribe.R',
                           '-e', "/usr/working_dir/" + exprName, '-c', "/usr/working_dir/" + cellName,
	                           '-g', "/usr/working_dir/GeneData.csv", '-o /usr/working_dir/', '-d', delay, '-l', low,
	                           '-m', method, '-x', fam, '--outFile ' + outFile,
	                           '--maxRegulatorsPerTarget', max_edges_per_target])

            if str(self.params['log']) == 'True':
                cmdToRun += ' --log'
            if str(self.params['ignorePT']) == 'True':
                cmdToRun += ' -i'

            cmdToRun += '\"'

            self._run_docker(cmdToRun, append=(idx > 0))

    def parseOutput(self):
        '''
        Function to parse outputs from SCRIBE.
        '''
        workDir = self.working_dir

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)
        colNames = PTData.columns
        edge_files = []
        for idx in range(len(colNames)):
            # Read output
            outFile = 'outFile'+str(idx)+'.csv'
            if not (workDir / outFile).exists():
                # Quit if output file does not exist
                print(str(workDir / outFile) + ' does not exist, skipping...')
                return
            edge_files.append(workDir / outFile)

        self._write_ranked_edges_from_edge_files(
            edge_files,
            sep=' ',
            header=None,
            names=['Gene1', 'Gene2', 'EdgeWeight'],
        )
