import os
import pandas as pd

from BLRun.runner import Runner


class SINGERunner(Runner):
    """Concrete runner for the SINGE GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for SINGE.
        If the folder/files under self.input_dir exist,
        this function will not do anything.
        '''

        ExpressionData = pd.read_csv(self.input_dir / self.exprData,
                                         header = 0, index_col = 0)
        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        for idx in range(len(colNames)):
            # Select cells belonging to each pseudotime trajectory
            colName = colNames[idx]
            index = PTData[colName].index[PTData[colName].notnull()]
            exprName = "ExpressionData"+str(idx)+".csv"
            newExpressionData = ExpressionData.loc[:,index].T
            newExpressionData['PseudoTime'] = PTData.loc[index,colName]
            newExpressionData.to_csv(self.working_dir / exprName,
                                 sep = ',', header  = True, index = False)

    def run(self):
        '''
        Function to run SINGE algorithm
        '''

        # if the parameters aren't specified, then use default parameters
        # TODO allow passing in multiple sets of hyperparameters
        # these must be in the right order!
        params_order = [
            'lambda', 'dT', 'num_lags', 'kernel_width',
            'prob_zero_removal', 'prob_remove_samples',
            'family'
        ]
        default_params = {
            'lambda': '0.01',
            'dT': '10',
            'num_lags': '5',
            'kernel_width': '4',
            'prob_zero_removal': '0',
            'prob_remove_samples': '0.2',
            'family': 'gaussian',
            'num_replicates': '2',
        }
        params = self.params
        for param, val in default_params.items():
            if param not in params:
                params[param] = val

        num_replicates = int(params['num_replicates'])
        replicates = []
        for replicate in range(num_replicates):
           replicates.append(' '.join('--' + p.replace('_', '-') + ' ' + str(params[p]) for p in params_order) + ' '.join(['', '--replicate', str(replicate), '--ID', str(replicate)]))
        params_str = '\n'.join(replicates)

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        for idx in range(len(colNames)):
            os.makedirs(str(self.working_dir / str(idx)), exist_ok = True)

            outFileSymlink = "out" + str(idx)
            inputFile = "/usr/working_dir/ExpressionData"+str(idx)+".csv"
            inputMat = "/usr/working_dir/ExpressionData"+str(idx)+".mat"
            geneListMat = "/usr/working_dir/GeneList"+str(idx)+".mat"
            paramsFile = "/usr/working_dir/hyperparameters.txt"

            '''
            This is a workaround for https://github.com/gitter-lab/SINGE/blob/master/code/parseParams.m#L39
            not allowing '/' characters in the outDir parameter.
            '''
            symlink_out_file = ' '.join(['ln -s', "/usr/working_dir/" + str(idx) + "/", outFileSymlink])

            '''
            See https://github.com/gitter-lab/SINGE/blob/master/README.md.  SINGE expects a data matfile with variables "X" and "ptime",
            and a gene_list matfile with the variable "gene_list".

            Saving fullKp is a very hacky workaround for https://github.com/gitter-lab/SINGE/blob/master/code/iLasso_for_SINGE.m#L56,
            that assumes this input was saved in matfile v7.3 which octave does not support.
            '''
            convert_input_to_matfile = 'octave -q --eval \\"CSV = csvread(\'' + inputFile + '\'); ' + \
                                 'X = sparse(CSV(2:end,1:end-1).\'); ptime = CSV(2:end,end).\'; ' + \
                                 'Kp2.Kp = single(ptime); Kp2.sumKp = single(ptime*X.\'); fullKp(1, ' + \
                                 str(int(params['dT'])*int(params['num_lags'])) + ') = Kp2; ' + \
                                 'save(\'-v7\',\'' + inputMat + '\', \'X\', \'ptime\', \'fullKp\'); ' + \
                                 'f = fopen(\'' + inputFile + '\'); gene_list = strsplit(fgetl(f), \',\')(1:end-1).\'; fclose(f); ' + \
                                 'save(\'-v7\',\'' + geneListMat + '\', \'gene_list\')\\"'

            cmdToRun = ' '.join(['docker run --rm --entrypoint /bin/sh',
                                f"-v {self.working_dir}:/usr/working_dir",
                                f'{self.image} -c \"echo \\"',
                                 params_str, '\\" >', paramsFile, '&&', symlink_out_file, '&&', convert_input_to_matfile,
                                 '&& time -v -o', "/usr/working_dir/time" + str(idx) + ".txt",
                                 '/usr/local/SINGE/SINGE.sh /usr/local/MATLAB/MATLAB_Runtime/v94 standalone',
                                 inputMat, geneListMat, outFileSymlink, paramsFile, '\"'])

            self._run_docker(cmdToRun, append=(idx > 0))

    def parseOutput(self):
        '''
        Function to parse outputs from SINGE.
        '''
        workDir = self.working_dir

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        edge_files = []

        for idx in range(len(colNames)):

            # Quit if output directory does not exist
            if not (workDir / str(idx) / 'SINGE_Ranked_Edge_List.txt').exists():
                print(str(workDir / str(idx) / 'SINGE_Ranked_Edge_List.txt') + ' does not exist, skipping...')
                return
            edge_files.append(workDir / str(idx) / 'SINGE_Ranked_Edge_List.txt')

        self._write_ranked_edges_from_edge_files(
            edge_files,
            sep='\t',
            header=0,
            names=['Gene1', 'Gene2', 'EdgeWeight'],
        )
