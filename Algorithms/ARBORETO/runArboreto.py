from optparse import OptionParser
import os
from pathlib import Path
import sys
import pandas as pd
from arboreto.algo import diy, grnboost2
from distributed import Client

DEFAULT_GENIE3_TREES = 500
GENIE3_RF_KWARGS = {
    'n_jobs': 1,
    'n_estimators': DEFAULT_GENIE3_TREES,
    'max_features': 'sqrt',
}

def parseArgs(args):
    parser = OptionParser()

    parser.add_option('', '--algo', type = 'str',
                      help='Algorithm to run. Can either by GENIE3 or GRNBoost2')

    parser.add_option('', '--inFile', type='str',
                      help='Path to input tab-separated expression SamplesxGenes file')

    parser.add_option('', '--outFile', type = 'str',
                      help='File where the output network is stored')

    parser.add_option('', '--nWorkers', type='int', default=None,
                      help='Number of local Dask workers. Defaults to CPU count.')

    parser.add_option('', '--threadsPerWorker', type='int', default=1,
                      help='Number of threads per local Dask worker.')

    parser.add_option('', '--genie3Trees', type='int', default=DEFAULT_GENIE3_TREES,
                      help='Number of random forest trees to use for GENIE3.')

    parser.add_option('', '--runRoot', type='str', default=None,
                      help='Batch mode root containing run_id/algorithm/working_dir folders.')

    parser.add_option('', '--runIds', type='str', default=None,
                      help='Comma-separated run IDs to execute in batch mode.')

    parser.add_option('', '--algorithmId', type='str', default=None,
                      help='Algorithm output directory name used in batch mode.')

    (opts, args) = parser.parse_args(args)

    return opts, args

def run_inference(algo, in_file, out_file, client, genie3_trees):
    inDF = pd.read_csv(in_file, sep = '\t', index_col = 0, header = 0)
    print(f"{algo} loaded expression shape {inDF.shape} from {in_file}", flush=True)

    normalized_algo = str(algo).upper()
    if normalized_algo == 'GENIE3':
        rf_kwargs = dict(GENIE3_RF_KWARGS)
        rf_kwargs['n_estimators'] = genie3_trees
        print(f"GENIE3 using n_estimators={rf_kwargs['n_estimators']}", flush=True)
        network = diy(
            expression_data=inDF.to_numpy(),
            regressor_type='RF',
            regressor_kwargs=rf_kwargs,
            client_or_address=client,
            gene_names=inDF.columns,
        )
    elif normalized_algo == 'GRNBOOST2':
        network = grnboost2(inDF.to_numpy(), client_or_address = client, gene_names = inDF.columns)
    else:
        raise ValueError("Wrong algorithm name. Should either be GENIE3 or GRNBoost2.")

    network.to_csv(out_file, index = False, sep = '\t')

def main(args):
    opts, args = parseArgs(args)

    n_workers = opts.nWorkers or os.cpu_count() or 1
    client = Client(
        processes=False,
        n_workers=n_workers,
        threads_per_worker=opts.threadsPerWorker,
    )

    try:
        if opts.runRoot:
            if not opts.runIds or not opts.algorithmId:
                raise ValueError("--runRoot batch mode requires --runIds and --algorithmId.")

            run_root = Path(opts.runRoot)
            run_ids = [run_id.strip() for run_id in opts.runIds.split(',') if run_id.strip()]
            for run_id in run_ids:
                working_dir = run_root / run_id / opts.algorithmId / 'working_dir'
                run_inference(
                    opts.algo,
                    working_dir / 'ExpressionData.csv',
                    working_dir / 'outFile.txt',
                    client,
                    opts.genie3Trees,
                )
        else:
            if not opts.inFile or not opts.outFile:
                raise ValueError("Single-run mode requires --inFile and --outFile.")
            run_inference(opts.algo, opts.inFile, opts.outFile, client, opts.genie3Trees)
    finally:
        client.close()
                        
if __name__ == "__main__":
    main(sys.argv)
