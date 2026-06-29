# Please refer to https://github.com/SPLab-aviyente/scSGL/blob/main/notebooks/demo.ipynb

import argparse
import pandas as pd #to load read GSD dataset
import numpy as np
import sys
sys.path.append('scSGL') #to add a path to search for the requested module

from pysrc.graphlearning import learn_signed_graph


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run scSGL algorithm.') 
    parser.add_argument('--expression_file', 
        help='Path to ExpressionData file')
    parser.add_argument('--ground_truth_net_file', 
        help='Path to groundTruthNetwork file')
    parser.add_argument('--pos_density', default='0.45', #to control the density of positive part of the learned signed graph
        help='Positive density')
    parser.add_argument('--neg_density', default='0.45', #to control the density of negative part of the learned signed graph
        help='Negative density')
    parser.add_argument('--assoc', default='correlation', #to infer a signed graph with correlation kernel
        help='Association type')
    parser.add_argument('--out_file',
        help='Path to output file')
    parser.add_argument('--max_regulators_per_target', type=int, default=0,
        help='Keep only this many strongest regulators per target. Use 0 to write all edges.')
    
    return parser

def parse_arguments():
    parser = get_parser()
    opts = parser.parse_args()

    return opts

def main(args):
    #python run_scSGL.py --expression_file scSGL/data/inputs/GSD/ExpressionData.csv --ground_truth_net_file scSGL/data/inputs/GSD/GroundTruthNetwork.csv --out_file outFile.txt

    opts = parse_arguments()
    expression_df = pd.read_csv(opts.expression_file, index_col=0)  #to read gene expression file
    ref_net_df = pd.read_csv(opts.ground_truth_net_file) #to read reference network file

    #Learn signed graph with the parameters
    G = learn_signed_graph(expression_df.to_numpy(), pos_density=float(opts.pos_density), neg_density=float(opts.neg_density),
                                assoc=opts.assoc, gene_names=np.array(expression_df.index))
    #G is a dataframe with each row indicating an edge between two genes.
    #Each edge is also associated with a weight, which is either positive or negative depending on the sign of the edge.
    if opts.max_regulators_per_target and opts.max_regulators_per_target > 0:
        G = G.copy()
        G["EdgeWeight"] = pd.to_numeric(G["EdgeWeight"], errors="coerce")
        G = G.dropna(subset=["Gene1", "Gene2", "EdgeWeight"])
        G["_abs_weight"] = G["EdgeWeight"].abs()
        G = (
            G.sort_values(["Gene2", "_abs_weight", "Gene1"], ascending=[True, False, True])
             .groupby("Gene2", sort=False)
             .head(opts.max_regulators_per_target)
             .drop(columns=["_abs_weight"])
        )

    G.to_csv(opts.out_file, index = False, sep = '\t')  #to write the output file

if __name__ == "__main__":
    main(sys.argv)
