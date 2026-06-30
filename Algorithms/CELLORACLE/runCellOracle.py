#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROMOTER_BASE_GRN_LOADERS = {
    "human": "load_human_promoter_base_GRN",
    "mouse": "load_mouse_promoter_base_GRN",
    "rat": "load_rat_promoter_base_GRN",
    "pig": "load_Pig_promoter_base_GRN",
    "chicken": "load_chicken_promoter_base_GRN",
    "zebrafish": "load_zebrafish_promoter_base_GRN",
    "xenopus_tropicalis": "load_xenopus_tropicalis_promoter_base_GRN",
    "drosophila": "load_drosophila_promoter_base_GRN",
    "c_elegans": "load_Celegans_promoter_base_GRN",
    "s_cerevisiae": "load_Scerevisiae_promoter_base_GRN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CellOracle for GRNScope.")
    parser.add_argument("--inFile", required=True)
    parser.add_argument("--outFile", required=True)
    parser.add_argument("--detailsFile", required=True)
    parser.add_argument("--species", default="human")
    parser.add_argument("--baseGrn", default="auto")
    parser.add_argument("--clusterName", default="Global")
    parser.add_argument("--maxGenes", type=int, default=3000)
    parser.add_argument("--maxCells", type=int, default=30000)
    parser.add_argument("--minCells", type=int, default=50)
    parser.add_argument("--pValueCutoff", type=float, default=0.05)
    parser.add_argument("--maxRegulatorsPerTarget", type=int, default=25)
    parser.add_argument("--randomSeed", type=int, default=1729)
    return parser.parse_args()


def read_expression(path: str) -> pd.DataFrame:
    expression = pd.read_csv(path, sep=None, engine="python", index_col=0)
    if expression.empty:
        raise ValueError("Expression matrix is empty.")
    expression = expression.apply(pd.to_numeric, errors="coerce")
    if expression.isna().any().any():
        raise ValueError("Expression matrix contains non-numeric values after preprocessing.")
    expression.index = expression.index.astype(str)
    expression.columns = expression.columns.astype(str)
    return expression.astype(np.float32, copy=False)


def cap_cells(expression: pd.DataFrame, max_cells: int, random_seed: int) -> pd.DataFrame:
    if max_cells <= 0 or expression.shape[1] <= max_cells:
        return expression
    rng = np.random.default_rng(random_seed)
    selected = np.sort(rng.choice(expression.shape[1], size=max_cells, replace=False))
    return expression.iloc[:, selected]


def cap_genes_by_variance(expression: pd.DataFrame, max_genes: int) -> pd.DataFrame:
    if max_genes <= 0 or expression.shape[0] <= max_genes:
        return expression
    variances = expression.var(axis=1).to_numpy(dtype=np.float64)
    selected = np.argpartition(variances, -max_genes)[-max_genes:]
    selected = selected[np.argsort(variances[selected])[::-1]]
    selected = np.sort(selected)
    return expression.iloc[selected, :]


def load_base_grn(species: str, base_grn: str):
    import celloracle as co

    normalized_species = species.strip().lower()
    normalized_base = base_grn.strip()

    if normalized_species == "mouse" and normalized_base in {"auto", "mouse_scATAC_atlas"}:
        for loader_name in (
            "load_mouse_scATAC_atlas_base_GRN",
            "load_TFinfo_df_mm9_mouse_atac_atlas",
        ):
            loader = getattr(co.data, loader_name, None)
            if loader is not None:
                return loader()

    loader_name = PROMOTER_BASE_GRN_LOADERS.get(normalized_species)
    if loader_name is None:
        raise ValueError(f"Unsupported CellOracle species: {species}")

    loader = getattr(co.data, loader_name, None)
    if loader is None:
        raise ValueError(
            f"Installed CellOracle does not provide base-GRN loader {loader_name}."
        )
    return loader()


def filter_base_grn_to_expression(base_grn: pd.DataFrame, genes: set[str]) -> pd.DataFrame:
    if not isinstance(base_grn, pd.DataFrame):
        return base_grn

    filtered = base_grn.copy()
    if "gene_short_name" in filtered.columns:
        filtered = filtered[filtered["gene_short_name"].astype(str).isin(genes)]

    retained_columns = []
    for column in filtered.columns:
        column_name = str(column)
        if column_name in {"peak_id", "gene_short_name"} or column_name in genes:
            retained_columns.append(column)

    if retained_columns:
        filtered = filtered.loc[:, retained_columns]

    if filtered.empty:
        raise ValueError(
            "The CellOracle base GRN has no target genes overlapping the expression matrix."
        )
    return filtered


def make_embedding(values: np.ndarray) -> np.ndarray:
    cell_count = values.shape[0]
    if cell_count == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if cell_count == 1:
        return np.zeros((1, 2), dtype=np.float32)

    try:
        from sklearn.decomposition import TruncatedSVD

        n_components = min(2, values.shape[0] - 1, values.shape[1] - 1)
        if n_components <= 0:
            return np.zeros((cell_count, 2), dtype=np.float32)
        embedding = TruncatedSVD(
            n_components=n_components,
            random_state=1729,
        ).fit_transform(values)
    except Exception:
        embedding = np.zeros((cell_count, min(2, values.shape[1])), dtype=np.float32)

    if embedding.shape[1] < 2:
        embedding = np.pad(embedding, ((0, 0), (0, 2 - embedding.shape[1])))
    return np.asarray(embedding[:, :2], dtype=np.float32)


def import_expression_into_oracle(oracle, adata, cluster_column_name: str) -> None:
    for method_name in (
        "import_anndata_as_normalized_count",
        "import_anndata_as_raw_count",
    ):
        method = getattr(oracle, method_name, None)
        if method is None:
            continue
        try:
            method(
                adata=adata,
                cluster_column_name=cluster_column_name,
                embedding_name="X_grnscope",
            )
            return
        except TypeError:
            method(adata=adata, cluster_column_name=cluster_column_name)
            return

    raise ValueError("Installed CellOracle Oracle object cannot import AnnData.")


def run_oracle(expression: pd.DataFrame, base_grn: pd.DataFrame, cluster_name: str):
    import anndata as ad
    import celloracle as co

    values = expression.T.to_numpy(dtype=np.float32, copy=True)
    adata = ad.AnnData(X=values)
    adata.obs_names = list(expression.columns)
    adata.var_names = list(expression.index)
    adata.obs["grnscope_cluster"] = cluster_name
    adata.obsm["X_grnscope"] = make_embedding(values)

    oracle = co.Oracle()
    import_expression_into_oracle(oracle, adata, "grnscope_cluster")
    oracle.import_TF_data(TF_info_matrix=base_grn)

    try:
        oracle.perform_PCA()
    except TypeError:
        oracle.perform_PCA(n_components=min(50, max(2, expression.shape[1] - 1)))

    try:
        oracle.knn_imputation(
            n_pca_dims=min(50, max(2, expression.shape[1] - 1)),
            k=max(5, min(30, expression.shape[1] // 10)),
            balanced=True,
            b_sight=max(5, min(30, expression.shape[1] // 10)),
            b_maxl=max(5, min(30, expression.shape[1] // 10)),
        )
    except TypeError:
        oracle.knn_imputation()

    try:
        return oracle.get_links(
            cluster_name_for_GRN_unit="grnscope_cluster",
            alpha=10,
            verbose_level=0,
        )
    except TypeError:
        return oracle.get_links(cluster_name_for_GRN_unit="grnscope_cluster")


def extract_links_dataframe(links, cluster_name: str) -> pd.DataFrame:
    if hasattr(links, "links_dict"):
        links_dict = links.links_dict
        if cluster_name in links_dict:
            return links_dict[cluster_name].copy()
        if links_dict:
            return next(iter(links_dict.values())).copy()

    if isinstance(links, pd.DataFrame):
        return links.copy()

    raise ValueError("CellOracle did not return a usable link table.")


def normalize_link_columns(links_df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "TF": "source",
        "tf": "source",
        "regulator": "source",
        "target_gene": "target",
        "gene_short_name": "target",
        "coef": "coef_mean",
        "coef_mean": "coef_mean",
    }
    links = links_df.rename(columns={k: v for k, v in rename_map.items() if k in links_df.columns})

    if "source" not in links.columns or "target" not in links.columns:
        raise ValueError(
            "CellOracle link table must contain source/target columns or TF/target_gene columns."
        )

    if "coef_mean" not in links.columns:
        for candidate in ("coef_abs", "score", "importance"):
            if candidate in links.columns:
                links["coef_mean"] = pd.to_numeric(links[candidate], errors="coerce")
                break

    if "coef_mean" not in links.columns:
        raise ValueError("CellOracle link table does not contain a coefficient column.")

    links["source"] = links["source"].astype(str).str.strip()
    links["target"] = links["target"].astype(str).str.strip()
    links["coef_mean"] = pd.to_numeric(links["coef_mean"], errors="coerce")
    if "coef_abs" not in links.columns:
        links["coef_abs"] = links["coef_mean"].abs()
    else:
        links["coef_abs"] = pd.to_numeric(links["coef_abs"], errors="coerce").fillna(
            links["coef_mean"].abs()
        )

    if "p" in links.columns:
        links["p"] = pd.to_numeric(links["p"], errors="coerce")
    elif "p_value" in links.columns:
        links["p"] = pd.to_numeric(links["p_value"], errors="coerce")
    if "-logp" in links.columns:
        links["neg_logp"] = pd.to_numeric(links["-logp"], errors="coerce")
    elif "neg_logp" in links.columns:
        links["neg_logp"] = pd.to_numeric(links["neg_logp"], errors="coerce")

    links = links[
        (links["source"] != "")
        & (links["target"] != "")
        & links["coef_mean"].notna()
        & np.isfinite(links["coef_mean"])
    ].copy()
    return links


def filter_and_cap_links(
    links: pd.DataFrame,
    *,
    p_value_cutoff: float,
    max_regulators_per_target: int,
) -> pd.DataFrame:
    filtered = links.copy()
    if "p" in filtered.columns and p_value_cutoff > 0:
        filtered = filtered[(filtered["p"].isna()) | (filtered["p"] <= p_value_cutoff)]

    filtered["_abs_score"] = filtered["coef_abs"].abs()
    filtered.sort_values(
        ["target", "_abs_score", "source"],
        ascending=[True, False, True],
        inplace=True,
    )
    filtered = filtered.drop_duplicates(["source", "target"], keep="first")

    if max_regulators_per_target > 0:
        filtered = filtered.groupby("target", sort=False).head(max_regulators_per_target)

    filtered.sort_values(
        ["_abs_score", "source", "target"],
        ascending=[False, True, True],
        inplace=True,
    )
    return filtered.drop(columns=["_abs_score"])


def main() -> None:
    args = parse_args()
    expression = read_expression(args.inFile)
    expression = cap_cells(expression, args.maxCells, args.randomSeed)
    expression = cap_genes_by_variance(expression, args.maxGenes)

    if expression.shape[1] < args.minCells:
        raise ValueError(
            f"CellOracle requires at least {args.minCells} cells for this scope; "
            f"got {expression.shape[1]}."
        )

    gene_set = set(map(str, expression.index))
    base_grn = filter_base_grn_to_expression(
        load_base_grn(args.species, args.baseGrn),
        gene_set,
    )

    links = extract_links_dataframe(
        run_oracle(expression, base_grn, args.clusterName),
        args.clusterName,
    )
    links = normalize_link_columns(links)
    links["scope"] = args.clusterName
    ranked_links = filter_and_cap_links(
        links,
        p_value_cutoff=args.pValueCutoff,
        max_regulators_per_target=args.maxRegulatorsPerTarget,
    )

    details_path = Path(args.detailsFile)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    ranked_links.to_csv(details_path, sep="\t", index=False)

    out_edges = ranked_links[["source", "target", "coef_mean"]].rename(
        columns={
            "source": "Gene1",
            "target": "Gene2",
            "coef_mean": "EdgeWeight",
        }
    )
    out_edges.to_csv(args.outFile, sep="\t", index=False)


if __name__ == "__main__":
    main()
