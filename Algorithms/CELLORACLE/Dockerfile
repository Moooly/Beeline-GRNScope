FROM mambaorg/micromamba:1.5.8

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        time \
    && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER
RUN micromamba install -y -n base -c conda-forge -c bioconda \
        python=3.10 \
        pip \
        setuptools \
        wheel \
        cython \
        numpy=1.26.4 \
        pandas=1.5.3 \
        scipy \
        scikit-learn \
        h5py \
        'matplotlib-base<3.7' \
        seaborn \
        numba \
        'anndata<=0.10.8' \
        'scanpy>=1.6' \
        umap-learn \
        pyarrow \
        tqdm \
        joblib \
        python-igraph \
        louvain \
        goatools \
        genomepy \
        'gimmemotifs>=0.14.4,<=0.17.2' \
        velocyto.py \
        bedtools \
        jupyter \
    && micromamba clean -a -y

RUN micromamba run -n base python -m pip install --no-cache-dir --only-binary=:all: --no-deps celloracle==0.20.0 \
    && micromamba run -n base python -c "import celloracle, velocyto, scanpy, anndata, genomepy, goatools, gimmemotifs, igraph, louvain; print('CellOracle image import check passed')"

COPY runCellOracle.py /runCellOracle.py

# BEELINE bind-mounts host-owned working directories and expects algorithm
# containers to write outputs there. Running as root avoids UID mismatches on
# Linux hosts while keeping the conda environment installed above.
USER root
