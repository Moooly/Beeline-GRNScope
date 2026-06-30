FROM mambaorg/micromamba:1.5.8

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        time \
    && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER
RUN micromamba install -y -n base -c conda-forge \
        python=3.10 \
        pip \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        anndata \
        scanpy \
    && micromamba clean -a -y

RUN pip install --no-cache-dir celloracle

COPY runCellOracle.py /runCellOracle.py
