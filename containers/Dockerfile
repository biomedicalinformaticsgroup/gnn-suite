# TODO: fix pytorch version and torch_geometric version - consider pip install[?]

FROM pytorch/pytorch:latest

LABEL org.opencontainers.image.title="gnn-suite"
LABEL org.opencontainers.image.description="A comprehensive GNN framework with PyTorch Geometric and CUDA 12.1 support"
LABEL org.opencontainers.image.url="https://github.com/biomedicalinformaticsgroup/gnn-suite"
LABEL org.opencontainers.image.documentation="https://github.com/biomedicalinformaticsgroup/gnn-suite"
LABEL org.opencontainers.image.source="https://github.com/biomedicalinformaticsgroup/gnn-suite"
LABEL org.opencontainers.image.revision="v1.0.0"
LABEL org.opencontainers.image.vendor="biomedicalinformaticsgroup"
LABEL org.opencontainers.image.authors="Giovanni Stracquadanio"

USER root
WORKDIR /
ARG PACKAGE_FILE=containers/pkg.list

# copy package list
COPY ${PACKAGE_FILE} pkg.list

# Install basic tools
RUN apt-get update \
    && apt-get install --yes $(cat pkg.list) \
    && apt-get autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch Geometric and extension packages
RUN TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])") && \
    CUDA_VERSION=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))") && \
    pip install --no-cache-dir torch-geometric && \
    pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv \
        -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html

# Install other Python packages
RUN pip install --no-cache-dir \
    scikit-learn numpy pandas \
    seaborn matplotlib \
    typer pyyaml \
    optuna \
    mlflow \
    black pylint \
    jupyter ipykernel

# Add Tini
ENV TINI_VERSION=v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini

# setup CUDA variables and linking libraries
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
ENV CPATH=/usr/local/cuda/include:/usr/local/cuda/compat
ENV CUDA_HOME=/usr/local/cuda

ENV CUDA=cu118
RUN mkdir -p /usr/local/cuda/
RUN cd /usr/local/cuda/ && ln -s /usr/local/cuda/compat/libnvidia-* .

# Legacy pip installation method (now using pip install above after conda)
#RUN pip install torch-scatter \
#    && pip install torch-sparse \
#    && pip install torch-cluster \
#    && pip install torch-spline-conv \
#    && pip install torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html \
#    && pip install ogb

ENTRYPOINT ["/tini", "--"]
CMD ["python3"]

