ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}tensorflow/tensorflow:2.11.0-gpu

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
       git-core \
       wget \
       build-essential \
       libglib2.0-0 \
       libsm6 \
       libxrender-dev \
       libxext6 \
       libgl1-mesa-glx \
       libpq-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 --no-cache-dir install \
        git+https://github.com/wairas/happy-tools.git \
        git+https://github.com/wairas/happy-tools-keras.git

COPY bash.bashrc /etc/bash.bashrc

WORKDIR /workspace
