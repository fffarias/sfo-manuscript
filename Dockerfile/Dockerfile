FROM ubuntu:20.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libbz2-dev \
        libffi-dev \
        libibverbs-dev \
        liblzma-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        llvm \
        tar \
        tk-dev \
        wget \
        xz-utils \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

RUN wget  --no-check-certificate https://www.python.org/ftp/python/3.9.9/Python-3.9.9.tgz && \
    tar xvf Python-3.9.9.tgz && \
    cd Python-3.9.9 && \
    ./configure  --enable-optimizations --with-ensurepip=install && \
    make -j 8 && \
    make altinstall

RUN /usr/local/bin/python3.9 -m venv /venv && \
    /venv/bin/pip3.9 install --upgrade pip && \
    /venv/bin/pip3.9 install wheel && \
    /venv/bin/pip3.9 install matplotlib && \
    /venv/bin/pip3.9 install pandas && \
    /venv/bin/pip3.9 install -U ray

RUN git clone https://github.com/devitocodes/devito && \
    cd devito && \
    /venv/bin/pip3.9 install .

RUN git clone https://github.com/Sohl-Dickstein/Sum-of-Functions-Optimizer && \
    cd Sum-of-Functions-Optimizer

ENV PYTHONPATH=/Sum-of-Functions-Optimizer
