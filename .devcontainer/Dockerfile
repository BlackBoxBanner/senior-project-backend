# Use a base image with Debian/Ubuntu and install Python 3.10.17 manually
FROM mcr.microsoft.com/devcontainers/base:bullseye

# Install all build tools, libraries (including libgl1) and Python build dependencies
RUN apt-get update && \
    apt-get install -y \
      cmake \
      make \
      g++ \
      wget \
      build-essential \
      zlib1g-dev \
      libglib2.0-0 \
      libncurses5-dev \
      liblzma-dev \
      sqlite3 \
      libsqlite3-dev \
      libgdbm-dev \
      libnss3-dev \
      libssl-dev \
      libreadline-dev \
      libffi-dev \
      curl \
      libbz2-dev \
      libgl1 && \
    curl -O https://www.python.org/ftp/python/3.10.17/Python-3.10.17.tgz && \
    tar -xf Python-3.10.17.tgz && \
    cd Python-3.10.17 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    ln -s /usr/local/bin/python3.10 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.10 /usr/local/bin/pip && \
    cd .. && \
    rm -rf Python-3.10.17 Python-3.10.17.tgz && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Set default user (usually set to vscode)
ARG USERNAME=vscode