# https://hub.docker.com/r/nvidia/cuda/
# https://gitlab.com/nvidia/container-images/cuda/tree/ubuntu18.04

FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
  libpng-dev libjpeg-dev python3-opencv ca-certificates \
  python3-dev build-essential pkg-config git curl wget automake libtool && \
  rm -rf /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
  python3 get-pip.py && \
  rm get-pip.py

# Python dependencies
RUN pip3 --no-cache-dir install \
    Pillow \
    pyyaml \
    tqdm

# Install project specific dependencies
RUN pip3 --no-cache-dir install \
    Keras==2.3.1 \
    tensorflow-gpu==1.13.1 \
    cmake

RUN pip3 --no-cache-dir install \
    dlib \
    imutils \
    scipy \
    Pandas \
    tqdm \
    h5py \
    tables

WORKDIR /home

