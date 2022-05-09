FROM nvidia/cudagl:11.4.2-devel-ubuntu20.04
LABEL AUTHOR="Chanwoo Lee" EMAIL="leechanwoo25@gmail.com"

# Ignore user-interative warning while installing some packages
ENV DEBIAN_FRONTEND=noninteractive

RUN export os_arch="ubuntu2004/x86_64" \
&& rm /etc/apt/sources.list.d/cuda.list \
&& rm /etc/apt/sources.list.d/nvidia-ml.list \
&& apt-key del 7fa2af80 \
&& apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${os_arch}/3bf863cc.pub \
&& apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/${os_arch}/7fa2af80.pub

# Install basic packages without cmake
RUN apt-get update && apt-get upgrade -y \
&& apt-get install -y --no-install-recommends \
  ssh wget curl git unzip build-essential ninja-build git vim \
  lsb-release ca-certificates mesa-utils \
&& apt -y autoremove && apt clean \
&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install cmake & python3
RUN apt-get update \
&& apt-get install -y --no-install-recommends python3-dev python3-pip \
&& apt -y autoremove && apt clean \
&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set up denoise project
RUN git config --global user.email "leechanwoo25@gmail.com" \
&& git config --global user.name "Chanwoo Lee" \
&& pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html \
&& pip3 install datetime termcolor scipy keras-ncp pytorch-lightning

WORKDIR /root
