# Build: sudo docker build -t hyperlight .
# Run: sudo docker run -v $(pwd):/host --gpus all -it --rm hyperlight

# FROM nvidia/cuda:11.4.0-base-ubuntu20.04
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

# Basic setup
ENV COLMAP_VERSION=3.7
ARG CMAKE_VERSION=3.23.3
ENV PYTHON_VERSION=3.8
ENV OPENCV_VERSION=4.6.0.66
ENV CERES_SOLVER_VERSION=2.1.0


# Add environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV PATH=$PATH:${CUDA_HOME}/bin
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PATH="/usr/bin/cmake/bin:${PATH}"
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
        wget vim pkg-config unzip rsync tmux \
        ninja-build x11-apps \
        # it is added from  instantngp provided dockerfile after this line
        ffmpeg tk-dev libxi-dev libc6-dev libbz2-dev \ 
        libffi-dev libomp-dev libssl-dev \
	    zlib1g-dev libcgal-dev libgdbm-dev \
	    libglew-dev qtbase5-dev checkinstall libmetis-dev \
	    libglfw3-dev libeigen3-dev libgflags-dev \
        libxrandr-dev libopenexr-dev libsqlite3-dev \
	    libxcursor-dev libcgal-qt5-dev libxinerama-dev \
	    libboost-all-dev libfreeimage-dev libncursesw5-dev \
	    libatlas-base-dev libqt5opengl5-dev libgoogle-glog-dev \
	    libsuitesparse-dev libreadline-gplv2-dev \
    && wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh \
        -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && mkdir /usr/bin/cmake \
    && /tmp/cmake-install.sh --skip-license --prefix=/usr/bin/cmake \
    && rm /tmp/cmake-install.sh \
    && apt-get install sudo \
    && apt autoremove -y && apt clean -y && rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /opt

RUN echo "Installing Ceres Solver ver. ${CERES_SOLVER_VERSION}..." \
	&& git clone https://github.com/ceres-solver/ceres-solver \
	&& cd ./ceres-solver \
	&& git checkout ${CERES_SOLVER_VERSION} \
	&& mkdir ./build \
	&& cd ./build \
	&& cmake ../ -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF \
	&& make -j8 \
	&& make install \
    && cd ../.. && rm -r ceres-solver

RUN echo "Installing COLMAP ver. ${COLMAP_VERSION}..." \
	&& git clone https://github.com/colmap/colmap \
	&& cd ./colmap \
	&& git checkout ${COLMAP_VERSION} \
	&& mkdir ./build \
	&& cd ./build \
	&& cmake ../ \
	&& make -j8 \
	&& make install \
	&& colmap -h \
    && cd ../.. && rm -r colmap

# # Create non root user and setup environment.
# RUN useradd -m -d /home/hyperlight -u 1000 hyperlight

# # Switch to new uer and workdir.
# USER 1000:1000
# WORKDIR /home/hyperlight

# # Add local user binary folder to PATH variable.
# ENV PATH="${PATH}:/home/hyperlight/.local/bin"
SHELL ["/bin/bash", "-c"]

# Install Miniconda with given python version
RUN echo "Installing Anaconda with Python ver. ${PYTHON_VERSION}..." \ 
    && curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && /opt/conda/bin/conda install -y python=${PYTHON_VERSION} \
    && /opt/conda/bin/conda clean -ya
ENV PATH=/opt/conda/bin:$PATH
ENV PATH=/root/.local/bin:$PATH 

# Install PyTorch
RUN /opt/conda/bin/python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
        torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# COPY dependencies/nerfstudio /home/hyperlight/nerfstudio
# USER root
# RUN chown -R user:user /home/hyperlight/nerfstudio
# USER 1000:1000

# Specify cuda compute
ARG CUDA_COMPUTE=75
ENV TCNN_CUDA_ARCHITECTURES=${CUDA_COMPUTE}
ENV CMAKE_CUDA_ARCHITECTURES=${CUDA_COMPUTE}

# Install required libraries
RUN /opt/conda/bin/python -m pip --no-cache-dir install git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch \
    && /opt/conda/bin/python -m pip install --upgrade pip setuptools
    
RUN /opt/conda/bin/python -m pip --no-cache-dir install cmake==${CMAKE_VERSION} \
    opencv-python==${OPENCV_VERSION} opencv-contrib-python==${OPENCV_VERSION} \
    aiohttp==3.8.1 aiortc==1.3.2 appdirs>=1.4 av==9.2.0 tyro>=0.3.31 gdown==4.5.1 \
    ninja==1.10.2.3 functorch==0.2.1 h5py>=2.9.0 imageio==2.21.1 ipywidgets>=7.6 \
    jupyterlab==3.3.4 matplotlib==3.5.3 mediapy==1.1.0 msgpack==1.0.4 \
    msgpack_numpy==0.4.8 nerfacc==0.2.1 open3d>=0.16.0 plotly==5.7.0 protobuf==3.20.0 \
    pyngrok==5.1.0 python-socketio==5.7.1 requests rich==12.5.1 tensorboard==2.9.0 \
    u-msgpack-python>=2.4.1 nuscenes-devkit>=1.1.1 wandb>=0.13.3 Pillow==9.3.0 \
    hydra-core hydra-colorlog hydra-optuna-sweeper tqdm \
    pytorch-lightning torchmetrics kornia scipy scikit-image \
    && /opt/conda/bin/conda install -y ffmpeg=4.2.2 mpi4py \
    && /opt/conda/bin/conda clean -ya
	
WORKDIR /

ENV PIP3I="python3 -m pip install  --upgrade "

RUN $PIP3I timm tensorboardX blobfile gpustat torchinfo fairseq==0.10.0 click einops


RUN mkdir -p /hooks

# Added removal of cuda-compat package 
RUN apt remove -y cuda-compat-11-3

# WORKDIR ~
# # Install ninja 
# RUN /opt/conda/bin/python -m pip install ninja

# RUN echo "Cloning threestudio and installing requirements ..." \
# 	&& git clone https://github.com/threestudio-project/threestudio.git \
# 	&& cd ./threestudio \
# 	&& pip install -r requirements.txt
	
