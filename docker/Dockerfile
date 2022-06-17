FROM ubuntu:focal

# Tell debconf to not ask questions
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    wget \
    curl \
    gnupg2 \
    lsb-release \
    python3-dev \
    python3-pip \
    sudo

# Change debconf back
RUN echo 'debconf debconf/frontend select Dialog' | debconf-set-selections

# Clean copy the source files
COPY ./entrypoint.sh /opt/ude_gym/entrypoint.sh
COPY ./Notebooks /Notebooks

ENV HOME=/root \
    TERM=xterm \
    DEBIAN_FRONTEND=noninteractive

# Install GUI and some of goodies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        dbus-x11 \
        git \
        x11-xserver-utils \
        xvfb \
        xfce4 \
        xfce4-goodies \
        mesa-utils \
        wget \
        xubuntu-icon-theme \
        less \
        systemd-sysv \
        whoopsie \
        vim \
        vlc \
        unzip

# Install gym dependencies
RUN apt-get install -y \
    python-numpy \
    python-dev \
    cmake \
    zlib1g-dev \
    libjpeg-dev \
    xvfb \
    xorg-dev \
    python-opengl \
    libboost-all-dev \
    libsdl2-dev \
    libosmesa6-dev \
    libglew-dev \
    patchelf \
    swig


# Install Mujoco
RUN mkdir -p /root/.mujoco
RUN wget https://www.roboti.us/download/mjpro150_linux.zip && unzip mjpro150_linux.zip \
    && cp -R ./mjpro150 /root/.mujoco/
RUN curl "https://www.roboti.us/file/mjkey.txt" -o "mjkey.txt" && \
    cat mjkey.txt > /root/.mujoco/mjkey.txt
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/root/.mujoco/mjpro150/bin"

# Install our pip dependencies
RUN pip3 install setuptools \
    lockfile \
    "numpy>=1.19.5" \
    pyarrow==3.0.0 \
    "grpcio>=1.35.0" \
    "gym>=0.18.0,<0.22.0" \
    "mujoco_py>=1.50.1.68" \
    Pillow==8.4.0 \
    "protobuf<=3.20.1" \
    "mypy-protobuf>=2.2" \
    "flake8>=3.5,<4.0.0" \
    pytest-flake8==1.0.7 \
    pytest-pep257==0.0.5 \
    pytest-timeout==1.4.2 \
    boto3==1.18.2 \
    ude_gym_bridge==0.1.4 \
    ude==0.1.5

# By default, run deepracer nodes.
CMD /opt/ude_gym/entrypoint.sh

# Expose both RDP port and UDE Server port
EXPOSE 3389 3003
