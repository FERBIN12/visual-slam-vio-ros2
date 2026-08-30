FROM osrf/ros:humble-desktop

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV LD_LIBRARY_PATH=/usr/local/lib

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    pkg-config \
    libeigen3-dev \
    libopencv-dev \
    libboost-all-dev \
    libglew-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libepoxy-dev \
    libwayland-dev \
    libxkbcommon-dev \
    libpython3.10-dev \
    libpython3-dev \
    python3-numpy \
    python3-opencv \
    python3-pip \
    ros-humble-ros-gz \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-message-filters \
    ros-humble-tf2-ros \
    ros-humble-rviz2 \
    ros-humble-rosbag2-storage-default-plugins \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch v0.9.2 https://github.com/stevenlovegrove/Pangolin.git /tmp/Pangolin \
    && cmake -S /tmp/Pangolin -B /tmp/Pangolin/build -DCMAKE_BUILD_TYPE=Release -DBUILD_PANGOLIN_PYTHON=OFF \
    && cmake --build /tmp/Pangolin/build -j"$(nproc)" \
    && cmake --install /tmp/Pangolin/build \
    && rm -rf /tmp/Pangolin

RUN python3 -m pip install --no-cache-dir \
    evo==1.29.0 matplotlib pyyaml

WORKDIR /workspace
COPY prototype/ /workspace/prototype/

ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["source /opt/ros/humble/setup.bash && exec bash"]
