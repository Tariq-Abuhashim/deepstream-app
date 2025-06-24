#!/bin/bash
DS_PATH="/opt/nvidia/deepstream/deepstream-6.0"
APP_NAME=deepstream_detr_app

function build() {
  echo "[INFO] Building..."
  mkdir -p build && cd build
  cmake ..
  make -j$(nproc)
  cd ..
}

function run() {
  INPUT=${1:-"file:///path/to/your/video.mp4"}
  echo "[INFO] Running on $INPUT"

  GST_PLUGIN_PATH=$DS_PATH/lib \
  LD_LIBRARY_PATH=$DS_PATH/lib:$LD_LIBRARY_PATH \
  ./build/$APP_NAME $INPUT
}

if [ "$1" == "--build" ]; then
  build
elif [ "$1" == "--run" ]; then
  run "$2"
else
  echo "Usage: $0 --build | --run <uri>"
fi

