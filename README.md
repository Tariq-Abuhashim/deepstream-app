# Minimal working DeepStream application to detect and track objects.

## Objectives
1- A working DeepStream app in C++ using a DETR object detector.  
2- Support for object tracking and metadata counting.  
This is a simplified object tracker that is using IoU strategy to associate and match bounding boxes.

## Build
```
mkdir build && cd build
cmake ..
make
```

## Expert and serialise the model
This example is using DETR model `facebook/detr-resnet-50`.  
```
python3 export_detr_onnx.py
trtexec --onnx=detr.onnx --saveEngine=detr.engine --fp16

```

## Run
```
export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:/home/mrt/src/demo/deepstream_detr_app/build:$LD_LIBRARY_PATH
GST_DEBUG=3 ./build/deepstream_detr_app "file://$PWD/videos/palace.mp4"
```

## Next steps
- Example using python API
- Publishing meta data
- ByteTrack integration
- Jetson Orin testing


## Reference:  
https://missionsystems.atlassian.net/wiki/x/BIBysg
