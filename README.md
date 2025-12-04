# Minimal working DeepStream application to detect and track objects.

## Objectives

1- A working DeepStream app in C++ using a DETR object detector.  
2- Support for object tracking and metadata counting.  
This is a simplified object tracker that is using IoU strategy to associate and match bounding boxes.

## Requirements

```
$ deepstream-app --version-all
deepstream-app version 6.0.0
DeepStreamSDK 6.0.0
CUDA Driver Version: 11.4
CUDA Runtime Version: 11.4
TensorRT Version: 8.6
cuDNN Version: 8.2
libNVWarp360 Version: 2.0.1d3
```

## Folder structure:
   
	deepstream-app/
	├── build
	├── build_and_run.sh
	├── build_model.py
	├── CMakeLists.txt
	├── config_infer_primary_detr.txt
	├── consumer.py
	├── export_detr_onnx.py
	├── HELP.txt
	├── image-to-mp4.py
	├── includes
	├── labels_coco.txt
	├── nvdsinfer_customparser_detr.cpp
	├── orbslam.cpp
	├── README.md
	├── serialise_engine.py
	├── track.cpp
	├── videos
	└── weights

## Build

```
$ mkdir build && cd build
$ cmake .. && make
```

## Export and serialise the model

This example is using DETR model `facebook/detr-resnet-50`.  
```
$ python3 export_detr_onnx.py
$ python3 -m onnxsim detr.onnx simplified_detr.onnx --no-large-tensor
$ trtexec --onnx=simplified_detr.onnx --saveEngine=detr.engine --fp16

```

## Pipeline explained:

	uridecodebin                # Reads and decodes your video (any codec)
	 → nvstreammux              # DeepStream batcher input
	 → nvinfer (pgie)           # Object detection (your DETR model)
	 → nvtracker                # Object tracker
	 → nvvideoconvert           # Color/format conversion
	 → tee                      # Split stream into two parallel branches
		 ↘ queue_osd → nvvideoconvert_osd → nvdsosd → sink (for display)
		 ↘ queue_app → nvvideoconvert_app → capsfilter_app → appsink (for slam)	


## Run a minimal pipeline that just displays the file:

```
	gst-launch-1.0 filesrc location=/home/mrt/dev/window-tracker/deepstream-app/videos/flight.mp4 ! qtdemux ! decodebin ! nvvideoconvert ! nveglglessink
```

## Run the complete pipeline:

```
	./build/deepstream_orbslam <uri> <ORBvoc.txt> <settings.yaml>

	// Example using kitti
	./build/deepstream-orbslam \
	file:///home/mrt/dev/window-tracker/deepstream-app/videos/drive.mp4 \
	../ORB_SLAM3/Vocabulary/ORBvoc.txt \
	/media/mrt/Whale/data/kitti/07/KITTI04-12.yaml \
	--headless
	
	// Example using vulcan
	./build/deepstream-orbslam \
	file:///home/mrt/dev/window-tracker/deepstream-app/videos/vulcan.mp4 \
	../ORB_SLAM3/Vocabulary/ORBvoc.txt \
	../ORB_SLAM3/Examples/Monocular/vulcan.yaml \
	--headless

	// Print tracks out
	./build/deepstream-orbslam \
	file:///home/mrt/dev/window-tracker/deepstream-app/videos/vulcan.mp4 \
	../ORB_SLAM3/Vocabulary/ORBvoc.txt \
	../ORB_SLAM3/Examples/Monocular/vulcan.yaml \
	--headless | python3 consumer.py
	
	// Redirect tracks to a file
	./build/deepstream-orbslam \
	file:///home/mrt/dev/window-tracker/deepstream-app/videos/vulcan.mp4 \
	../ORB_SLAM3/Vocabulary/ORBvoc.txt \
	../ORB_SLAM3/Examples/Monocular/vulcan.yaml \
	--headless > tracks.txt
```

## Video dimensions and the GPU:

	Note: Convert your input video to a smaller resolution before running DeepStream
	This ensures even NVDEC can open it safely:
	ffmpeg -i 2024-06-28-03-47-19-uotf-orbit-16-down.mp4 -vf scale=1280:720 -c:v libx264 -crf 20 flight.mp4

## Change Logs:

	June, 19, 2025, initially implemented as main.cpp
	August, 26, 2025, initial orbslam updates
	October, 07, 2025, working orbslam handler (not tested with orbslam)
