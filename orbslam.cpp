
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cstdio>
#include <queue>
#include <deque>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <string>
#include <vector>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <nvdsmeta.h>
#include <gstnvdsmeta.h>
#include <nvds_meta.h>  // Core metadata definitions
#include <nvds_infer.h>
#include <nvdsinfer_custom_impl.h>
#include <nvds_tracker_meta.h>
#include "nvbufsurftransform.h"

#include "nvbufsurface.h"
#include "gstnvdsmeta.h"

#include <opencv2/opencv.hpp>

#include <atomic>
std::atomic<bool> g_stop{false};

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define EIGEN_DONT_ALIGN_STATICALLY
#include "System.h"

#include "help.h"
#include "transforms.h"
#include "types.h"
#include "config.h"
#include "gst_callbacks.h"
#include "nvbuf.h"

// Compile using -DNDEBUG to enable debug messages
#ifdef NDEBUG
    #define CERR std::cerr
#else
    #define CERR if (false) std::cerr
#endif

// define global frame queue from appsink → SLAM worker
std::deque<FramePacket> g_queue; // image + detections
std::mutex g_mutex;
std::condition_variable g_cond;

const size_t MAX_QUEUE = 50; // tune down from 1283

// Global model config and transform params
ModelConfig g_model_config;
TransformParams g_transform_params;

/* Appsink callback: grab frame + NvDs metadata
- Appsink is a sink plugin that supports many different methods for making the 
	application get a handle on the GStreamer data in a pipeline.
- Appsink provides external API functions.
- The normal way of retrieving samples from appsink is by using the 
	"gst_app_sink_pull_sample" and "gst_app_sink_pull_preroll" methods.
- Appsink will internally use a queue to collect buffers from the streaming thread.
- If the application is not pulling samples fast enough, the queue will consume 
	a lot of memory over time. Properties to manage include: 
		"max-buffers", "max-time", "max-bytes" and "leaky-type".
- Why appsink (not the probe) for objects? 
	With tee, the same buffer + metadata reaches the appsink branch. 
	Reading NvDsBatchMeta in the appsink callback keeps frames+objects naturally in sync.
- Reference:
	https://gstreamer.freedesktop.org/documentation/applib/gstappsink.html?gi-language=c
*/
// Thread-safe queue to hand frames to ORB-SLAM3
static GstFlowReturn orbslam_handler(GstElement *sink, gpointer user_data) {
    static int frame_num = 0;
    frame_num++;
    
    CERR << "[APPSINK] ========== Handler called, frame " << frame_num << " ==========\n";
    
    GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    if (!sample) {
        CERR << "[APPSINK] no sample\n";
        return GST_FLOW_OK;
    }
    CERR << "[APPSINK] Got sample\n";

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
        gst_sample_unref(sample);
        CERR << "[APPSINK] no buffer\n";
        return GST_FLOW_OK;
    }
    CERR << "[APPSINK] Got buffer\n";
    
    // Get caps info
    GstCaps *caps = gst_sample_get_caps(sample);
    GstStructure *s = caps ? gst_caps_get_structure(caps, 0) : nullptr;
    const gchar *format = s ? gst_structure_get_string(s, "format") : nullptr;
    int width = 0, height = 0;
    if (s) {
        gst_structure_get_int(s, "width", &width);
        gst_structure_get_int(s, "height", &height);
    }
    CERR << "[APPSINK] new-sample fmt=" << (format?format:"(null)") 
              << " " << width << "x" << height << std::endl;

    // Extract NvBufSurface
    CERR << "[APPSINK] Extracting NvBufSurface...\n";
    NvBufSurface *surf = nullptr;
    /*
    struct NvBufSurface {
		uint32_t numFilled;           // How many frames in this batch
		NvBufSurfaceParams *surfaceList;  // Array of surface descriptors
		// ... other fields
	};
	*/
    
    /* 1. Initial State - 64 Byte Buffer
    */
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        g_printerr("[APPSINK] Failed to map buffer\n");
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }
	CERR << "[APPSINK] Buffer mapped, size: " << map.size << " bytes\n";
        
    /* 2. Surface Contains Pointers to GPU Memory
    */
    surf = (NvBufSurface*)map.data;
	gst_buffer_unmap(buffer, &map);
	CERR << "[APPSINK] Buffer unmapped\n";
	
	if (!surf || surf->numFilled == 0) {
        g_printerr("[APPSINK] Invalid NvBufSurface\n");
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

    CERR << "[APPSINK] Got NvBufSurface, numFilled=" << surf->numFilled << std::endl;
    CERR << "[APPSINK] Surface info: memType=" << surf->memType
              << " colorFormat=" << surf->surfaceList[0].colorFormat
              << " width=" << surf->surfaceList[0].width
              << " height=" << surf->surfaceList[0].height << std::endl;
    
    /* Surface parameters
    NvBufSurfaceParams *params = &surf->surfaceList[0];
    At this point, dataPtr points to GPU memory - your CPU can't read it directly! 
	 - width: 1384
	 - height: 512
	 - colorFormat: 33 (NV12)
	 - pitch: 1536 (row stride in bytes)
	 - dataPtr: 0x7f3766400000 (GPU memory address!)
	*/
    
    
    
    // Check if this is NVMM (GPU) memory
    // memType values: 0=default, 1=pinned, 2=device, 3=unified, 4=surface_array
    bool is_gpu_memory = (surf->memType == NVBUF_MEM_CUDA_DEVICE ||
                          surf->memType == NVBUF_MEM_DEFAULT ||
                          surf->memType == NVBUF_MEM_SURFACE_ARRAY ); // ||
                          // surf->surfaceList[0].mappedAddr.addr[0] == nullptr);
    
    /* 3. Map GPU Memory to CPU
    After mapping, params->mappedAddr.addr[0] now points to CPU-accessible memory!
    */
    CERR << "[APPSINK] Mapping surface to CPU...\n";
    CERR << "memType=" << surf->memType
          << " mappedAddr=" << surf->surfaceList[0].mappedAddr.addr[0]
          << " dataPtr=" << surf->surfaceList[0].dataPtr << std::endl;

    // Create OpenCV Mat
   	cv::Mat image_final;  // This will be our one and only owned copy
   	
	// ALWAYS use the copy function - it handles both GPU and CPU memory
	CERR << "[APPSINK] Copying surface to CPU...\n";
	
	//if (!copyNvToCpuAndMakeBGR(surf, image_final)) {
	if (!copyNvToCpuAndMakeGRAY(surf, image_final)) {
		g_printerr("[APPSINK] Failed to copy to CPU\n");
		gst_sample_unref(sample);
		return GST_FLOW_OK;
	}
    
    if (image_final.empty()) {
        g_printerr("[APPSINK] Failed to create BGR image\n");
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }
    CERR << "[APPSINK] BGR image created: " << image_final.cols 
    											<< "x" << image_final.rows << std::endl;
    
    // 6. Extract metadata
    CERR << "[APPSINK] Extracting metadata...\n";
    std::vector<DetectedObject> objs;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (batch_meta) {
        for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
            auto *fmeta = (NvDsFrameMeta*)l_frame->data;
            for (NvDsMetaList *l_obj = fmeta->obj_meta_list; l_obj; l_obj = l_obj->next) {
                auto *ometa = (NvDsObjectMeta *)l_obj->data;
                DetectedObject o;
                o.id = ometa->object_id;
                o.class_id = ometa->class_id;
                o.confidence = ometa->confidence;
                o.left = ometa->rect_params.left;
                o.top = ometa->rect_params.top;
                o.width = ometa->rect_params.width;
                o.height = ometa->rect_params.height;
                
                // Transform coordinates back to original video size
                transform_detection(o, g_transform_params);
                
                objs.push_back(o);
            }
            // NOTE: If we batch streams, we may want to key by fmeta->source_id
            // and/or ensure we send only the packet that matches this appsink branch.
            break; // batch-size=1 in config → take the first frame
            // if batch-size>1 or use multiple sources, then we should iterate frames 
            // and route by NvDsFrameMeta::source_id
        }
    }
    CERR << "[APPSINK] Metadata extracted, objects: " << objs.size() << "\n";
    
    // Queue the frame
    CERR << "[APPSINK] Queueing frame...\n";
    //double ts = (double)cv::getTickCount() / cv::getTickFrequency();
    // Get actual frame timestamp from GStreamer buffer
	guint64 pts = GST_BUFFER_PTS(buffer);  // nanoseconds
	double ts = (double)pts / 1e9;  // convert to seconds
    
    {
        std::lock_guard<std::mutex> lk(g_mutex);
        if (g_queue.size() >= MAX_QUEUE) {
        	g_queue.pop_front();
        }
        cv::Mat owned = image_final.clone();
        g_queue.push_back(FramePacket{ owned, ts, std::move(objs) });
        CERR << "[APPSINK] Frame queued, queue size: " << g_queue.size() << "\n";
    }
    g_cond.notify_one();
    
    CERR << "[APPSINK] Unreferencing sample...\n";
    gst_sample_unref(sample);
    CERR << "[APPSINK] Handler complete, returning GST_FLOW_OK\n";
    return GST_FLOW_OK;
}

/*
Usage:
./build/deepstream-orbslam \
file:///home/mrt/dev/window-tracker/deepstream-app/videos/vulcan.mp4 \
../ORB_SLAM3/Examples/Monocular/vulcan.yaml \
--config=/home/mrt/dev/window-tracker/deepstream-app/models/windows-res101/config_infer_primary_detr.txt \
--vocabulary=ORBvoc.txt
*/

int main(int argc, char *argv[]) {

	// No arguments at all
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
	// Defaults
	std::string vocabulary = "ORBvoc.txt";
    std::string config     = "config_pgie_detr_pcc5_res101.txt";
    bool headless = false;
    bool orbslam = true;
    bool no_stdout = false;

	std::string vocabulary_override;
    std::string config_override;
	std::vector<std::string> positional;
    
    // Parse arguments
	for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
		if (arg=="-h" || arg=="--help") {
            print_usage();
            return 0;
        }

		// Flags
		else if (arg=="--headless") { headless  = true; }
		else if (arg=="--no-stdout") { no_stdout = true; }
		
		// Named options with =value
		else if (arg.rfind("--vocabulary=") == 0) {
			vocabulary_override = get_value(arg);
		}
		else if (arg.rfind("-c") == 0 || arg.rfind("--config=") == 0){
			config_override = get_value(arg);
		}

		// Positional arguments
		else {
            positional.push_back(arg);
        }
	}
	
	if (positional.size() < 2) {
        CERR << "ERROR: missing <uri> <settings>\n";
        print_usage();
        return 1;
    }
    
    // Assign positional arguments
    std::string uri      = positional[0];  // MP4
    std::string settings = positional[1];  // YAML

	// Override config
	if (!config_override.empty()) {
		config = config_override;
	}

	// Override vocabulary
	if (!vocabulary_override.empty()) {
		vocabulary = vocabulary_override;
	}

    // Validate files
    if (uri.rfind("file://", 0) == 0) {
        std::string path = uri.substr(7);
        if (!file_exists(path)) {
            CERR << "URI file not found: " << path << "\n";
            return 1;
        }
    }
    if (!file_exists(vocabulary)) {
        CERR << "Vocabulary file not found: " << vocabulary << "\n";
        return 1;
    }
    if (!file_exists(settings)) {
        CERR << "Settings file not found: " << settings << "\n";
        return 1;
    }
    if (!file_exists(config)) {
        CERR << "Config file not found: " << config << "\n";
        return 1;
    }
    
	CERR << "[ORBSLAM]"      << "\n";
    CERR << "  uri:        " << uri << "\n";
    CERR << "  settings:   " << settings << "\n";
    CERR << "  vocabulary: " << vocabulary << "\n";
    CERR << "  config:     " << config << "\n";
    CERR << "  headless:   " << (headless ? "true" : "false") << "\n";
    CERR << "  stdout?     " << (no_stdout ? "NO" : "YES") << " mode.\n";
	//CERR << "  Using resolution: " << height << "x" << width << "\n";

	// Load model configuration
	CERR << "[INFO] Loading model configuration...\n";
	if (!load_model_config(config, g_model_config)) {
        CERR << "[ERROR] Failed to load model configuration\n";
        return 1;
    }

	g_model_config.print();
	
    // Detect video dimensions
    int video_width = 0, video_height = 0;
    if (get_video_dimensions(uri, video_width, video_height)) {
        CERR << "[INFO] Detected video dimensions: " 
                  << video_width << "x" << video_height << "\n";
    } else {
        CERR << "[INFO] Could not detect video dimensions, using model input size\n";
        video_width = g_model_config.infer_width;
        video_height = g_model_config.infer_height;
    }
    
    // Calculate coordinate transformation parameters
    if (g_model_config.maintain_aspect_ratio) {
        calculate_letterbox_params(video_width, video_height,
                                   g_model_config.infer_width, g_model_config.infer_height,
                                   g_transform_params);
    } else {
        // Simple stretch - no transformation needed
        g_transform_params.scale_x = (float)video_width / g_model_config.infer_width;
        g_transform_params.scale_y = (float)video_height / g_model_config.infer_height;
        g_transform_params.offset_x = 0;
        g_transform_params.offset_y = 0;
        g_transform_params.original_w = video_width;
        g_transform_params.original_h = video_height;
        g_transform_params.model_w = g_model_config.infer_width;
        g_transform_params.model_h = g_model_config.infer_height;
    }
    
    // Scale camera intrinsics to model dimensions
    cv::FileStorage fs(settings, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        CERR << "[ERROR] Could not open settings file: " << settings << "\n";
        return -1;
    }
    
    // Check if nodes exist before reading
	cv::FileNode fn_fx = fs["Camera1.fx"];
	cv::FileNode fn_fy = fs["Camera1.fy"];
	cv::FileNode fn_cx = fs["Camera1.cx"];
	cv::FileNode fn_cy = fs["Camera1.cy"];
	cv::FileNode fn_width = fs["Camera.width"];
	cv::FileNode fn_height = fs["Camera.height"];

	if (fn_fx.empty() || fn_fy.empty() || fn_cx.empty() || fn_cy.empty()) {
		CERR << "[ERROR] Camera parameters not found in settings file\n";
		return -1;
	}

	float orig_fx = (float)fn_fx;
	float orig_fy = (float)fn_fy;
	float orig_cx = (float)fn_cx;
	float orig_cy = (float)fn_cy;
	int cam_width = (int)fn_width;
	int cam_height = (int)fn_height;

	CERR << "[DEBUG] Read camera params: fx=" << orig_fx << " fy=" << orig_fy 
		      << " cx=" << orig_cx << " cy=" << orig_cy << "\n";

	fs.release();
    
    // Calculate scaled intrinsics
    ScaledIntrinsics scaled = scale_intrinsics(
        orig_fx, orig_fy, orig_cx, orig_cy,
        cam_width, cam_height,
        g_model_config.infer_width, g_model_config.infer_height
    );
    
    // Create temporary settings file with scaled intrinsics
    std::string temp_settings = "/tmp/orbslam_scaled_settings.yaml";
    std::ofstream temp_fs(temp_settings);
    
    // Copy original settings and override camera parameters
    std::ifstream orig_file(settings);
    std::string line;
    while (std::getline(orig_file, line)) {
        // Skip original camera parameter lines
        if (line.find("Camera.fx:") != std::string::npos) {
            temp_fs << "Camera.fx: " << scaled.fx << "\n";
        } else if (line.find("Camera.fy:") != std::string::npos) {
            temp_fs << "Camera.fy: " << scaled.fy << "\n";
        } else if (line.find("Camera.cx:") != std::string::npos) {
            temp_fs << "Camera.cx: " << scaled.cx << "\n";
        } else if (line.find("Camera.cy:") != std::string::npos) {
            temp_fs << "Camera.cy: " << scaled.cy << "\n";
        } else if (line.find("Camera.width:") != std::string::npos) {
            temp_fs << "Camera.width: " << scaled.width << "\n";
        } else if (line.find("Camera.height:") != std::string::npos) {
            temp_fs << "Camera.height: " << scaled.height << "\n";
        } else {
            temp_fs << line << "\n";
        }
    }
    orig_file.close();
    temp_fs.close();
    
    CERR << "[INFO] Created temporary settings file: " << temp_settings << "\n";
    //CERR << "[INFO] Initialise ORBSLAM" << "\n";
    //ORB_SLAM3::System SLAM(vocabulary.c_str(), temp_settings.c_str(), ORB_SLAM3::System::MONOCULAR, !headless);
    
    /* Check if tracker library exists
    */
	std::string tracker_lib = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so";
	std::ifstream lib_file(tracker_lib);
	if (!lib_file.good()) {
		CERR << "Tracker library not found: " << tracker_lib << std::endl;
		CERR << "Available tracker libraries:" << std::endl;
		int status = system("ls -la /opt/nvidia/deepstream/deepstream/lib/libnvds_mot*");
	}

	/* Check if config file exists
    */
	std::string config_file = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml";
	std::ifstream conf_file(config_file);
	if (!conf_file.good()) {
		CERR << "Tracker config file not found: " << config_file << std::endl;
		CERR << "Available config files:" << std::endl;
		int status = system("find /opt/nvidia/deepstream -name '*tracker*.yml' -o -name '*tracker*.txt' 2>/dev/null");
	}
	
	/* Initializes the GStreamer library */
    gst_init(nullptr, nullptr);
    
    /* Initialize CUDA */
    cudaError_t cuda_err = cudaSetDevice(0);
    if (cuda_err != cudaSuccess) {
        g_printerr("[ERROR] Failed to set CUDA device 0: %s\n", 
                   cudaGetErrorString(cuda_err));
        g_printerr("[INFO] Available GPUs:\n");
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            g_printerr("  GPU %d: %s\n", i, prop.name);
        }
        return -1;
    }
    g_printerr("[INFO] CUDA device 0 initialized successfully\n");
    
    /* Create the empty pipeline */
	GstElement *pipeline  = gst_pipeline_new("deepstream-pipeline");


	/* Element factory 
		Basic object detection and tracking branch
    */
    GstElement *source    = gst_element_factory_make("uridecodebin", "src");
    GstElement *nvvidconv_pre = gst_element_factory_make("nvvideoconvert", "nvvidconv_pre");
    GstElement *capsfilter_pre = gst_element_factory_make("capsfilter", "capsfilter_pre");
    GstElement *streammux = gst_element_factory_make("nvstreammux", "streammux");
    GstElement *pgie      = gst_element_factory_make("nvinfer", "pgie");
    if (!pgie) {
    	g_printerr("Failed to create pgie element\n");
    	return -1;
	}
    GstElement *tracker   = gst_element_factory_make("nvtracker", "tracker");
    GstElement *nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvidconv");
    
    if (!pipeline || !source || !nvvidconv_pre || !capsfilter_pre || 
        !streammux || !pgie || !tracker || !nvvidconv) {
        g_printerr("Failed to create essential pipeline elements\n");
        return -1;
    }

    /* Configure elements using model config */
	g_object_set(G_OBJECT(source), "uri", uri.c_str(), NULL);

		/* Insert a preprocessing branch between uridecodebin and nvstreammux
		This forces a smaller frame size, for example 1280×720 (within GPU macroblocks limit (8192): 1280×720 → 3600 macroblocks).
		
		uridecodebin
		   → nvvideoconvert_pre
		   → capsfilter_pre (forces 1280×720)
		   → nvstreammux
    */
    GstCaps *pre_caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=NV12, memory:NVBufSurface");
	gst_caps_set_simple(pre_caps,
		"width", G_TYPE_INT, g_model_config.infer_width,
		"height", G_TYPE_INT, g_model_config.infer_height,
		NULL);
    g_object_set(G_OBJECT(capsfilter_pre), "caps", pre_caps, NULL);
    gst_caps_unref(pre_caps);
    
    g_object_set(G_OBJECT(streammux),
        "width", g_model_config.infer_width,
        "height", g_model_config.infer_height,
        "batch-size", g_model_config.batch_size,
        "batched-push-timeout", 40000,
        "input-meta-queue", FALSE,
    	"gpu-id", 0,  // Explicit GPU ID
        NULL);

	CERR << "about to set pgie config-file-path: " << config << std::endl;
    g_object_set(G_OBJECT(pgie),
		"config-file-path", config.c_str(),
		"batch-size", g_model_config.batch_size,
		NULL);

	g_object_set(G_OBJECT(tracker),
		"tracker-width", g_model_config.infer_width,
        "tracker-height", g_model_config.infer_height,
		"ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
		"ll-config-file", "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml",
		"enable-batch-process", TRUE,
		"gpu-id", 0,
		"enable-past-frame", TRUE,
		NULL);
    g_printerr("Tracker element created successfully\n");
    
    
    /* Element factory 
		Display branch or headless
    */
    GstElement *nvosd = NULL;
    GstElement *sink = NULL;
	if (headless) {
		sink = gst_element_factory_make("fakesink", "sink");
		if (!sink) {
		    CERR << "Failed to create fakesink for headless mode." << std::endl;
		    return -1;
		}
		g_object_set(G_OBJECT(sink), "sync", FALSE, "async", FALSE, NULL);
	} else {
		nvosd = gst_element_factory_make("nvdsosd", "nvosd");
		sink  = gst_element_factory_make("nveglglessink", "sink");
		if (!nvosd || !sink) {
		    CERR << "Failed to create display sink elements." << std::endl;
		    return -1;
		}
		g_object_set(G_OBJECT(sink), "sync", FALSE, "async", FALSE, NULL);
	}

    /* Element factory
    	orbslam branch
    */
	GstElement *tee;
	GstElement *queue_osd;
	GstElement *nvvidconv_osd;
	GstElement *queue_app;
	GstElement *nvvidconv_app;
	GstElement *capsfilter_app;
	GstElement *appsink;
		  
    if (orbslam) {
		tee            = gst_element_factory_make("tee", "tee");
		queue_osd      = gst_element_factory_make("queue", "queue_osd");
		nvvidconv_osd  = gst_element_factory_make("nvvideoconvert", "nvvidconv_osd");
		queue_app      = gst_element_factory_make("queue", "queue_app");
		nvvidconv_app  = gst_element_factory_make("nvvideoconvert", "nvvidconv_app");
		capsfilter_app = gst_element_factory_make("capsfilter", "capsfilter_app");
		appsink        = gst_element_factory_make("appsink", "appsink");

		if (!tee || !queue_osd || !nvvidconv_osd || !queue_app || !nvvidconv_app || !capsfilter_app || !appsink) {
		    CERR << "Failed to create an orbslam element." << std::endl;
		    return -1;
		}
		
		g_object_set(G_OBJECT(appsink),
			"emit-signals", TRUE,
			"sync", FALSE,
			"max-buffers", 1, // prevent backlog if SLAM is slower
			"drop", TRUE,
			NULL);     
		g_object_set(G_OBJECT(queue_osd), 
             "max-size-buffers", 5,
             "max-size-time", 1000000000,  // 1 second
             "leaky", 1,  // leak old buffers
             NULL);
		g_object_set(G_OBJECT(queue_app), "leaky", 2, "max-size-buffers", 1, NULL);
		
		GstCaps *caps = gst_caps_from_string("video/x-raw(memory:NVMM)"); //, format=NV12");
		//GstCaps *caps = gst_caps_from_string("video/x-raw, format=RGBA, memory=CPU");
		//GstCaps *caps = gst_caps_from_string("video/x-raw(memory:SystemMemory),format=NV12");
		g_object_set(G_OBJECT(capsfilter_app), "caps", caps, NULL);
		gst_caps_unref(caps);
    }
    
    /* print factories 
	*/
    #define CHECK_ELEM(e) if (!(e)) { g_printerr("Missing element: %s\n", #e); return -1; }
	CHECK_ELEM(pipeline);
	CHECK_ELEM(source);
	CHECK_ELEM(streammux);
	CHECK_ELEM(pgie);
	CHECK_ELEM(tracker);
	CHECK_ELEM(nvvidconv);
	if (!headless) {
		CHECK_ELEM(nvosd);
	}
	if (orbslam) {
		CHECK_ELEM(tee);
		CHECK_ELEM(queue_osd);
		CHECK_ELEM(nvvidconv_osd);
		CHECK_ELEM(sink);
		CHECK_ELEM(queue_app);
		//CHECK_ELEM(nvvidconv_app);
		//CHECK_ELEM(capsfilter_app);
		CHECK_ELEM(appsink);
	}
	
	/* Build the pipeline.
	   Add multiple elements into a GstBin at once.
    */
    if (orbslam) {
		if (headless) {
			gst_bin_add_many(GST_BIN(pipeline),
							source, nvvidconv_pre, capsfilter_pre, streammux,
                         	pgie, tracker, nvvidconv,
                         	tee,
                         	queue_osd, nvvidconv_osd, sink, // no nvosd here
                         	queue_app, nvvidconv_app, capsfilter_app, appsink,
                         	NULL);
		} else {
			gst_bin_add_many(GST_BIN(pipeline),
							source, nvvidconv_pre, capsfilter_pre, streammux,
                         	pgie, tracker, nvvidconv,
                         	tee,
                         	queue_osd, nvvidconv_osd, nvosd, sink,
                         	queue_app, nvvidconv_app, capsfilter_app, appsink,
                        	 NULL);
		}
    } else {
		gst_bin_add_many(GST_BIN(pipeline),
					source, nvvidconv_pre, capsfilter_pre, streammux,
					pgie, tracker, nvvidconv, 
					nvosd, sink,
					NULL);
    }

	/* Attempts to link each element’s src pad to the next element’s sink pad in order
    streammux → pgie → tracker → nvvidconv → nvosd → sink
    Pads must exist already.
    Elements wth static pads, src and sink pads exist after creation.
    Elements with dynamic pads, src and sink pads must be created first. 
    */
    
	/* Hanble dynamic pads
    */
    
    //g_signal_connect(source, "pad-added", G_CALLBACK(pad_added_handler), streammux);
    
    g_signal_connect(source, "pad-added", G_CALLBACK(pad_added_handler), nvvidconv_pre);
    
    // Link the preprocessing chain
	if (!gst_element_link(nvvidconv_pre, capsfilter_pre)) {
		g_printerr("[LINK] Failed to link nvvidconv_pre -> capsfilter_pre\n");
		gst_object_unref(pipeline);
		return -1;
	}
	g_printerr("[LINK] ✓ nvvidconv_pre -> capsfilter_pre linked\n");
	
	// Link capsfilter_pre -> streammux using request pad
	GstPad *filter_src = gst_element_get_static_pad(capsfilter_pre, "src");
	GstPad *mux_sink = gst_element_get_request_pad(streammux, "sink_0");
	if (!filter_src || !mux_sink) {
		g_printerr("[LINK] Failed to get pads for capsfilter_pre -> streammux\n");
		if (filter_src) gst_object_unref(filter_src);
		if (mux_sink) gst_object_unref(mux_sink);
		return -1;
	}

	if (gst_pad_link(filter_src, mux_sink) != GST_PAD_LINK_OK) {
		g_printerr("[LINK] Failed to link capsfilter_pre -> streammux\n");
		gst_object_unref(filter_src);
		gst_object_unref(mux_sink);
		return -1;
	}
	g_printerr("[LINK] ✓ capsfilter_pre -> streammux linked\n");
	gst_object_unref(filter_src);
	gst_object_unref(mux_sink);

    if (orbslam) {
		g_signal_connect(appsink, "new-sample", G_CALLBACK(orbslam_handler), NULL); // Orbslam updates
	}

    //GstPad *mux_sink_pad_ref = NULL;
    GstPad *tee_src_pad_osd = NULL, *tee_src_pad_app = NULL;
    GstPad *queue_osd_sink_pad = NULL, *queue_app_sink_pad = NULL;
 	if (orbslam) {
	
		// Link main path (before tee)
		if (gst_element_link_many(streammux, pgie, tracker, nvvidconv, tee, NULL) != TRUE) {
			g_printerr("Main pipeline elements could not be linked.\n");
			gst_object_unref (pipeline);
			return -1;
		}

		/*
		(after tee) - Request Pads
		A tee element can have multiple output branches, so it does not have one static "src" pad.
		Instead, it has a pad template:	src_%u (request pad)
		We must request a new pad
		GStreamer creates a new pad named src_0, then src_1, etc.
		*/

		// Request a new src pad from tee for OSD branch
		tee_src_pad_osd = gst_element_get_request_pad(tee, "src_%u");
		queue_osd_sink_pad = gst_element_get_static_pad(queue_osd, "sink");
		if (gst_pad_link(tee_src_pad_osd, queue_osd_sink_pad) != GST_PAD_LINK_OK) {
			g_printerr("Tee to OSD branch could not be linked.\n");
			gst_object_unref(queue_osd_sink_pad);
			gst_element_release_request_pad(tee, tee_src_pad_osd);
			gst_object_unref(tee_src_pad_osd);
			tee_src_pad_osd = NULL;
			return -1;
		}

		// Request a new src pad from tee for App branch
		tee_src_pad_app = gst_element_get_request_pad(tee, "src_%u");
		queue_app_sink_pad = gst_element_get_static_pad(queue_app, "sink");
		if (gst_pad_link(tee_src_pad_app, queue_app_sink_pad) != GST_PAD_LINK_OK) {
			g_printerr("Tee to App branch could not be linked.\n");
			gst_object_unref(queue_app_sink_pad);
			gst_element_release_request_pad(tee, tee_src_pad_app);
			gst_object_unref(tee_src_pad_app);
			tee_src_pad_app = NULL;
			return -1;
		}

		// Now link the OSD branch normally
		if (headless) {
			if (gst_element_link_many(queue_osd, nvvidconv_osd, sink, NULL) != TRUE) {
				g_printerr("Headless OSD branch could not be linked.\n");
				gst_object_unref (pipeline);
				return -1;
			}
		} else {
			if (gst_element_link_many(queue_osd, nvvidconv_osd, nvosd, sink, NULL) != TRUE) {
				g_printerr("Display OSD branch could not be linked.\n");
				gst_object_unref (pipeline);
				return -1;
			}
		}


		// And link the App branch normally
		if (gst_element_link_many(queue_app, nvvidconv_app, capsfilter_app, appsink, NULL) != TRUE) {
			g_printerr("App branch could not be linked.\n");
			gst_object_unref (pipeline);
			return -1;
		}
	} else {
		
		// Link main path and display sink
	    if (gst_element_link_many(streammux, pgie, tracker, nvvidconv, nvosd, sink, NULL) != TRUE) {
		    g_printerr("Elements could not be linked.\n");
		    gst_object_unref (pipeline);
		    return -1;
		}
	}


	/* Attach a print prob to nvtracker sink pad */
	if (!no_stdout) {
		GstPad *tracker_src_pad = gst_element_get_static_pad(tracker, "src");
		gst_pad_add_probe(tracker_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
		              print_meta_probe, (gpointer)"nvtracker", NULL);
		gst_object_unref(tracker_src_pad);
	}

		
	/* Start playing */
	GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
	if (ret == GST_STATE_CHANGE_FAILURE) {
		g_printerr("Pipeline failed to start (PGIE init likely failed)\n");
		gst_object_unref(pipeline);
		return -1;
	}
	
	GstState state;
	gst_element_get_state(pipeline, &state, NULL, 5 * GST_SECOND);
	g_printerr("Pipeline state: %d\n", state);
 
    
    /*Orbslam
    */
    CERR << "Running DeepStream + ORB-SLAM3 pipeline..." << std::endl;
    ORB_SLAM3::System SLAM(vocabulary.c_str(), temp_settings.c_str(), ORB_SLAM3::System::MONOCULAR, !headless);
    std::thread orbslamThread([&]() {
    	cv::Mat gray; // keep local copies to avoid repeated allocations
    	double last_ts = 0.0;
		while (true) {

			FramePacket pkt;
			{
				std::unique_lock<std::mutex> lock(g_mutex);
				
				// Wait with timeout instead of indefinitely
		        if (!g_cond.wait_for(lock, std::chrono::milliseconds(100), 
		            []{ return !g_queue.empty() || g_stop.load(); })) {
		            // Timeout - allow viewer to update
		            continue;
		        }
				
				// If stop requested and no frames left -> exit
		        if (g_stop.load() || g_queue.empty()) {
		            CERR << "[ORBSLAM] Stop flag and empty queue -> exiting thread.\n";
		            break;
		        }
            
            	if (!g_queue.empty()) {
					pkt = std::move(g_queue.front());
					g_queue.pop_front();
				} else {
					continue;
				}
			}
			
			cv::Mat frame = pkt.frame;
			
			if (frame.empty()) {
            	CERR << "[ORBSLAM] Received empty frame packet, skipping\n";
            	continue;
        	}
        	
		    if (frame.channels() == 1) {
            	gray = frame;
		    } else if (frame.channels() == 3) {
		        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		    } else if (frame.channels() == 4) {
		        cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
		    } else {
		        CERR << "Unsupported channels: " << frame.channels() << std::endl;
		        continue;  // Changed from return to continue
		    }
		    
		    // Timestamp: pkt.timestamp (seconds). ORB-SLAM expects seconds (double).
            // If you used cv::getTickCount for pkt.timestamp earlier, it's fine.
        	double ts = pkt.timestamp;
		    
		    CERR << "[ORBSLAM] Processing frame: " 
                  << std::fixed << std::setprecision(6) << ts << ", " 
                  << gray.cols << "x" << gray.rows << "x" << gray.channels() << "\n";
        
		    // Validate timestamp
		    if (ts <= last_ts && last_ts > 0) {
		        CERR << "[ORBSLAM] WARNING: Non-increasing timestamp! " 
		                  << "last=" << last_ts << " current=" << ts << "\n";
		    }
		    last_ts = ts;

		    // Track the frame
		    try {
		        SLAM.TrackMonocular(gray, ts);
		    } catch (const std::exception& e) {
		        CERR << "[ORBSLAM] Exception during tracking: " << e.what() << "\n";
		    }
		    
		    // object-level adapter (new implementation)
			// e.g., associate 2D boxes with landmarks, semantic constraints, etc.
			// TrackMonocularWithObjects(SLAM, pkt.frame, pkt.timestamp, pkt.objs);
			
			// sleep small amount to throttle
          	std::this_thread::sleep_for(std::chrono::milliseconds(1));
          	
		} // while loop
	});

	/* The GstBus is where the pipeline posts messages
    gst_bus_timed_pop_filtered blocks forever (GST_CLOCK_TIME_NONE) until:
    An ERROR message arrives
    An EOS (end-of-stream) message arrives
    */
    /* Wait until error or EOS */
    GstBus *bus = gst_element_get_bus(pipeline);
    
	g_signal_connect(bus, "message::error", G_CALLBACK(bus_error_callback), NULL);
	g_signal_connect(bus, "message::warning", G_CALLBACK(bus_warning_callback), NULL);

    GstMessage *msg = gst_bus_timed_pop_filtered(
    		bus, 
    		GST_CLOCK_TIME_NONE,
            (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS)
    );
    
    /* Parse message */
    if (msg != NULL) {
        GError *err;
        gchar *debug_info;
        
        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_ERROR:
				gst_message_parse_error (msg, &err, &debug_info);
				g_printerr ("Error received from element %s: %s\n",
				    GST_OBJECT_NAME (msg->src), err->message);
				g_printerr ("Debugging information: %s\n",
				    debug_info ? debug_info : "none");
				g_clear_error (&err);
				g_free (debug_info);
				break;
            case GST_MESSAGE_EOS:
                g_print ("End-Of-Stream reached.\n");
				// signal to stop orbslam
                //g_stop.store(true);
                //g_cond.notify_all();
                break;
            default:
            	/* We should not reach here because we only asked for ERRORs and EOS */
                g_printerr ("Unexpected message received.\n");
                break;
        }
        gst_message_unref(msg);
    }
    
    CERR << "Stopping pipeline..." << std::endl;
    
    // pause pipeline and signal orbslam thread to exit
    gst_element_set_state(pipeline, GST_STATE_PAUSED); // Pause
	g_stop.store(true);
    g_cond.notify_all();
    
    CERR << "Joining ORBSLAM ..." << std::endl;
    orbslamThread.join();
    
	// Clear queue after thread is done
    CERR << "Clearing frame queue..." << std::endl;
    {
        std::lock_guard<std::mutex> lk(g_mutex);
        g_queue.clear();
    }
    CERR << "✓ Queue cleared" << std::endl;
    
    CERR << "Shutting down ORBSLAM ..." << std::endl;
    if (SLAM.isShutDown() == false) {
    	SLAM.Shutdown();
	}
    CERR << "✓ ORBSLAM shutdown complete" << std::endl;
    
        
    // Save camera trajectory
	// Creating a directory
	CERR << "Saving Map data ..." << std::endl;
	if (mkdir("sparse", 0777) == -1) {
		if (errno != EEXIST) {
			cerr << "Error creating directory: " << strerror(errno) << endl;
    	} else {
        	CERR << "Directory already exists, using it..." << endl;
    	}
	}
	else {
		CERR << "Directory created";
	}
	SLAM.WriteCamerasText("sparse/cameras.txt");
	SLAM.WriteImagesText("sparse/images.txt");
	SLAM.WritePoints3DText("sparse/points3D.txt");
	CERR << "✓ Map data saved" << std::endl;
	
	// stop the pipeline
    //gst_element_set_state(pipeline, GST_STATE_NULL);
	
	// recommened on arm64
	CERR << "Exiting cleanly..." << std::endl;
	std::quick_exit(0);
	
	// recommened on aarch64
	// Flush all output before exit
	/*
    CERR << "Exiting cleanly..." << std::endl;
    CERR.flush();
    CERR.flush();
    
    // Use POSIX _exit - most reliable on embedded Linux
    _exit(EXIT_SUCCESS);
    */
    
    return 0;
}

