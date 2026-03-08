/*
   DeepStream Object Detection and Tracking Pipeline
   
   Pipeline:
   uridecodebin → nvstreammux → nvinfer (DETR) → nvtracker → nvvideoconvert → nvdsosd → nveglglessink

   Usage:
   ./build/deepstream-track file:///path/to/video.mp4
   ./build/deepstream-track file:///path/to/video.mp4 --headless
   ./build/deepstream-track file:///home/mrt/dev/window-tracker/deepstream-app/videos/vulcan.mp4 \
   --config=/home/mrt/dev/window-tracker/deepstream-app/models/windows-res101/config_infer_primary_detr.txt
*/

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

#include "types.h"
#include "config.h"
#include "transforms.h"
#include "gst_callbacks.h"
#include "help.h"

// Global configuration
std::atomic<bool> g_stop{false};
ModelConfig g_model_config;
TransformParams g_transform_params;

// Global map to track object histories
std::map<guint64, BBoxHistory> g_object_histories; // <obj_id, obj_history>
std::mutex g_history_mutex;
guint g_current_frame_num;

int main(int argc, char *argv[]) {

	// No arguments at all
    if (argc < 2) {
        //print_usage();
        return 1;
    }
    
	// Defaults
    std::string config     = "config_pgie_detr_pcc5_res101.txt";
    bool headless = false;
    bool no_stdout = false;

    std::string config_override;
	std::vector<std::string> positional;
    
    // Parse arguments
	for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
		if (arg=="-h" || arg=="--help") {
            //print_usage();
            return 0;
        }

		// Flags
		else if (arg=="--headless") { headless = true; }
		else if (arg=="--no-stdout") { no_stdout = true; }
		
		// Named options with =value
		else if (arg.rfind("-c") == 0 || arg.rfind("--config=") == 0){
			config_override = get_value(arg);
		}

		// Positional arguments
		else {
            positional.push_back(arg);
        }
	}
	
	if (positional.size() < 1) {
        std::cerr << "ERROR: missing <uri>\n";
        //print_usage();
        return 1;
    }
    
    // Assign positional arguments
    std::string uri      = positional[0];  // MP4

	// Override config
	if (!config_override.empty()) {
		config = config_override;
	}

    // Validate files
    if (uri.rfind("file://", 0) == 0) {
        std::string path = uri.substr(7);
        if (!file_exists(path)) {
            std::cerr << "URI file not found: " << path << "\n";
            return 1;
        }
    }
    if (!file_exists(config)) {
        std::cerr << "Config file not found: " << config << "\n";
        return 1;
    }
    
	std::cerr << "[TRACKER]"      << "\n";
    std::cerr << "  uri:        " << uri << "\n";
    std::cerr << "  config:     " << config << "\n";
    std::cerr << "  headless:   " << (headless ? "true" : "false") << "\n";
    std::cerr << "  no-stdout:    " << (no_stdout ? "true" : "false") << "\n";
	//std::cerr << "  Using resolution: " << height << "x" << width << "\n";

	// Load model configuration
	std::cerr << "[INFO] Loading model configuration...\n";
	if (!load_model_config(config, g_model_config)) {
        std::cerr << "[ERROR] Failed to load model configuration\n";
        return 1;
    }
	g_model_config.print();
	
    // Detect video dimensions
    int video_width = 0, video_height = 0;
    if (get_video_dimensions(uri, video_width, video_height)) {
        std::cerr << "[INFO] Detected video dimensions: " 
                  << video_width << "x" << video_height << "\n";
    } else {
        std::cerr << "[INFO] Could not detect video dimensions, using model input size\n";
        video_width = g_model_config.infer_width;
        video_height = g_model_config.infer_height;
    }
    
    // Calculate coordinate transformation parameters
    if (g_model_config.maintain_aspect_ratio) {
        calculate_letterbox_params(video_width, video_height,
                                   g_model_config.infer_width, 
                                   g_model_config.infer_height,
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
    
    /* Check if tracker library exists */
	const char* tracker_lib = "/opt/nvidia/deepstream/deepstream-6.0/lib/libnvds_nvmultiobjecttracker.so";
	std::ifstream lib_file(tracker_lib);
	if (!lib_file.good()) {
		std::cerr << "Tracker library not found: " << tracker_lib << std::endl;
		std::cerr << "Available tracker libraries:" << std::endl;
		int status = system("ls -la /opt/nvidia/deepstream/deepstream*/lib/libnvds_mot*");
		return -1;
	}

	/* Check if config file exists */
	const char* tracker_config = "/opt/nvidia/deepstream/deepstream-6.0/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml";
	//const char* tracker_config = "/opt/nvidia/deepstream/deepstream-6.0/samples/configs/deepstream-app/config_tracker_IOU.yml";
	std::ifstream conf_file(tracker_config);
	if (!conf_file.good()) {
		std::cerr << "Tracker config file not found: " << tracker_config << std::endl;
		std::cerr << "Available config files:" << std::endl;
		int status = system("find /opt/nvidia/deepstream -name '*tracker*.yml' -o -name '*tracker*.txt' 2>/dev/null");
		return -1;
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
    Elements: source, streammux, pgie, tracker, nvvidconv, nvosd and sink
    nvvidconv_pre + capsfilter_pre — convert the raw source to a format streammux accepts (e.g. force NV12, fix resolution)
    streammux — package into batched buffer + attach NvDsBatchMeta
    pgie	  — run inference, outputs NV12 (it's more efficient for inference)
    tracker   — assign IDs, populate tracking metadata, passes frames through unchanged, still NV12
    nvvidconv — converts NV12 → RGBA
    nvosd     — needs RGBA to draw metadata overlays (boxes, labels, etc.) on the frame
    sink      — display or encode
    */
    GstElement *source = gst_element_factory_make("uridecodebin", "src");
    GstElement *nvvidconv_pre = gst_element_factory_make("nvvideoconvert", "nvvidconv_pre");
    GstElement *capsfilter_pre = gst_element_factory_make("capsfilter", "capsfilter_pre");
    GstElement *streammux = gst_element_factory_make("nvstreammux", "streammux");
    GstElement *pgie = gst_element_factory_make("nvinfer", "pgie");
    GstElement *tracker = gst_element_factory_make("nvtracker", "tracker");
    GstElement *nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvidconv");
    
    if (!pipeline || !source || !nvvidconv_pre || !capsfilter_pre || 
        !streammux || !pgie || !tracker || !nvvidconv) {
        g_printerr("Failed to create essential pipeline elements\n");
        return -1;
    }

	/* Configure elements using model config */
	g_object_set(G_OBJECT(source), "uri", uri.c_str(), NULL);

	// Preprocessing: resize to model input dimensions
	// Force NvBufSurface memory type that nvstreammux needs
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

	std::cerr << "about to set pgie config-file-path: " << config << std::endl;
    g_object_set(G_OBJECT(pgie),
		"config-file-path", config.c_str(),
		"batch-size", g_model_config.batch_size,
		NULL);
		
	// Calculate tracker dimensions (must be multiple of 32 for NvDCF)
	int tracker_width = round_to_multiple_of_32(g_model_config.infer_width);
	int tracker_height = round_to_multiple_of_32(g_model_config.infer_height);
	std::cerr << "[INFO] Model dimensions: " << g_model_config.infer_width 
		      << "x" << g_model_config.infer_height << "\n";
	std::cerr << "[INFO] Tracker dimensions (rounded to multiple of 32): " 
		      << tracker_width << "x" << tracker_height << "\n";

	g_object_set(G_OBJECT(tracker),
		"tracker-width", tracker_width,
		"tracker-height", tracker_height,
		"ll-lib-file", tracker_lib,
		"ll-config-file", tracker_config,
		"enable-batch-process", TRUE,
		"gpu-id", 0,
		"enable-past-frame", TRUE,
		NULL);
	g_printerr("Tracker configured with lib: %s\n", tracker_lib);
	g_printerr("Tracker configured with config: %s\n", tracker_config);
    
    
    /* Element factory 
		Display branch or headless
    */
    GstElement *nvosd = NULL;
    GstElement *sink = NULL;
    
	if (headless) {
		sink = gst_element_factory_make("fakesink", "sink");
		if (!sink) {
		    std::cerr << "Failed to create fakesink for headless mode." << std::endl;
		    return -1;
		}
		g_object_set(G_OBJECT(sink), "sync", FALSE, "async", FALSE, NULL);
	} else {
		nvosd = gst_element_factory_make("nvdsosd", "nvosd");
		sink  = gst_element_factory_make("nveglglessink", "sink");
		if (!nvosd || !sink) {
		    std::cerr << "Failed to create display sink elements." << std::endl;
		    return -1;
		}
		g_object_set(G_OBJECT(sink), "sync", FALSE, "async", FALSE, NULL);
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

	/* Add multiple elements into a GstBin at once
    */
    if (headless) {
		gst_bin_add_many(GST_BIN(pipeline),
					source, nvvidconv_pre, capsfilter_pre, streammux,
					pgie, tracker, nvvidconv, sink,
					NULL);
    } else {
		gst_bin_add_many(GST_BIN(pipeline),
					source, nvvidconv_pre, capsfilter_pre, streammux,
					pgie, tracker, nvvidconv, nvosd, sink,
					NULL);
    }

	// Connect dynamic pad handler
    g_signal_connect(source, "pad-added", G_CALLBACK(pad_added_handler), nvvidconv_pre);

    // Link preprocessing chain
    if (!gst_element_link(nvvidconv_pre, capsfilter_pre)) {
        g_printerr("Failed to link nvvidconv_pre -> capsfilter_pre\n");
        return -1;
    }

    // Link capsfilter_pre -> streammux using request pad
    GstPad *filter_src = gst_element_get_static_pad(capsfilter_pre, "src");
    GstPad *mux_sink = gst_element_get_request_pad(streammux, "sink_0");
    if (!filter_src || !mux_sink) {
        g_printerr("Failed to get pads for capsfilter_pre -> streammux\n");
        if (filter_src) gst_object_unref(filter_src);
        if (mux_sink) gst_object_unref(mux_sink);
        return -1;
    }
    if (gst_pad_link(filter_src, mux_sink) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link capsfilter_pre -> streammux\n");
        gst_object_unref(filter_src);
        gst_object_unref(mux_sink);
        return -1;
    }
    gst_object_unref(filter_src);
    gst_object_unref(mux_sink);

	// Link main pipeline
    if (headless) {
        if (!gst_element_link_many(streammux, pgie, tracker, nvvidconv, sink, NULL)) {
            std::cerr << "Elements could not be linked\n";
            return -1;
        }
    } else {
        if (!gst_element_link_many(streammux, pgie, tracker, nvvidconv, nvosd, sink, NULL)) {
            std::cerr << "Elements could not be linked\n";
            return -1;
        }
    }

	// Add probe to print detections (optional)
    if (!no_stdout) {
/*
        GstPad *osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
        gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, 
                         osd_sink_pad_buffer_probe, NULL, NULL);
        gst_object_unref(osd_sink_pad);
*/
/*        GstPad *tracker_src_pad = gst_element_get_static_pad(tracker, "src");
		gst_pad_add_probe(tracker_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
		              print_meta_probe, (gpointer)"nvtracker", NULL);
		gst_object_unref(tracker_src_pad);
*/
    }
    
    // Display filtering probe (modifies visualization)
	if (!headless) {
		GstPad *nvvidconv_src_pad = gst_element_get_static_pad(nvvidconv, "src");
		gst_pad_add_probe(nvvidconv_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
		                  filtered_display_probe, NULL, NULL); // includes Kalman filtering
		gst_object_unref(nvvidconv_src_pad);
	}

	/* set your pipeline to PLAYING state.
    */
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    std::cout << "Running DeepStream pipeline..." << std::endl;

	/* The GstBus is where the pipeline posts messages
    gst_bus_timed_pop_filtered blocks forever (GST_CLOCK_TIME_NONE) until:
    An ERROR message arrives
    An EOS (end-of-stream) message arrives
    Problem: misses other messages
    */
    /* Wait until error or EOS */
    GstBus *bus = gst_element_get_bus(pipeline);
    
	g_signal_connect(bus, "message::error", G_CALLBACK(bus_error_callback), NULL);
	g_signal_connect(bus, "message::warning", G_CALLBACK(bus_warning_callback), NULL);

	// gst_bus_timed_pop_filtered is BLOCKING
    GstMessage *msg = gst_bus_timed_pop_filtered(
    		bus, 
    		GST_CLOCK_TIME_NONE, // = Wait forever until we get ERROR or EOS
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
                break;
            default:
            	/* We should not reach here because we only asked for ERRORs and EOS */
                g_printerr ("Unexpected message received.\n");
                break;
        }
        gst_message_unref(msg);
    }

    // recommened on arm64
	std::cerr << "Exiting cleanly..." << std::endl;
	std::quick_exit(0);
	
	// recommened on aarch64
	// Flush all output before exit
	/*
    std::cerr << "Exiting cleanly..." << std::endl;
    std::cerr.flush();
    std::cerr.flush();
    
    // Use POSIX _exit - most reliable on embedded Linux
    _exit(EXIT_SUCCESS);
    */

    return 0;
}

