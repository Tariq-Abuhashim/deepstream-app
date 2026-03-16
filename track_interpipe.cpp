/*
   DeepStream Object Detection and Tracking Pipeline with GstInterpipe
   
   This is a refactored version of track.cpp using GstInterpipe for dynamic
   source switching and modular pipeline architecture.
   
   Usage:
   ./build/deepstream-interpipe-track file:///path/to/video.mp4
   ./build/deepstream-interpipe-track file:///path/to/video.mp4 --headless
   ./build/deepstream-interpipe-track file:///path/to/video.mp4 --config=config.txt
   ./build/deepstream-interpipe-track file:///home/mrt/dev/window-tracker/deepstream-app/videos/vulcan.mp4 \
   --config=/home/mrt/dev/window-tracker/deepstream-app/models/windows-res101/config_infer_primary_detr.txt
*/

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstring>
#include <signal.h>
#include <unistd.h>

#include <gst/gst.h>
#include <nvdsmeta.h>
#include <gstnvdsmeta.h>

#include <cuda_runtime.h>

#include <atomic>

#include "types.h"
#include "config.h"
#include "transforms.h"
#include "gst_callbacks.h"
#include "help.h"

#include "InterpipeTracker.h"

// Global configuration
std::atomic<bool> g_stop{false};
ModelConfig g_model_config;
TransformParams g_transform_params;

// Global map to track object histories
std::map<guint64, BBoxHistory> g_object_histories; // <obj_id, obj_history>
std::mutex g_history_mutex;
guint g_current_frame_num;

InterpipeTracker* g_tracker = nullptr;
volatile sig_atomic_t g_interrupted = 0;

GMainLoop* g_main_loop = nullptr;  // Add this global

void signal_handler(int signum) {
    g_interrupted = 1;
    g_print("\n[SIGNAL] Caught interrupt, shutting down...\n");
    if (g_main_loop) {
        g_main_loop_quit(g_main_loop);
    }
}

// Output bus watch
const std::set<std::string> InterpipeTracker::OUTPUT_NAMES = {
    "display", "headless", "recorder", "metadata"
};

int main(int argc, char* argv[]) 
{

	// Install signal handler
	signal(SIGINT, signal_handler);

	// Parse arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <uri> [options]\n";
        std::cerr << "Options:\n";
        std::cerr << "  --headless          Run without display\n";
        std::cerr << "  --config=FILE       Inference config file\n";
        std::cerr << "  --record=FILE       Record output to file\n";
        std::cerr << "  --metadata          Enable metadata extraction\n";
        return 1;
    }

	// Defaults
    std::string config = "config_pgie_detr_pcc5_res101.txt";
    bool headless = false; // PASS !!
    bool no_stdout = false; // PASS !!
    bool record = false; // NOT TESTED !!
    bool extract_metadata = false; // NOT TESTED !!
    std::string record_file;
    std::string uri;

	// Parse command line
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--headless") {
            headless = true;
        } else if (arg == "--no-stdout") {
        	no_stdout = true;
		} else if (arg.rfind("--record=", 0) == 0) {
            record = true;
            record_file = get_value(arg);
        } else if (arg == "--metadata") {
            extract_metadata = true;
        } else if (arg.rfind("--config=", 0) == 0) {
            config = get_value(arg);
        } else {
            uri = arg;
        }
    }

	if (uri.empty()) {
        std::cerr << "ERROR: Missing <uri>\n";
        return 1;
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

	// Print configuration
    std::cout << "\n[TRACKER]\n";
    std::cout << "uri:       " << uri << "\n";
    std::cout << "config:    " << config << "\n";
    std::cout << "mode:      " << (headless ? "headless" : "display") << "\n";
    std::cerr << "no-stdout: " << (no_stdout ? "true" : "false") << "\n";
    if (record) {
        std::cout << "recording: " << record_file << "\n";
    }
    if (extract_metadata) {
        std::cout << "metadata:  enabled\n";
    }
	
    // Initialize CUDA
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
    g_print("[INFO] CUDA device 0 initialized successfully\n");

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

	// Configuration
	const int infer_width = g_model_config.infer_width;
    const int infer_height = g_model_config.infer_height;
    const int tracker_width = round_to_multiple_of_32(g_model_config.infer_width);
    const int tracker_height = round_to_multiple_of_32(g_model_config.infer_height);

	std::cout << "[INFO] Model dimensions: " << infer_width << "x" << infer_height << "\n";
    std::cout << "[INFO] Tracker dimensions: " << tracker_width << "x" << tracker_height << "\n";
    
    // Tracker configuration
    const char* tracker_lib = "/opt/nvidia/deepstream/deepstream-6.0/lib/libnvds_nvmultiobjecttracker.so";
    const char* tracker_config = "/opt/nvidia/deepstream/deepstream-6.0/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml";
    
    // Check tracker files exist
    if (!file_exists(tracker_lib)) {
        std::cerr << "Tracker library not found: " << tracker_lib << "\n";
		std::cerr << "Available tracker libraries:" << std::endl;
		int status = system("ls -la /opt/nvidia/deepstream/deepstream*/lib/libnvds_mot*");
        return -1;
    }
    if (!file_exists(tracker_config)) {
        std::cerr << "Tracker config not found: " << tracker_config << "\n";
		std::cerr << "Available config files:" << std::endl;
		int status = system("find /opt/nvidia/deepstream -name '*tracker*.yml' -o -name '*tracker*.txt' 2>/dev/null");
        return -1;
    }

    // Initialize GStreamer
    //gst_init(&argc, &argv);  // if you want GStreamer to parse its own command-line args
	gst_init(nullptr, nullptr); // we are handling everything ourselves

	// Create InterpipeTracker
    InterpipeTracker tracker;
    g_tracker = &tracker; // For signal handler
    
    tracker.set_config(infer_width, infer_height, 1);
    
    std::cout << "\n=== Creating Pipelines ===\n";

	// 1. Create input pipeline
	GstElement* input = tracker.create_file_input_pipeline(uri, "video_source");
	if (!input) {
        g_printerr("Failed to create input pipeline\n");
        return -1;
    }
    tracker.add_pipeline("input", input);

	// 2. Create processing pipeline
	GstElement* processing = tracker.create_processing_pipeline(
		config,
		"video_source",
		tracker_lib,
		tracker_config,
		tracker_width,
		tracker_height);
	if (!processing) {
		g_printerr("Failed to create processing pipeline\n");
		return -1;
	}
	tracker.add_pipeline("processing", processing);
	
	// Attach print_meta_probe to the tracker's src pad
	GstElement* tracker_elem = gst_bin_get_by_name(GST_BIN(processing), "tracker");
	if (tracker_elem && !no_stdout) {
		GstPad* src_pad = gst_element_get_static_pad(tracker_elem, "src");
		gst_pad_add_probe(src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        	print_meta_probe, (gpointer)"tracker_src", NULL);
        gst_object_unref(src_pad);
    	gst_object_unref(tracker_elem);
	}

	// 3. Create output pipeline(s)
	if (headless) {
		GstElement* output = tracker.create_headless_pipeline();
		if (!output) {
			g_printerr("Failed to create headless output pipeline\n");
			return -1;
		}
		tracker.add_pipeline("headless", output);
	} else {
		GstElement* display = tracker.create_display_pipeline(false);
		if (!display) {
			g_printerr("Failed to create display pipeline\n");
			return -1;
		}
		tracker.add_pipeline("display", display);
		
		// Add filtered display probe
		GstElement* conv_elem = gst_bin_get_by_name(GST_BIN(processing), "nvvidconv");
		if (conv_elem) {
			GstPad* conv_src_pad = gst_element_get_static_pad(conv_elem, "src");
			if (conv_src_pad) {
				gulong probe_id = gst_pad_add_probe(conv_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                            filtered_display_probe, NULL, NULL); // includes Kalman filtering
                gst_object_unref(conv_src_pad);
			} else {
				g_print("✗ Could not get nvvidconv src pad\n");
			}
			gst_object_unref(conv_elem);
		} else {
			g_print("✗ Could not find nvvidconv element in pipeline\n");
		}
	}

	// 4. Optional: Create recorder pipeline
	if (record) {
		GstElement* recorder = tracker.create_recorder_pipeline(record_file);
		if (!recorder) {
			g_printerr("Failed to create recorder pipeline\n");
			return -1;
		}
		tracker.add_pipeline("recorder", recorder);
		
GstElement* rec_src = gst_bin_get_by_name(GST_BIN(recorder), "src");
if (rec_src) {
    GstPad* p = gst_element_get_static_pad(rec_src, "src");
    gst_pad_add_probe(p, GST_PAD_PROBE_TYPE_BUFFER,
        [](GstPad*, GstPadProbeInfo*, gpointer) -> GstPadProbeReturn {
            static int n = 0;
            if (++n == 1) g_printerr("[PROBE] recorder interpipesrc: FIRST buffer\n");
            if (n % 30 == 0) g_printerr("[PROBE] recorder interpipesrc: %d buffers\n", n);
            return GST_PAD_PROBE_OK;
        }, NULL, NULL);
    gst_object_unref(p);
    gst_object_unref(rec_src);
}

g_timeout_add(3000, [](gpointer data) -> gboolean {
    GstElement* rec = static_cast<GstElement*>(data);
    GstState state, pending;
    gst_element_get_state(rec, &state, &pending, 0);
    g_printerr("[RECORDER] state=%s pending=%s\n",
        gst_element_state_get_name(state),
        gst_element_state_get_name(pending));

    // Walk all elements and print their state
    GstIterator* it = gst_bin_iterate_elements(GST_BIN(rec));
    GValue item = G_VALUE_INIT;
    while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
        GstElement* e = GST_ELEMENT(g_value_get_object(&item));
        GstState es, ep;
        gst_element_get_state(e, &es, &ep, 0);
        g_printerr("[RECORDER]   %s: %s (pending: %s)\n",
            GST_ELEMENT_NAME(e),
            gst_element_state_get_name(es),
            gst_element_state_get_name(ep));
        g_value_reset(&item);
    }
    gst_iterator_free(it);
    return G_SOURCE_REMOVE;
}, recorder);

    }

	// 5. Optional: Create metadata extraction pipeline
/*	if (extract_metadata) {
		GstElement* metadata = tracker.create_metadata_pipeline(
			G_CALLBACK(metadata_callback), 
			nullptr
		);
		if (!metadata) {
			g_printerr("Failed to create metadata pipeline\n");
			return -1;
		}
		tracker.add_pipeline("metadata", metadata);
    } */

    // Create main loop before starting pipelines and registering bus watches.
    GMainLoop* loop = g_main_loop_new(NULL, FALSE);
    g_main_loop = loop;

	// Determine output pipeline name for bus watch.
    if (!tracker.has_any_output()) {
		g_printerr("[FATAL] No output pipeline registered\n");
		g_main_loop_unref(loop);
		tracker.stop_all();
		return -1;
	}
	tracker.watch_all_outputs((GstBusFunc)bus_callback, loop);

    // FIX #2: Register input bus watch BEFORE start_all() to avoid the race
    // where EOS is posted before the watch is installed (e.g. very short videos).
    //
    // Also handles the live-source EOS forwarding:
    // interpipesrc has is-live=TRUE (required for nvstreammux).  Live sources
    // silently drop EOS events, so when the file source ends we manually push
    // EOS into the processing pipeline so it propagates to the output sink.
    GstElement* input_pipe = tracker.get_pipeline("input");
    if (input_pipe) {
        struct EosFwdData {
            GstElement* processing_pipeline;
            GMainLoop*  loop;
        };
        auto* eos_data = new EosFwdData{ tracker.get_pipeline("processing"), loop };

        GstBus* input_bus = gst_element_get_bus(input_pipe);
        auto input_bus_cb = [](GstBus*, GstMessage* msg, gpointer user_data) -> gboolean {
            auto* d = static_cast<EosFwdData*>(user_data);

            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                GError* err; gchar* dbg;
                gst_message_parse_error(msg, &err, &dbg);
                g_printerr("[BUS/input] Error from %s: %s\n",
                           GST_OBJECT_NAME(msg->src), err->message);
                g_printerr("[BUS/input] Debug: %s\n", dbg ? dbg : "none");
                g_clear_error(&err);
                g_free(dbg);
                g_main_loop_quit(d->loop);

            } else if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS) {
                g_print("[BUS/input] Input EOS -- forwarding EOS into processing pipeline\n");
                if (d->processing_pipeline) {
                    g_print("[BUS/input] Sending EOS to processing pipeline\n");
                    gst_element_send_event(d->processing_pipeline, gst_event_new_eos());
                }
            }
            return TRUE;
        };
        gst_bus_add_watch(input_bus, (GstBusFunc)input_bus_cb, eos_data);
        gst_object_unref(input_bus);
    }

	// Start all pipelines AFTER bus watches and probes are registered.
	std::cout << "Starting pipelines...\n";
    tracker.start_all();
    
    g_timeout_add(1000, [](gpointer data) -> gboolean {
    GstElement* proc = static_cast<InterpipeTracker*>(data)->get_pipeline("processing");
    if (proc) {
        GstElement* isink = gst_bin_get_by_name(GST_BIN(proc), "processing_output");
        if (isink) {
            guint n = 0;
            g_object_get(isink, "num-listeners", &n, NULL);
            g_printerr("[INTERPIPE] processing_output num-listeners: %u\n", n);
            gst_object_unref(isink);
        }
    }
    return G_SOURCE_REMOVE;
}, &tracker);
    
	// Start processing immediately
	//tracker.start_pipeline("processing");

    // FIX #2 (cont.): Removed g_usleep(2000000) here — it served no purpose
    // and introduced a race condition where EOS from a short video could arrive
    // during the sleep, before the input bus watch was registered (old code
    // registered the watch after the sleep).

	// Add probes to verify data flow into mux
/*
	GstElement* proc = tracker.get_pipeline("processing");
	if (proc) {
		GstElement* interpipesrc_elem = gst_bin_get_by_name(GST_BIN(proc), "interpipesrc");
		if (interpipesrc_elem) {
		    GstPad* src_pad = gst_element_get_static_pad(interpipesrc_elem, "src");
		    if (src_pad) {
		        gst_pad_add_probe(src_pad, GST_PAD_PROBE_TYPE_BUFFER,
		            [](GstPad* pad, GstPadProbeInfo* info, gpointer) -> GstPadProbeReturn {
		                static int count = 0;
		                if (++count % 30 == 0)
		                    g_print("[PROBE] interpipesrc src pad: %d buffers received\n", count);
		                return GST_PAD_PROBE_OK;
		            }, NULL, NULL);
		        gst_object_unref(src_pad);
		    }
		    gst_object_unref(interpipesrc_elem);
		}
	}
*/
    
    std::cout << "Running DeepStream pipeline...\n";
    std::cout << "Press Ctrl+C to stop\n\n";
    
    tracker.print_status();
	
	// Add a periodic status timeout
	g_timeout_add_seconds(60, [](gpointer data) -> gboolean {
		g_print("[TIMEOUT] Still running after 60 seconds...\n");
		return TRUE;
	}, nullptr);
	
	// g_main_loop_run() blocks until g_main_loop_quit() is called.
    // The old manual while(!g_main_loop_is_running()) loop was broken:
    // is_running() returns FALSE until run() is entered, so the loop
    // exited immediately without processing any GLib events — which is
    // why the pipeline appeared to do nothing and shut down instantly.
	
	// Simple version
	// Blocks until g_main_loop_quit() called
	// it can't easily check for keyboard input
	g_print("About to enter main loop...\n");
	g_main_loop_run(loop);
	g_print("Exited main loop!\n"); 
    if (g_interrupted) {
    	g_print("\nInterrupted by user\n");
	}
	
    // Run main loop (will exit on EOS, Error, or Ctrl+C)
	// Manual iteration
	/*
    while (!g_interrupted && g_main_loop_is_running(loop)) {
        g_main_context_iteration(g_main_loop_get_context(loop), FALSE);
        usleep(10000);  // 10ms
    }
    if (g_interrupted) {
        g_print("\nInterrupted by user\n");
    }
    */
    
    // Clean shutdown
    std::cout << "\nShutting down...\n";
    g_main_loop_unref(loop);
    
    tracker.stop_all();
    g_tracker = nullptr;
    
    std::cout << "Exiting cleanly\n";
    return 0;
}
