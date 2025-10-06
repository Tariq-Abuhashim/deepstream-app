/*
   Folder structure:
   deepstream-app/
   ├── build
   ├── build_and_run.sh
   ├── CMakeLists.txt
   ├── config_infer_primary_detr.txt
   ├── config_tracker_NvDCF_perf.yml
   ├── detr.engine
   ├── detr.onnx
   ├── export_detr_onnx.py
   ├── includes
   ├── labels_coco.txt
   ├── main.cpp
   ├── nvdsinfer_customparser_detr.cpp
   └── tracker_config.txt

Pipeline:
   uridecodebin → nvstreammux → nvinfer → nvtracker → nvvideoconvert → tee
                                                       				 ↘ queue → nvosd → sink
                                                        			 ↘ queue → appsink (ORB-SLAM3)

run:
./deepstream_orbslam <uri> <ORBvoc.txt> <settings.yaml>

June, 19, 2025, initially implemented as main.cpp
August, 26, 2025, orbslam updates

*/

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <nvdsmeta.h>
#include <gstnvdsmeta.h>
#include <nvds_meta.h>  // Core metadata definitions
#include <nvds_infer.h>
#include <nvdsinfer_custom_impl.h>
#include <nvds_tracker_meta.h>

#include "nvbufsurface.h"
//#include "nvbufsurfutil.h"
#include "gstnvdsmeta.h"

#include <opencv2/opencv.hpp>

#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <iostream>
#include <string>
#include <vector>



// Orbslam'
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define EIGEN_DONT_ALIGN_STATICALLY
#include "System.h"

//#include <glib.h>
//#include <cstring>
//#include <nvds_meta.h>

#define PGIE_CONFIG_FILE "config_infer_primary_detr.txt"
//#define TRACKER_CONFIG_FILE "tracker_config.txt"

// define detected objects
struct DetectedObject {
    int id;           // DeepStream tracker ID
    int class_id;     // COCO class (e.g., 0 person, 2 car)
    float confidence;
    float left;       // bbox in pixels
    float top;
    float width;
    float height;
};

// define frame packet
struct FramePacket {
    cv::Mat frame_bgr;                  // Image for SLAM
    double timestamp;                   // seconds
    std::vector<DetectedObject> objs;   // 2D detections for this frame
};

// define global frame queue from appsink → SLAM worker
//std::queue<cv::Mat> frameQueue; // image only
std::queue<FramePacket> g_queue; // image + detections
std::mutex g_mutex;
std::condition_variable g_cond;


// (Optional) keep if we still want console logs on the OSD branch
static GstPadProbeReturn osd_sink_pad_buffer_probe(	GstPad *pad, 
													GstPadProbeInfo *info, 
													gpointer user_data) 
{
    GstBuffer *buf = (GstBuffer *)info->data;
    if (!buf) {
        std::cerr << "[PROBE] no buffer\n";
        return GST_PAD_PROBE_OK;
    }
    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) {
        g_print("[PROBE] batch_meta == NULL on OSD sink pad\n");
        return GST_PAD_PROBE_OK;
    }
    
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; 
    	l_frame != NULL; l_frame = l_frame->next) 
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
		int total_objs = 0;
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; 
        	l_obj != NULL; l_obj = l_obj->next) 
        {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
            total_objs++;
            //if (obj_meta->class_id == 0) person_count++;
            //if (obj_meta->class_id == 2) vehicle_count++;
            std::cout << "[PROBE] Frame " << frame_meta->frame_num
                      << " src_id=" << frame_meta->source_id
                      << " objID=" << obj_meta->object_id
                      << " class=" << obj_meta->class_id
                      << " conf=" << obj_meta->confidence
                      << " bbox=(" << obj_meta->rect_params.left << "," << obj_meta->rect_params.top
                      << "," << obj_meta->rect_params.width << "x" << obj_meta->rect_params.height << ")"
                      << std::endl;
        }
        std::cout << "[PROBE] Frame " << frame_meta->frame_num 
        		<< " total_objs=" << total_objs 
        		<< std::endl;
        /*std::cout << "Frame " << frame_meta->frame_num
                  << " | Persons: " << person_count
                  << " | Vehicles: " << vehicle_count << std::endl;*/
    }
    return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn print_meta_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    GstBuffer *buf = (GstBuffer *)info->data;
    if (!buf) {
        std::cerr << "[PROBE] no buffer\n";
        return GST_PAD_PROBE_OK;
    }

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) {
        std::cout << "[PROBE] batch_meta == NULL on pad " << (char*)user_data << std::endl;
        return GST_PAD_PROBE_OK;
    }

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
        NvDsFrameMeta *fmeta = (NvDsFrameMeta*)l_frame->data;
        int obj_count = 0;
        for (NvDsMetaList *l_obj = fmeta->obj_meta_list; l_obj; l_obj = l_obj->next) {
            NvDsObjectMeta *om = (NvDsObjectMeta*)l_obj->data;
            obj_count++;
            std::cout << "[PROBE][" << (char*)user_data << "] frame#" << fmeta->frame_num
                      << " src=" << fmeta->source_id
                      << " objID=" << om->object_id
                      << " class=" << om->class_id
                      << " conf=" << om->confidence
                      << " bbox=(" << om->rect_params.left << "," << om->rect_params.top
                      << "," << om->rect_params.width << "x" << om->rect_params.height << ")"
                      << std::endl;
        }
        std::cout << "[PROBE][" << (char*)user_data << "] frame#" << fmeta->frame_num
                  << " total_objs=" << obj_count << std::endl;
    }
    return GST_PAD_PROBE_OK;
}


static void pad_added_handler(GstElement *src, GstPad *new_pad, gpointer user_data) {
    GstElement *streammux = (GstElement *)user_data;
    static gboolean linked = FALSE;

    if (linked) {
        g_print("Pad already linked, ignoring extra pads\n");
        return;
    }

    GstPad *sink_pad = gst_element_get_request_pad(streammux, "sink_0");
    if (!sink_pad) {
        g_printerr("Failed to get request pad sink_0 from streammux\n");
        return;
    }

    if (gst_pad_link(new_pad, sink_pad) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link source pad to streammux sink pad\n");
    } else {
        linked = TRUE;
        g_print("Source pad linked to streammux sink pad successfully\n");
    }

    gst_object_unref(sink_pad);
}


NvBufSurface* getNvDsMetaSurface(GstBuffer *buffer) {
    // Check if buffer has the metadata
    gpointer state = NULL;
    GstMeta *meta;
    
    while ((meta = gst_buffer_iterate_meta(buffer, &state))) {
        if (meta->info->api == g_type_from_name("NvDsMetaApi")) {
            NvDsMeta *nvds_meta = (NvDsMeta *)meta;
            if (nvds_meta->meta_type == NVDS_BATCH_GST_META) {
                // Found batch meta - extract surface
                return (NvBufSurface*)nvds_meta->meta_data;
            }
        }
    }
    return nullptr;
}


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
    
    std::cout << "[APPSINK] ========== Handler called, frame " << frame_num << " ==========\n";
    
    GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    if (!sample) {
        std::cout << "[APPSINK] no sample\n";
        return GST_FLOW_OK;
    }
    std::cout << "[APPSINK] Got sample\n";

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
        gst_sample_unref(sample);
        std::cout << "[APPSINK] no buffer\n";
        return GST_FLOW_OK;
    }
    std::cout << "[APPSINK] Got buffer\n";
    
    // Get caps info
    GstCaps *caps = gst_sample_get_caps(sample);
    GstStructure *s = caps ? gst_caps_get_structure(caps, 0) : nullptr;
    const gchar *format = s ? gst_structure_get_string(s, "format") : nullptr;
    int width = 0, height = 0;
    if (s) {
        gst_structure_get_int(s, "width", &width);
        gst_structure_get_int(s, "height", &height);
    }
    std::cout << "[APPSINK] new-sample fmt=" << (format?format:"(null)") 
              << " " << width << "x" << height << std::endl;

    // Extract NvBufSurface
    std::cout << "[APPSINK] Extracting NvBufSurface...\n";
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
    if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        std::cout << "[APPSINK] Buffer mapped, size: " << map.size << " bytes\n";
        
    /* 2. Surface Contains Pointers to GPU Memory
    */
        surf = (NvBufSurface*)map.data;
        gst_buffer_unmap(buffer, &map);
        std::cout << "[APPSINK] Buffer unmapped\n";
    } 
    std::cout << "[APPSINK] Got NvBufSurface, numFilled=" << surf->numFilled << std::endl;
    
    if (!surf || surf->numFilled == 0) {
        g_printerr("[APPSINK] Invalid NvBufSurface\n");
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }
    
    /* Surface parameters
    NvBufSurfaceParams *params = &surf->surfaceList[0];
    At this point, dataPtr points to GPU memory - your CPU can't read it directly! 
	 - width: 1384
	 - height: 512
	 - colorFormat: 33 (NV12)
	 - pitch: 1536 (row stride in bytes)
	 - dataPtr: 0x7f3766400000 (GPU memory address!)
	*/ 
    
    /* 3. Map GPU Memory to CPU
    After mapping, params->mappedAddr.addr[0] now points to CPU-accessible memory!
    */
    std::cout << "[APPSINK] Mapping surface to CPU...\n";
    if (NvBufSurfaceMap(surf, 0, 0, NVBUF_MAP_READ) != 0) {
        g_printerr("[APPSINK] NvBufSurfaceMap failed\n");
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }
    std::cout << "[APPSINK] Surface mapped\n";
    
    /* 4. Sync Data to ensure CPU has latest data
    Ensures any pending GPU operations are complete and data is fully available to CPU.
    */
    std::cout << "[APPSINK] Syncing to CPU...\n";
    NvBufSurfaceSyncForCpu(surf, 0, 0);
    std::cout << "[APPSINK] Sync complete\n";
    
    /* Get surface parameters
	*/
    NvBufSurfaceParams *params = &surf->surfaceList[0];
    std::cout << "[APPSINK] NVMM surface - colorFormat=" << params->colorFormat 
              << " width=" << params->width 
              << " height=" << params->height 
              << " pitch=" << params->pitch << std::endl;  
    
    // Create OpenCV Mat from mapped memory
    cv::Mat bgr;
    
    /* 5. Handle color format (33 = NV12 in NVMM)
    NV12 is a YUV color format that saves space by storing color information at lower resolution than brightness:
    Memory Layout of NV12 (1384x512 image):

		┌─────────────────────────────┐
		│  Y Plane (Luminance)        │  
		│  1536 × 512 bytes           │  ← pitch=1536, height=512
		│  (Brightness for each pixel)│     (extra 152 bytes per row for alignment)
		├─────────────────────────────┤
		│  UV Plane (Chrominance)     │
		│  1536 × 256 bytes           │  ← pitch=1536, height=256 (half height)
		│  (Color, interleaved U/V)   │     Each pair (U,V) covers 2x2 pixels
		└─────────────────────────────┘

		Total size: (1536 × 512) + (1536 × 256) = 786,432 + 393,216 = 1,179,648 bytes
		
	Human eyes are more sensitive to brightness than color, so this saves ~50% memory without visible quality loss
    */
    if (params->colorFormat == 33 || params->colorFormat == NVBUF_COLOR_FORMAT_NV12) {
        std::cout << "[APPSINK] Processing NV12 from NVMM\n";
        
        /* Extract Pixel Data
        */
        // Start of Y plane
        uint8_t *y_data = (uint8_t*)(params->mappedAddr.addr[0] ? params->mappedAddr.addr[0] : params->dataPtr);
        if (!y_data) {
            g_printerr("[APPSINK] NULL Y data pointer\n");
            NvBufSurfaceUnMap(surf, 0, 0);
            gst_sample_unref(sample);
            return GST_FLOW_OK;
        }
        // Start of UV plane
        uint8_t *uv_data = y_data + (params->pitch * params->height);
        
        // Now we have CPU pointers to the actual pixel data!
        
        /* Create OpenCV Wrappers
        */
        // Wrap GPU memory with OpenCV Mat (no copy yet)
        // These Mats are just "views" - they don't own the memory.
        std::cout << "[APPSINK] Creating Y plane...\n";
        cv::Mat y_plane(params->height, params->width, CV_8UC1, y_data, params->pitch);
        std::cout << "[APPSINK] Creating UV plane...\n";
        cv::Mat uv_plane(params->height / 2, params->width / 2, CV_8UC2, uv_data, params->pitch);
        // Clone (Deep Copy)
        std::cout << "[APPSINK] Cloning Y plane...\n";
        cv::Mat y_copy = y_plane.clone();
        std::cout << "[APPSINK] Cloning UV plane...\n";
        cv::Mat uv_copy = uv_plane.clone();
        // Reconstruct Full NV12 for OpenCV
        std::cout << "[APPSINK] Assembling NV12...\n";
        cv::Mat nv12(params->height + params->height / 2, params->width, CV_8UC1);
        y_copy.copyTo(nv12(cv::Rect(0, 0, params->width, params->height)));
        cv::Mat uv_single_channel = uv_copy.reshape(1, params->height / 2);
        uv_single_channel.copyTo(nv12(cv::Rect(0, params->height, params->width, params->height / 2)));
        // Convert to BGR
        std::cout << "[APPSINK] Converting NV12 to BGR...\n";
        cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
        std::cout << "[APPSINK] Conversion complete: " << bgr.cols << "x" << bgr.rows << std::endl;
        
    } else {
        g_printerr("[APPSINK] Unsupported color format: %d\n", params->colorFormat);
    }
    
    // 6. Unmap the surface (GPU Memory)
    std::cout << "[APPSINK] Unmapping surface...\n";
    NvBufSurfaceUnMap(surf, 0, 0);
    std::cout << "[APPSINK] Surface unmapped\n";
    
    if (bgr.empty()) {
        g_printerr("[APPSINK] Failed to create BGR image\n");
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }
    std::cout << "[APPSINK] BGR image created: " << bgr.cols << "x" << bgr.rows << std::endl;
    
    // 7. Extract metadata
    std::cout << "[APPSINK] Extracting metadata...\n";
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
                objs.push_back(o);
            }
            // NOTE: If we batch streams, we may want to key by fmeta->source_id
            // and/or ensure we send only the packet that matches this appsink branch.
            break; // batch-size=1 in config → take the first frame
            // if batch-size>1 or use multiple sources, then we should iterate frames 
            // and route by NvDsFrameMeta::source_id
        }
    }
    std::cout << "[APPSINK] Metadata extracted, objects: " << objs.size() << "\n";
    
    // Queue the frame
    std::cout << "[APPSINK] Queueing frame...\n";
    double ts = (double)cv::getTickCount() / cv::getTickFrequency();
    
    {
        std::lock_guard<std::mutex> lk(g_mutex);
        g_queue.push(FramePacket{ bgr.clone(), ts, std::move(objs) });
        std::cout << "[APPSINK] Frame queued, queue size: " << g_queue.size() << "\n";
    }
    g_cond.notify_one();
    
    std::cout << "[APPSINK] Unreferencing sample...\n";
    gst_sample_unref(sample);
    std::cout << "[APPSINK] Handler complete, returning GST_FLOW_OK\n";
    return GST_FLOW_OK;
}


// Bus callback function (add this before main())
static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_ERROR: {
            GError *err;
            gchar *debug;
            gst_message_parse_error(msg, &err, &debug);
            std::cerr << "Error: " << err->message << std::endl;
            if (debug) std::cerr << "Debug: " << debug << std::endl;
            g_error_free(err);
            g_free(debug);
            break;
        }
        case GST_MESSAGE_WARNING: {
            GError *err;
            gchar *debug;
            gst_message_parse_warning(msg, &err, &debug);
            std::cerr << "Warning: " << err->message << std::endl;
            if (debug) std::cerr << "Debug: " << debug << std::endl;
            g_error_free(err);
            g_free(debug);
            break;
        }
        default:
            break;
    }
    return TRUE;
}

static gboolean bus_error_callback(GstBus *bus, GstMessage *msg, gpointer data) {
    GError *err;
    gchar *debug;
    gst_message_parse_error(msg, &err, &debug);
    g_printerr("ERROR: %s\n", err->message);
    if (debug) g_printerr("DEBUG: %s\n", debug);
    g_error_free(err);
    g_free(debug);
    return FALSE;  // Remove from event loop
}

static gboolean bus_warning_callback(GstBus *bus, GstMessage *msg, gpointer data) {
    GError *err;
    gchar *debug;
    gst_message_parse_warning(msg, &err, &debug);
    g_printerr("WARNING: %s\n", err->message);
    if (debug) g_printerr("DEBUG: %s\n", debug);
    g_error_free(err);
    g_free(debug);
    return FALSE;  // Remove from event loop
}

int main(int argc, char *argv[]) {
	if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <uri> <ORBvoc.txt> <settings.yaml>" << std::endl;
        return -1;
    }
    
    bool orbslam = true;
    
	/* Initializes the GStreamer library
    */
    gst_init(&argc, &argv);

	/* Element factory 
    */
    GstElement *pipeline  = gst_pipeline_new("deepstream-pipeline");
    GstElement *source    = gst_element_factory_make("uridecodebin", "src");
    GstElement *streammux = gst_element_factory_make("nvstreammux", "streammux");
    GstElement *pgie      = gst_element_factory_make("nvinfer", "pgie");
    GstElement *tracker   = gst_element_factory_make("nvtracker", "tracker");
    GstElement *nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvidconv");
    GstElement *nvosd     = gst_element_factory_make("nvdsosd", "nvosd");
    GstElement *sink      = gst_element_factory_make("nveglglessink", "sink");

    if (!pipeline || !source || !streammux || !pgie || !tracker || !nvvidconv || !nvosd || !sink) {
        std::cerr << "Failed to create elements." << std::endl;
        return -1;
    }
     
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
		    std::cerr << "Failed to create a display element." << std::endl;
		    return -1;
		}
    }	

	/* GObject Property sitter 
	*/
    g_object_set(G_OBJECT(source), "uri", argv[1], NULL);
    g_object_set(G_OBJECT(pgie), "config-file-path", PGIE_CONFIG_FILE, NULL);
    
    // Check if tracker library exists
	std::string tracker_lib = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so";
	std::ifstream lib_file(tracker_lib);
	if (!lib_file.good()) {
		std::cerr << "Tracker library not found: " << tracker_lib << std::endl;
		std::cerr << "Available tracker libraries:" << std::endl;
		int status = system("ls -la /opt/nvidia/deepstream/deepstream/lib/libnvds_mot*");
	}

	// Check if config file exists
	std::string config_file = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml";
	std::ifstream conf_file(config_file);
	if (!conf_file.good()) {
		std::cerr << "Tracker config file not found: " << config_file << std::endl;
		std::cerr << "Available config files:" << std::endl;
		int status = system("find /opt/nvidia/deepstream -name '*tracker*.yml' -o -name '*tracker*.txt' 2>/dev/null");
	}

    g_object_set(G_OBJECT(tracker),
             "tracker-width", 640,  // Match streammux width
             "tracker-height", 384,  // Match streammux height
             "ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
             "ll-config-file", "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml",
             "enable-batch-process", TRUE,
             "gpu-id", 0,
             "enable-past-frame", TRUE,
             NULL);
    g_print("Tracker element created successfully\n");
    
    g_object_set(G_OBJECT(streammux), "width", 1382, "height", 512, "batch-size", 1, 
    			"batched-push-timeout", 40000, NULL);
    g_object_set(G_OBJECT(sink), "sync", FALSE, "async", FALSE, NULL);

	if (orbslam) {
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
		
		GstCaps *caps = gst_caps_from_string("video/x-raw(memory:NVMM)");
		//GstCaps *caps = gst_caps_from_string("video/x-raw, format=RGBA, memory=CPU");
		//GstCaps *caps = gst_caps_from_string("video/x-raw(memory:SystemMemory),format=NV12");
		g_object_set(G_OBJECT(capsfilter_app), "caps", caps, NULL);
		gst_caps_unref(caps);
	}

	/* Add multiple elements into a GstBin at once
    */
    if (orbslam) {
		gst_bin_add_many(GST_BIN(pipeline),
					source, streammux, pgie, tracker, nvvidconv,
					tee,
					queue_osd, nvvidconv_osd, nvosd, sink,
					queue_app, nvvidconv_app, capsfilter_app, appsink,
					NULL);
    } else {
		gst_bin_add_many(GST_BIN(pipeline),
		                 source, streammux, pgie, tracker, nvvidconv, nvosd, sink,
		                 NULL);
    }

	/* Hanble dynamic pads
    */
    g_signal_connect(source, "pad-added", G_CALLBACK(pad_added_handler), streammux);
    if (orbslam) {
		g_signal_connect(appsink, "new-sample", G_CALLBACK(orbslam_handler), NULL); // Orbslam updates
	}

	/* Attempts to link each element’s src pad to the next element’s sink pad in order
    streammux → pgie → tracker → nvvidconv → nvosd → sink
    Pads must exist already.
    Elements wth static pads, src and sink pads exist after creation.
    Elements with dynamic pads, src and sink pads must be created first. 
    */
    
    /* Orbslam updates 
    streammux → pgie → tracker → nvvidconv → tee
           										↘→ queue_osd → nvosd → sink
           										↘→ queue → nvvidconv → capsfilter → appsink (ORB-SLAM3)
    */
 	if (orbslam) { 
		// Link main path (before tee)
		if (!gst_element_link_many(streammux, pgie, tracker, nvvidconv, tee, NULL)) {
			g_printerr("Main pipeline elements could not be linked.\n");
			return -1;
		}

		// Request a new src pad from tee for OSD branch
		GstPad *tee_src_pad_osd = gst_element_get_request_pad(tee, "src_%u");
		GstPad *queue_osd_sink_pad = gst_element_get_static_pad(queue_osd, "sink");
		if (gst_pad_link(tee_src_pad_osd, queue_osd_sink_pad) != GST_PAD_LINK_OK) {
			g_printerr("Tee to OSD branch could not be linked.\n");
			return -1;
		}
		gst_object_unref(queue_osd_sink_pad);
		gst_object_unref(tee_src_pad_osd);

		// Request a new src pad from tee for App branch
		GstPad *tee_src_pad_app = gst_element_get_request_pad(tee, "src_%u");
		GstPad *queue_app_sink_pad = gst_element_get_static_pad(queue_app, "sink");
		if (gst_pad_link(tee_src_pad_app, queue_app_sink_pad) != GST_PAD_LINK_OK) {
			g_printerr("Tee to App branch could not be linked.\n");
			return -1;
		}
		gst_object_unref(queue_app_sink_pad);
		gst_object_unref(tee_src_pad_app);

		// Now link the OSD branch normally
		if (!gst_element_link_many(queue_osd, nvvidconv_osd, nvosd, sink, NULL)) {
			g_printerr("OSD branch could not be linked.\n");
			return -1;
		}


		// And link the App branch normally
		if (!gst_element_link_many(queue_app, nvvidconv_app, capsfilter_app, appsink, NULL)) {
			g_printerr("App branch could not be linked.\n");
			return -1;
		}
	} else {
	    if (!gst_element_link_many(streammux, pgie, tracker, nvvidconv, nvosd, sink, NULL)) {
		    g_printerr("Elements could not be linked.\n");
		    return -1;
		}
	}
	
	// After you have created and linked elements, but before gst_element_set_state(..., PLAYING):
/*	GstPad *pgie_src_pad = gst_element_get_static_pad(pgie, "src");
	if (pgie_src_pad) {
		gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER, print_meta_probe, (gpointer)"pgie_src", NULL);
		gst_object_unref(pgie_src_pad);
	} else {
		g_printerr("ERROR: Could not get pgie src pad\n");
	}*/

/*	GstPad *tracker_sink_pad = gst_element_get_static_pad(tracker, "sink");
	if (tracker_sink_pad) {
		gst_pad_add_probe(tracker_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, print_meta_probe, (gpointer)"tracker_sink", NULL);
		gst_object_unref(tracker_sink_pad);
	} else {
		g_printerr("ERROR: Could not get tracker sink pad\n");
	}*/
	
/*	GstPad *tracker_src_pad = gst_element_get_static_pad(tracker, "src");
	if (tracker_src_pad) {
		gst_pad_add_probe(tracker_src_pad, GST_PAD_PROBE_TYPE_BUFFER, print_meta_probe, (gpointer)"tracker_src", NULL);
		gst_object_unref(tracker_src_pad);
	} else {
		g_printerr("ERROR: Could not get tracker src pad\n");
	}*/
	
	// Check if elements reached PLAYING state
	GstState state;
	gst_element_get_state(tee, &state, NULL, GST_CLOCK_TIME_NONE);
	g_print("Tee state: %d (4=PLAYING)\n", state);

	gst_element_get_state(queue_osd, &state, NULL, GST_CLOCK_TIME_NONE);  
	g_print("Queue_osd state: %d (4=PLAYING)\n", state);

	gst_element_get_state(sink, &state, NULL, GST_CLOCK_TIME_NONE);
	g_print("Sink state: %d (4=PLAYING)\n", state);
	
	GstStateChangeReturn ret_app = gst_element_set_state(appsink, GST_STATE_PLAYING);
	g_print("Appsink state change result: %d (3=SUCCESS, 2=ASYNC, 1=NO_PREROLL, 0=FAILURE)\n", ret_app);

	GstStateChangeReturn ret_caps = gst_element_set_state(capsfilter_app, GST_STATE_PLAYING);  
	g_print("Capsfilter state change result: %d\n", ret_caps);

	GstStateChangeReturn ret_conv = gst_element_set_state(nvvidconv_app, GST_STATE_PLAYING);
	g_print("Nvvidconv_app state change result: %d\n", ret_conv);
	
	
	/* set your pipeline to PLAYING state.
    */
	GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
	if (ret == GST_STATE_CHANGE_FAILURE) {
		g_printerr("Pipeline failed to start!\n");
		return -1;
	}
	
    std::cout << "Running DeepStream + ORB-SLAM3 pipeline..." << std::endl;
    
    /*Orbslam
    */
    std::cout << argv[2] << std::endl;
    std::cout << argv[3] << std::endl;
    ORB_SLAM3::System SLAM(argv[2], argv[3], ORB_SLAM3::System::MONOCULAR, false);
    std::thread orbslamThread([&]() {
    	cv::Mat gray; // keep local copies to avoid repeated allocations
		while (true) {

			FramePacket pkt;
			{
				std::unique_lock<std::mutex> lock(g_mutex);
				g_cond.wait(lock, []{ return !g_queue.empty(); });
				pkt = std::move(g_queue.front());
				g_queue.pop();
			}
			if (pkt.frame_bgr.empty()) {
				std::cerr << "[ORBSLAM] Empty frame received, skipping\n";
				continue;
			}
		    cv::cvtColor(pkt.frame_bgr, gray, cv::COLOR_BGR2GRAY);
		    
		    // Timestamp: pkt.timestamp (seconds). ORB-SLAM expects seconds (double).
            // If you used cv::getTickCount for pkt.timestamp earlier, it's fine.
        	double ts = pkt.timestamp;
		    
		    std::cout << "[ORBSLAM] Processing frame: " 
		    		  << ts << ", " 
		    		  << gray.cols << "x" << gray.rows << "x" << gray.channels() << "\n";
		              
		    //std::cout << "[ORBSLAM] Processing frame: " << gray.cols << "x" << gray.rows 
		    //          << "x" << gray.channels() << " objs=" << pkt.objects.size() << "\n";

		    // plain SLAM
		    //SLAM.TrackMonocular(gray, ts);
		    
		    // object-level adapter (new implementation)
			// e.g., associate 2D boxes with landmarks, semantic constraints, etc.
			// TrackMonocularWithObjects(SLAM, pkt.frame_bgr, pkt.timestamp, pkt.objs);
			
			// sleep small amount to throttle
          	std::this_thread::sleep_for(std::chrono::milliseconds(1));
		} // while loop
	});
	orbslamThread.detach();

	/* The GstBus is where the pipeline posts messages
    gst_bus_timed_pop_filtered blocks forever (GST_CLOCK_TIME_NONE) until:
    An ERROR message arrives
    An EOS (end-of-stream) message arrives
    */
    GstBus *bus = gst_element_get_bus(pipeline);
    
	g_signal_connect(bus, "message::error", G_CALLBACK(bus_error_callback), NULL);
	g_signal_connect(bus, "message::warning", G_CALLBACK(bus_warning_callback), NULL);

    GstMessage *msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
                                                 (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    if (msg) { // !=NULL
        GError *err = nullptr;
        gchar *debug_info = nullptr;
        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_ERROR:
                gst_message_parse_error(msg, &err, &debug_info);
                std::cerr << "GStreamer ERROR: " << (err ? err->message : "(unknown)") << "\n";
                g_clear_error(&err);
                g_free(debug_info);
                break;
            case GST_MESSAGE_EOS:
                std::cout << "End of stream." << std::endl;
                break;
            default:
                std::cerr << "Unexpected message received." << std::endl;
                break;
        }
        gst_message_unref(msg);
    }

    /* shutdown sequence
    */
    gst_element_set_state(pipeline, GST_STATE_NULL); // stop the pipeline
    //gst_element_release_request_pad(tee, tee_src_pad_osd); // release pads
	//gst_element_release_request_pad(tee, tee_src_pad_app); // release pads
    gst_object_unref(bus); // free bus
    gst_object_unref(pipeline); // free pipeline
    return 0;
}

