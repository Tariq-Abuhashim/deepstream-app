
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
//#include "nvbufsurfutil.h"
#include "gstnvdsmeta.h"

#include <opencv2/opencv.hpp>

#include <queue>
#include <deque>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <iostream>
#include <string>
#include <vector>

#include <atomic>
std::atomic<bool> g_stop{false};

// Orbslam'
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define EIGEN_DONT_ALIGN_STATICALLY
#include "System.h"

//#include <glib.h>
//#include <cstring>
//#include <nvds_meta.h>

#define PGIE_CONFIG_FILE "config_infer_primary_detr.txt"
//#define TRACKER_CONFIG_FILE "tracker_config.txt"

#include <fstream>
#include <iostream>
#include <cstring>

// prints help
void print_help() {
    std::ifstream f("README.md");
    if (!f.is_open()) {
        std::cerr << "ERROR: README.md not found\n";
        return;
    }
    std::cout << f.rdbuf();
}

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
	// Explicit destructor for debugging
    ~FramePacket() {
		// cv::Mat handles its own memory, but let's be explicit
		frame_bgr.release();
    }
};

// define global frame queue from appsink → SLAM worker
//std::queue<cv::Mat> frameQueue; // image only
std::deque<FramePacket> g_queue; // image + detections
std::mutex g_mutex;
std::condition_variable g_cond;

const size_t MAX_QUEUE = 50; // tune down from 1283


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

static GstPadProbeReturn print_meta_probe(GstPad *pad, 
										  GstPadProbeInfo *info, 
										  gpointer user_data)
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

	guint64 timestamp = buf->pts;  // PTS in nanoseconds

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
        NvDsFrameMeta *fmeta = (NvDsFrameMeta*)l_frame->data;
		guint src_id	= fmeta->source_id;
		guint frame_id	= fmeta->frame_num;
		float frame_w	= fmeta->source_frame_width;
		float frame_h	= fmeta->source_frame_height;
		
        int obj_count = 0;
        for (NvDsMetaList *l_obj = fmeta->obj_meta_list; l_obj; l_obj = l_obj->next) {

            NvDsObjectMeta *obj_meta = (NvDsObjectMeta*)l_obj->data;

            obj_count++;

			// prep values
			guint track_id  = obj_meta->object_id;    // tracker ID
			guint class_id  = obj_meta->class_id;     // detector class
			float conf		= obj_meta->confidence;
			float left      = obj_meta->rect_params.left;
			float top       = obj_meta->rect_params.top;
			float width     = obj_meta->rect_params.width;
			float height    = obj_meta->rect_params.height;

			// Convert timestamp from nanoseconds → milliseconds
			double ts_ms = (double)timestamp / 1e6;

			// ---- PRINT TO STDOUT ----
			printf("%.3f,%u,%u,%.2f,%.2f,%.2f,%.2f\n",
                   ts_ms, track_id, class_id,
                   left, top, width, height);
			fflush(stdout);
        }

		// ---- PRINT TO STDOUT ----
		printf("[PROBE][%s] frame#[%u] total_objs=[%u]\n", (char*)user_data, frame_id, obj_count);
		fflush(stdout);
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


// helper to make a CPU NvBufSurface and copy with NvBufSurfTransform
static bool copyNvToCpuAndMakeBGR(NvBufSurface *src_surf, cv::Mat &bgr) {
    if (!src_surf || src_surf->numFilled < 1) {
        g_printerr("[copyNvToCpuAndMakeBGR] Invalid source surface\n");
        return false;
    }

    NvBufSurface *dst_surf = nullptr;
    NvBufSurfaceCreateParams create_params;
    memset(&create_params, 0, sizeof(create_params));
    create_params.gpuId = src_surf->gpuId;
    create_params.width = src_surf->surfaceList[0].width;
    create_params.height = src_surf->surfaceList[0].height;
    create_params.colorFormat = src_surf->surfaceList[0].colorFormat;
    create_params.layout = NVBUF_LAYOUT_PITCH;
    create_params.memType = NVBUF_MEM_CUDA_UNIFIED;   

    if (NvBufSurfaceCreate(&dst_surf, 1, &create_params) != 0) {
        g_printerr("[copyNvToCpuAndMakeBGR] NvBufSurfaceCreate failed\n");
        return false;
    }

    // Configure transform session
    NvBufSurfTransformConfigParams config_params;
    memset(&config_params, 0, sizeof(config_params));
    config_params.compute_mode = NvBufSurfTransformCompute_Default;
    NvBufSurfTransformSetSessionParams(&config_params);

    NvBufSurfTransformParams xform_params;
    memset(&xform_params, 0, sizeof(xform_params));
    xform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
    xform_params.transform_filter = NvBufSurfTransformInter_Default;

    // Perform GPU → CPU transform
    if (NvBufSurfTransform(src_surf, dst_surf, &xform_params) != 0) {
        g_printerr("[copyNvToCpuAndMakeBGR] NvBufSurfTransform failed\n");
        NvBufSurfaceDestroy(dst_surf);
        return false;
    }

    // ✅ Map the destination (CPU) buffer — not the source
    if (NvBufSurfaceMap(dst_surf, 0, 0, NVBUF_MAP_READ) != 0) {
        g_printerr("[copyNvToCpuAndMakeBGR] NvBufSurfaceMap failed (dst)\n");
        NvBufSurfaceDestroy(dst_surf);
        return false;
    }

    NvBufSurfaceSyncForCpu(dst_surf, 0, 0);
    NvBufSurfaceParams *params = &dst_surf->surfaceList[0];

    uint8_t *y_data = (uint8_t *)params->mappedAddr.addr[0];
    uint8_t *uv_data = y_data + (params->pitch * params->height);

    cv::Mat y(params->height, params->width, CV_8UC1, y_data, params->pitch);
    cv::Mat uv(params->height / 2, params->width / 2, CV_8UC2, uv_data, params->pitch);

    // Assemble NV12 for OpenCV conversion
    cv::Mat nv12(params->height + params->height / 2, params->width, CV_8UC1);
    y.copyTo(nv12(cv::Rect(0, 0, params->width, params->height)));
    cv::Mat uv_single = uv.reshape(1, params->height / 2);
    uv_single.copyTo(nv12(cv::Rect(0, params->height, params->width, params->height / 2)));

    cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);

    NvBufSurfaceUnMap(dst_surf, 0, 0);
    NvBufSurfaceDestroy(dst_surf);
    return true;
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
    std::cout << "memType=" << surf->memType
          << " mappedAddr=" << surf->surfaceList[0].mappedAddr.addr[0]
          << " dataPtr=" << surf->surfaceList[0].dataPtr << std::endl;
    cv::Mat bgr;
    // If buffer is on GPU, use transform to copy into CPU system memory
	if (surf->memType == NVBUF_MEM_CUDA_DEVICE || surf->memType == NVBUF_MEM_DEFAULT) {
		std::cout << "[APPSINK] Surface is GPU memory (NVBUF_MEM_CUDA_DEVICE), copying to CPU...\n";
		if (!copyNvToCpuAndMakeBGR(surf, bgr)) {
		    g_printerr("[APPSINK] Failed to transform NVMM -> CPU\n");
		    gst_sample_unref(sample);
		    return GST_FLOW_OK;
		}
	} else {
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
		//cv::Mat bgr;
		
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
	
	}
    
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
        if (g_queue.size() >= MAX_QUEUE) {
        	g_queue.pop_front();
        }
        g_queue.push_back(FramePacket{ bgr.clone(), ts, std::move(objs) });
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

// Usage: deepstream-orbslam <uri> <ORBvoc.txt> <settings.yaml> <config_infer_primary_detr.txt>
int main(int argc, char *argv[]) {

	if (strcmp(argv[1], "-help") || strcmp(argv[1], "--help")) {
		print_help();
		return 0;
	}

	if (argc < 4 || argc > 5) {
		std::cerr << "Usage: " << argv[0] << " <uri> <ORBvoc.txt> <settings.yaml> [--headless]"
				  << std::endl;
		return -1;
	}
    
    bool headless = (argc == 5 && std::string(argv[4]) == "--headless");
	bool orbslam = true;
	
	std::cout << "[INFO] Running in " << (headless ? "HEADLESS" : "DISPLAY") << " mode.\n";
    
    /* Check if tracker library exists
    */
	std::string tracker_lib = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so";
	std::ifstream lib_file(tracker_lib);
	if (!lib_file.good()) {
		std::cerr << "Tracker library not found: " << tracker_lib << std::endl;
		std::cerr << "Available tracker libraries:" << std::endl;
		int status = system("ls -la /opt/nvidia/deepstream/deepstream/lib/libnvds_mot*");
	}

	/* Check if config file exists
    */
	std::string config_file = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml";
	std::ifstream conf_file(config_file);
	if (!conf_file.good()) {
		std::cerr << "Tracker config file not found: " << config_file << std::endl;
		std::cerr << "Available config files:" << std::endl;
		int status = system("find /opt/nvidia/deepstream -name '*tracker*.yml' -o -name '*tracker*.txt' 2>/dev/null");
	}
	
	/* Initializes the GStreamer library */
    gst_init(&argc, &argv);
    
    /* Create the empty pipeline */
	GstElement *pipeline  = gst_pipeline_new("deepstream-pipeline");


	/* Element factory 
		Basic object detection and tracking branch
    */
    GstElement *source    = gst_element_factory_make("uridecodebin", "src");
    // → we can insert a preprocessing step here
    GstElement *streammux = gst_element_factory_make("nvstreammux", "streammux");
    GstElement *pgie      = gst_element_factory_make("nvinfer", "pgie");
    GstElement *tracker   = gst_element_factory_make("nvtracker", "tracker");
    GstElement *nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvidconv");
    
    if (!pipeline || !source || !streammux || !pgie || !tracker || !nvvidconv) {
        g_printerr ("Failed to create an essential pipeline element.");
        return -1;
    }
    
	g_object_set(G_OBJECT(source), "uri", argv[1], NULL);
    g_object_set(G_OBJECT(pgie), "config-file-path", PGIE_CONFIG_FILE, "batch-size", 1, NULL);
    g_object_set(G_OBJECT(streammux), 
    			"width", 1280, "height", 720, // 1382 x 512  for kitti dataset, but now it limited
    			"batch-size", 1,
    			"batched-push-timeout", 40000, NULL);
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

	/* Insert a preprocessing branch between uridecodebin and nvstreammux
		This forces a smaller frame size, for example 1280×720 (within GPU macroblocks limit (8192): 1280×720 → 3600 macroblocks).
		
		uridecodebin
		   → nvvideoconvert_pre
		   → capsfilter_pre (forces 1280×720)
		   → nvstreammux
    */

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
		    std::cerr << "Failed to create an orbslam element." << std::endl;
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
		
		GstCaps *caps = gst_caps_from_string("video/x-raw(memory:NVMM)");
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
				             source, streammux, pgie, tracker, nvvidconv,
				             tee,
				             queue_osd, nvvidconv_osd, sink,          // no nvosd here
				             queue_app, nvvidconv_app, capsfilter_app, appsink,
				             NULL);
		} else {
			gst_bin_add_many(GST_BIN(pipeline),
				             source, streammux, pgie, tracker, nvvidconv,
				             tee,
				             queue_osd, nvvidconv_osd, nvosd, sink,  // include nvosd for display
				             queue_app, nvvidconv_app, capsfilter_app, appsink,
				             NULL);
		}
    } else {
		gst_bin_add_many(GST_BIN(pipeline),
					source, streammux, pgie, tracker, nvvidconv, 
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
    g_signal_connect(source, "pad-added", G_CALLBACK(pad_added_handler), streammux);
    //g_signal_connect(source, "pad-added", G_CALLBACK(pad_added_handler), nvvidconv_pre);
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
	GstPad *tracker_src_pad = gst_element_get_static_pad(tracker, "src");
	gst_pad_add_probe(tracker_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                  print_meta_probe, (gpointer)"nvtracker", NULL);
	gst_object_unref(tracker_src_pad);

		
	/* Start playing */
	GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
	if (ret == GST_STATE_CHANGE_FAILURE) {
		g_printerr ("Unable to set the pipeline to the playing state.\n");
		gst_object_unref (pipeline);
		return -1;
	}
	
	GstState state;
	gst_element_get_state(pipeline, &state, NULL, 5 * GST_SECOND);
	g_print("Pipeline state: %d\n", state);
    
    
    /*Orbslam
    */
    std::cout << "Running DeepStream + ORB-SLAM3 pipeline..." << std::endl;
    std::cout << argv[2] << std::endl;
    std::cout << argv[3] << std::endl;
    ORB_SLAM3::System SLAM(argv[2], argv[3], ORB_SLAM3::System::MONOCULAR, !headless);
    std::thread orbslamThread([&]() {
    	cv::Mat gray; // keep local copies to avoid repeated allocations
		while (true) {

			FramePacket pkt;
			{
				std::unique_lock<std::mutex> lock(g_mutex);
				//g_cond.wait(lock, []{ return !g_queue.empty(); });
				g_cond.wait(lock, []{ return !g_queue.empty() || g_stop.load(); });
				
				// If stop requested and no frames left -> exit
		        if (g_stop.load() || g_queue.empty()) {
		            std::cout << "[ORBSLAM] Stop flag and empty queue -> exiting thread.\n";
		            break;
		        }
            
            	if (!g_queue.empty()) {
					pkt = std::move(g_queue.front());
					g_queue.pop_front();
				} else {
					continue;
				}
			}
			if (pkt.frame_bgr.empty()) {
            	std::cerr << "[ORBSLAM] Received empty frame packet, skipping\n";
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
		    SLAM.TrackMonocular(gray, ts);
		    
		    // object-level adapter (new implementation)
			// e.g., associate 2D boxes with landmarks, semantic constraints, etc.
			// TrackMonocularWithObjects(SLAM, pkt.frame_bgr, pkt.timestamp, pkt.objs);
			
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
    
    std::cout << "Stopping pipeline..." << std::endl;
    
    // pause pipeline and signal orbslam thread to exit
    gst_element_set_state(pipeline, GST_STATE_PAUSED); // Pause
	g_stop.store(true);
    g_cond.notify_all();
    
    std::cout << "Joining ORBSLAM ..." << std::endl;
    orbslamThread.join();
    
	// Clear queue after thread is done
    std::cout << "Clearing frame queue..." << std::endl;
    {
        std::lock_guard<std::mutex> lk(g_mutex);
        g_queue.clear();
    }
    std::cout << "✓ Queue cleared" << std::endl;
    
    std::cout << "Shutting down ORBSLAM ..." << std::endl;
    SLAM.Shutdown();
    std::cout << "✓ ORBSLAM shutdown complete" << std::endl;
    
        
    // Save camera trajectory
	// Creating a directory
	std::cout << "Saving Map data ..." << std::endl;
	if (mkdir("sparse", 0777) == -1)
		cerr << "Error :  " << strerror(errno) << endl;
	else
		cout << "Directory created";
	SLAM.WriteCamerasText("sparse/cameras.txt");
	SLAM.WriteImagesText("sparse/images.txt");
	SLAM.WritePoints3DText("sparse/points3D.txt");
	std::cout << "✓ Map data saved" << std::endl;
	
	// stop the pipeline
    //gst_element_set_state(pipeline, GST_STATE_NULL);
	
	// recommened on arm64
	std::cout << "Exiting cleanly..." << std::endl;
	std::quick_exit(0);
	
	// recommened on aarch64
	// Flush all output before exit
	/*
    std::cout << "Exiting cleanly..." << std::endl;
    std::cout.flush();
    std::cerr.flush();
    
    // Use POSIX _exit - most reliable on embedded Linux
    _exit(EXIT_SUCCESS);
    */
	
	/*
    std::cout << "Setting pipeline to NULL state..." << std::endl;
    gst_element_set_state(pipeline, GST_STATE_NULL);
    std::cout << "✓ Pipeline stopped" << std::endl;  
	gst_object_unref(bus);
	gst_object_unref(pipeline);
	*/

    /* shutdown sequence
    */  
/* 
	// 2. Release tee request pads if they exist
    if (tee_src_pad_osd) {
    	std::cout << "Release tee request osd pad..." << std::endl;
        gst_element_release_request_pad(tee, tee_src_pad_osd);
        gst_object_unref(tee_src_pad_osd);
    }
    if (tee_src_pad_app) {
    	std::cout << "Release tee request app pad..." << std::endl;
        gst_element_release_request_pad(tee, tee_src_pad_app);
        gst_object_unref(tee_src_pad_app);
    }

	// 3. Unref the pads that were obtained during linking
    if (queue_osd_sink_pad) {
    	std::cout << "Unref osd sink pad..." << std::endl;
        gst_object_unref(queue_osd_sink_pad);
    }
    if (queue_app_sink_pad) {
    	std::cout << "Unref app sink pad..." << std::endl;
        gst_object_unref(queue_app_sink_pad);
    }
	
    // 4. Free the bus
    std::cout << "Free the bus..." << std::endl;
    gst_object_unref(bus);

    // 5. Free the pipeline (this also frees all contained elements)
    std::cout << "Free the pipeline..." << std::endl;
    gst_object_unref(pipeline);

    std::cout << "Cleanup complete." << std::endl;
*/

    return 0;
}

