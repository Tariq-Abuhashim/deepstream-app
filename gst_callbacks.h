#pragma once

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <nvdsmeta.h>
#include <gstnvdsmeta.h>
#include <iostream>

// variable declared, but exists somewhere else
extern std::map<guint64, BBoxHistory> g_object_histories; // <obj_id, obj_history>
extern std::mutex g_history_mutex;
extern guint g_current_frame_num;

const int MIN_FRAMES_TO_DISPLAY = 3;
const int BBOX_SMOOTHING_WINDOW = 5;  // Average over last 5 frames
const int FRAMES_BEFORE_REMOVAL = 30;  // Remove after 30 frames of not being seen

// Print metadata probe
inline GstPadProbeReturn print_meta_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    GstBuffer *buf = (GstBuffer *)info->data;
    if (!buf) return GST_PAD_PROBE_OK;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) {
        std::cerr << "[PROBE] batch_meta == NULL on pad " << (char*)user_data << "\n";
        return GST_PAD_PROBE_OK;
    }

    guint64 timestamp = buf->pts;  // PTS in nanoseconds

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
        NvDsFrameMeta *fmeta = (NvDsFrameMeta*)l_frame->data;
        guint src_id = fmeta->source_id;
        guint frame_id = fmeta->frame_num;
        
        int obj_count = 0;
        for (NvDsMetaList *l_obj = fmeta->obj_meta_list; l_obj; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta*)l_obj->data;
            obj_count++;

            guint track_id = obj_meta->object_id;    // tracker ID
            guint class_id = obj_meta->class_id;     // detector class
            float conf = obj_meta->confidence;
            float left = obj_meta->rect_params.left;
            float top = obj_meta->rect_params.top;
            float width = obj_meta->rect_params.width;
            float height = obj_meta->rect_params.height;

			// Convert timestamp from nanoseconds → milliseconds
            double ts_ms = (double)timestamp / 1e6;

            // ---- PRINT TO STDOUT (CSV format) ----
            printf("%.3f,%u,%u,%.2f,%.2f,%.2f,%.2f\n",
                   ts_ms, track_id, class_id, left, top, width, height);
            fflush(stdout);
        }

		std::cerr << "[PROBE] frame=" << fmeta->frame_num 
				  << " src=" << fmeta->source_id
				  << " obj_count=" << obj_count
				  << " batch_meta=" << (batch_meta ? "OK" : "NULL") << "\n";

        // ---- PRINT TO STDERR (log format) ----
        //std::cerr << "[PROBE][" << (char*)user_data << "] frame#[" << frame_id 
        //          << "] total_objs=[" << obj_count << "]\n";
        std::cerr.flush();
    }
    return GST_PAD_PROBE_OK;
}

// OSD sink pad buffer probe (optional, for debugging)
inline GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad *pad, 
                                                    GstPadProbeInfo *info, 
                                                    gpointer user_data) {
    GstBuffer *buf = (GstBuffer *)info->data;
    if (!buf) {
    	std::cerr << "[PROBE] no buffer\n";
    	return GST_PAD_PROBE_OK;
    }
    
    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) {
        g_printerr("[PROBE] batch_meta == NULL on OSD sink pad\n");
        return GST_PAD_PROBE_OK;
    }
    
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; 
		l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        int total_objs = 0;
        
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; 
             l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
            total_objs++;
            //if (obj_meta->class_id == 0) person_count++;
            //if (obj_meta->class_id == 2) vehicle_count++;
            
            std::cerr << "[PROBE] Frame " << frame_meta->frame_num
                      << " src_id=" << frame_meta->source_id
                      << " objID=" << obj_meta->object_id
                      << " class=" << obj_meta->class_id
                      << " conf=" << obj_meta->confidence
                      << " bbox=(" << obj_meta->rect_params.left 
                      << "," << obj_meta->rect_params.top
                      << "," << obj_meta->rect_params.width 
                      << "x" << obj_meta->rect_params.height << ")\n";
        }
        std::cerr << "[PROBE] Frame " << frame_meta->frame_num 
                  << " total_objs=" << total_objs << "\n";
    }
    return GST_PAD_PROBE_OK;
}

// Dynamic pad handler
inline void pad_added_handler(GstElement *src, GstPad *new_pad, gpointer user_data) {
    GstElement *next_element = (GstElement *)user_data;
    static gboolean linked = FALSE;

    if (linked) {
    	// Why needed? Because uridecodebin might create multiple pads:
		// Video pad → We want to link this one
		// Audio pad → Ignore this one
		// Subtitle pad → Ignore this one
        g_printerr("[PAD] Pad already linked, ignoring extra pads\n");
        return;
    }
    
    // Check if this is a mux element (needs request pad) or regular element (has static pad)
    GstPad *sink_pad = NULL;
    if (GST_IS_ELEMENT(next_element) && gst_element_get_pad_template(next_element, "sink_0")) {
    	// For mux elements, request a pad
        sink_pad = gst_element_get_request_pad(next_element, "sink_0");
        if (!sink_pad) {
            g_printerr("[PAD] Failed to get request pad sink_0 from %s\n", 
                      GST_ELEMENT_NAME(next_element));
            return;
        }
    } else {
    	// For regular elements, get the static sink pad
        sink_pad = gst_element_get_static_pad(next_element, "sink");
        if (!sink_pad) {
            g_printerr("[PAD] Failed to get static sink pad from %s\n", 
                      GST_ELEMENT_NAME(next_element));
            return;
        }
    }

    GstPadLinkReturn ret = gst_pad_link(new_pad, sink_pad);
    if (ret != GST_PAD_LINK_OK) {
        g_printerr("[PAD] Failed to link source pad to %s: %d\n", 
                  GST_ELEMENT_NAME(next_element), ret);
    } else {
        linked = TRUE;
        g_printerr("[PAD] Source pad linked to %s successfully\n", 
                  GST_ELEMENT_NAME(next_element));
    }

    gst_object_unref(sink_pad);
}

// Bus callbacks
inline gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_ERROR: {
            GError *err;
            gchar *debug;
            gst_message_parse_error(msg, &err, &debug);
            std::cerr << "Error: " << err->message << "\n";
            if (debug) std::cerr << "Debug: " << debug << "\n";
            g_error_free(err);
            g_free(debug);
            break;
        }
        case GST_MESSAGE_WARNING: {
            GError *err;
            gchar *debug;
            gst_message_parse_warning(msg, &err, &debug);
            std::cerr << "Warning: " << err->message << "\n";
            if (debug) std::cerr << "Debug: " << debug << "\n";
            g_error_free(err);
            g_free(debug);
            break;
        }
        default:
            break;
    }
    return TRUE;
}

inline gboolean bus_error_callback(GstBus *bus, GstMessage *msg, gpointer data) {
    GError *err;
    gchar *debug;
    gst_message_parse_error(msg, &err, &debug);
    g_printerr("ERROR: %s\n", err->message);
    if (debug) g_printerr("DEBUG: %s\n", debug);
    g_error_free(err);
    g_free(debug);
    return FALSE;
}

inline gboolean bus_warning_callback(GstBus *bus, GstMessage *msg, gpointer data) {
    GError *err;
    gchar *debug;
    gst_message_parse_warning(msg, &err, &debug);
    g_printerr("WARNING: %s\n", err->message);
    if (debug) g_printerr("DEBUG: %s\n", debug);
    g_error_free(err);
    g_free(debug);
    return FALSE;
}

// Compute Averaged bounding box from history
NvOSD_RectParams compute_averaged_bbox(const std::deque<NvOSD_RectParams>& boxes) {

	if (boxes.empty()) {
        return NvOSD_RectParams{0, 0, 0, 0};
    }
    
	NvOSD_RectParams avg;
	avg.left = boxes.back().left; // Use the LATEST position
    avg.top = boxes.back().top;
    avg.width = 0;
    avg.height = 0;
    
    for (const auto& box : boxes) {
    	avg.width += box.width;
    	avg.height += box.height;
    }
    avg.width /= boxes.size();
    avg.height /= boxes.size();

	return avg;
}

// Compute Exponential Moving Average bounding box from history
NvOSD_RectParams compute_expo_averaged_bbox(const std::deque<NvOSD_RectParams>& boxes) {
    if (boxes.empty()) {
        return NvOSD_RectParams{0, 0, 0, 0};
    }
    
    // Give much more weight to recent frames
    float total_weight = 0;
    NvOSD_RectParams avg = {0, 0, 0, 0};
    
    for (size_t i = 0; i < boxes.size(); i++) {
        // Exponential weighting: most recent gets MUCH more weight
        float weight = pow(2.0f, (float)i);  // 1, 2, 4, 8, 16...
        
        avg.left += boxes[i].left * weight;
        avg.top += boxes[i].top * weight;
        avg.width += boxes[i].width * weight;
        avg.height += boxes[i].height * weight;
        
        total_weight += weight;
    }
    
    avg.left /= total_weight;
    avg.top /= total_weight;
    avg.width /= total_weight;
    avg.height /= total_weight;
    
    return avg;
}

// Filtered display probe - only show tracked objects
static GstPadProbeReturn filtered_display_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {

	static int frame_debug = 0;
	frame_debug++;
	if (frame_debug % 30 == 0)
		g_print("Frame %d processed\n", g_current_frame_num);
	
	GstBuffer *buf = (GstBuffer *)info->data;
	NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
	if (!batch_meta) {
        return GST_PAD_PROBE_OK;
    }
    
    std::lock_guard<std::mutex> lock(g_history_mutex);
    
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next){
    	NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
    	g_current_frame_num = frame_meta->frame_num;
    	
    	// FIX #4: Only predict for tracks that have been seen at least once
    	// (frame_count > 0). Calling predict() on a default-constructed
    	// KalmanFilter before any update() has been called reads uninitialized
    	// state matrices, producing garbage bboxes or a crash that silently
    	// kills the pipeline.
    	for (auto& pair : g_object_histories) {
    		if (pair.second.frame_count > 0) {
    			pair.second.kf.predict();
    		}
    	}
    	
    	// Update histories with current detections
    	// First pass, set all visible objects as TRUE and update
    	for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next){
    		NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
    		
    		//if (obj_meta->object_id == UNTRACKED_OBJECT_ID)
    		//	continue;
    			
    		guint64 obj_id = obj_meta->object_id;
    		BBoxHistory &history = g_object_histories[obj_id];
    		
    		// Update Kalman filter with new measurement
    		history.kf.update(obj_meta->rect_params);
    		
    		history.boxes.push_back(obj_meta->rect_params);
    		history.frame_count++;
    		history.currently_visible = true;
    		history.last_seen_frame = frame_meta->frame_num;
    		
    		// Calculate velocity
/*    		if (history.boxes.size() >= 2) {
    			auto& prev = history.boxes[history.boxes.size() - 2];
    			auto& curr = history.boxes.back();
    			float dx = curr.left - prev.left;
    			float dy = curr.top - prev.top;
    			float velocity = sqrt(dx*dx + dy*dy);
    			history.avg_velocity = history.avg_velocity * 0.7f + velocity * 0.3f; // EMA
    		} */
    		
    		// Keep only recent boxes for smoothing
    		if (history.boxes.size() > BBOX_SMOOTHING_WINDOW) {
    			history.boxes.pop_front();
    		}
    	}
    	
    	// Filter and smooth display
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL;) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
            NvDsMetaList *next = l_obj->next;
            
            guint64 obj_id = obj_meta->object_id;
            BBoxHistory &history = g_object_histories[obj_id];
            
            if (false) {  //(history.frame_count < MIN_FRAMES_TO_DISPLAY) {
                // Don't display this object
                nvds_remove_obj_meta_from_frame(frame_meta, obj_meta);
            } else {
				// FIX #4: Only apply Kalman bbox if the filter has been
				// properly initialized (frame_count > 0 guarantees at least
				// one update() call has been made before getBBox() is used).
				if (history.frame_count > 0) {
					auto bbox = history.kf.getBBox();
					if (bbox.width > 0 && bbox.height > 0) {
						obj_meta->rect_params = bbox;
					} else {
						g_print("Invalid Kalman bbox for obj %lu, keeping raw detection\n",
						        (unsigned long)obj_id);
					}
				}
				obj_meta->rect_params = history.kf.getBBox();
				//NvOSD_RectParams filtered_bbox = history.kf.getBBox();
				
				// Apply to metadata
				//obj_meta->rect_params.left = filtered_bbox.left;
				//obj_meta->rect_params.top = filtered_bbox.top;
				//obj_meta->rect_params.width = filtered_bbox.width;
				//obj_meta->rect_params.height = filtered_bbox.height;
				
				// CRITICAL: Set border properties AFTER updating coordinates
				//obj_meta->rect_params.border_width = 4;
				//obj_meta->rect_params.border_color.red = 0.0;
				//obj_meta->rect_params.border_color.green = 1.0;
				//obj_meta->rect_params.border_color.blue = 0.0;
				//obj_meta->rect_params.border_color.alpha = 1.0;
						    
                //std::cerr << "[KALMAN FILTER] left=" << filtered_bbox.left
                //		  << " top="  << filtered_bbox.top
                //		  << " width="   << filtered_bbox.width
                //		  << " height="   << filtered_bbox.height << "\n";
            }
            
            l_obj = next;
        }
        
        // Clean up old tracks
        for (auto it = g_object_histories.begin(); it != g_object_histories.end();) {			
            if (it->second.kf.getAge() >= FRAMES_BEFORE_REMOVAL) {
                it = g_object_histories.erase(it);
            } else {
                ++it;
            }
		}
    }
    
    return GST_PAD_PROBE_OK;
}
