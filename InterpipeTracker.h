/*

	A GstInterpipe implementation of track.cpp.
	
	Architecture:

	Input:      filesrc → qtdemux → h264parse → avdec_h264 →
	            nvvideoconvert → interpipesink name=video_source

	Processing: interpipesrc listen-to=video_source →
	            nvstreammux → nvinfer → nvtracker → nvvideoconvert → queue →
	            interpipesink name=processing_output

	Output:     interpipesrc listen-to=processing_output →
	            nvdsosd → nveglglessink   (display)
	            OR fakesink               (headless)
	            OR nvv4l2h264enc → ...    (recorder)
	            OR appsink                (metadata)

*/

#pragma once

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <nvdsmeta.h>
#include <gstnvdsmeta.h>
#include <string>
#include <map>
#include <vector>
#include <iostream>

class InterpipeTracker {
private:
	std::map<std::string, GstElement*> pipelines;
	std::string current_source;
    int infer_width  = 800;
    int infer_height = 608;
    int batch_size   = 1;
	
public:
    
    void set_config(int width, int height, int batch = 1) {
        infer_width  = width;
        infer_height = height;
        batch_size   = batch;
    }
	
	GstElement* create_pipeline(const std::string& pipe_desc) {
		GError* error = nullptr;
		GstElement* pipeline = gst_parse_launch(pipe_desc.c_str(), &error);
		if (error) {
			g_printerr("Failed to create pipeline: %s\n", error->message);
			g_error_free(error);
			return nullptr;
		}
		return pipeline;
	}
	
	// File input pipeline.
	// Uses uridecodebin (auto-selects best decoder: nvv4l2decoder on Jetson,
	// hardware NVDEC on desktop). Output is NvBufSurface NVMM, exactly what
	// nvstreammux needs.
    struct InputPadData {
        GstElement* conv_pre;
        bool linked;
    };

	GstElement* create_file_input_pipeline(const std::string& uri,
                                           const std::string& sink_name) {
        GstElement* pipeline = gst_pipeline_new("input-pipeline");
        GstElement* source   = gst_element_factory_make("uridecodebin",   "source");
        GstElement* conv_pre = gst_element_factory_make("nvvideoconvert", "nvvidconv_pre");
        GstElement* caps_pre = gst_element_factory_make("capsfilter",     "capsfilter_pre");
        GstElement* queue    = gst_element_factory_make("queue",          "queue");
        GstElement* isink    = gst_element_factory_make("interpipesink",  sink_name.c_str());

        if (!pipeline || !source || !conv_pre || !caps_pre || !queue || !isink) {
            g_printerr("Failed to create input pipeline elements\n");
            if (pipeline) gst_object_unref(pipeline);
            return nullptr;
        }

        g_object_set(G_OBJECT(source), "uri", uri.c_str(), NULL);

        gst_bin_add_many(GST_BIN(pipeline),
            source, conv_pre, caps_pre, queue, isink, NULL);

        GstCaps* caps = gst_caps_from_string(
            "video/x-raw(memory:NVMM), format=NV12, memory:NVBufSurface");
        if (caps) {
            gst_caps_set_simple(caps,
                "width",  G_TYPE_INT, infer_width,
                "height", G_TYPE_INT, infer_height,
                NULL);
        } else {
            caps = gst_caps_new_simple("video/x-raw",
                "format", G_TYPE_STRING, "NV12",
                "width",  G_TYPE_INT,    infer_width,
                "height", G_TYPE_INT,    infer_height,
                NULL);
            GstCapsFeatures* f = gst_caps_features_new("memory:NVMM", NULL);
            gst_caps_set_features(caps, 0, f);
        }
        g_object_set(G_OBJECT(caps_pre), "caps", caps, NULL);
        gst_caps_unref(caps);

        g_object_set(G_OBJECT(queue),
            "leaky",            2,
            "max-size-buffers", 10,
            "max-size-bytes",   0,
            "max-size-time",    0,
            NULL);
        g_object_set(G_OBJECT(isink),
            "sync",        FALSE,
            "async",       FALSE,
            "forward-eos", TRUE,
            NULL);

        // conv_pre -> caps_pre -> queue -> isink are static -- link them now
        if (!gst_element_link_many(conv_pre, caps_pre, queue, isink, NULL)) {
            g_printerr("Failed to link nvvideoconvert -> capsfilter -> queue -> interpipesink\n");
            gst_object_unref(pipeline); return nullptr;
        }

        // uridecodebin has dynamic pads -- connect conv_pre when video pad appears
        auto* pad_data = new InputPadData{conv_pre, false};
        g_signal_connect(source, "pad-added",
            G_CALLBACK(+[](GstElement*, GstPad* pad, gpointer user_data) {
                auto* d = static_cast<InputPadData*>(user_data);
                if (d->linked) return;

                GstCaps* pcaps = gst_pad_get_current_caps(pad);
                if (!pcaps) pcaps = gst_pad_query_caps(pad, NULL);
                GstStructure* s = gst_caps_get_structure(pcaps, 0);
                bool is_video = g_str_has_prefix(gst_structure_get_name(s), "video/");
                gst_caps_unref(pcaps);
                if (!is_video) return;

                GstPad* sink = gst_element_get_static_pad(d->conv_pre, "sink");
                if (sink && !gst_pad_is_linked(sink)) {
                    if (gst_pad_link(pad, sink) == GST_PAD_LINK_OK) {
                        d->linked = true;
                        g_print("[PAD] uridecodebin -> nvvideoconvert linked\n");
                    } else {
                        g_printerr("[PAD] Failed to link uridecodebin -> nvvideoconvert\n");
                    }
                }
                if (sink) gst_object_unref(sink);
            }), pad_data);

        g_print("Created file input pipeline: %s -> %s\n", uri.c_str(), sink_name.c_str());
        return pipeline;
	}

	GstElement* create_rtsp_input_pipeline(const std::string& location,
                                           const std::string& sink_name) {
        std::string pipe_desc = 
            "rtspsrc location=" + location + " latency=100 protocols=tcp ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "nvvideoconvert ! "
            "video/x-raw(memory:NVMM),format=NV12"
            ",width="  + std::to_string(infer_width)  +
            ",height=" + std::to_string(infer_height) + " ! "
            "queue leaky=2 max-size-buffers=10 ! "
            "interpipesink name=" + sink_name +
            " sync=false async=false forward-eos=true";
        GstElement* pipeline = create_pipeline(pipe_desc);
        if (pipeline)
            g_print("Created RTSP input pipeline: %s → %s\n", location.c_str(), sink_name.c_str());
        return pipeline;
    }

    GstElement* create_camera_input_pipeline(const std::string& device,
                                             const std::string& sink_name) {
        std::string pipe_desc = 
            "v4l2src device=" + device + " ! "
            "video/x-raw,width=1920,height=1080,framerate=30/1 ! "
            "nvvideoconvert ! "
            "video/x-raw(memory:NVMM),format=NV12"
            ",width="  + std::to_string(infer_width)  +
            ",height=" + std::to_string(infer_height) + " ! "
            "queue leaky=2 max-size-buffers=10 ! "
            "interpipesink name=" + sink_name +
            " sync=false async=false forward-eos=true";
        GstElement* pipeline = create_pipeline(pipe_desc);
        if (pipeline)
            g_print("Created camera input pipeline: %s → %s\n", device.c_str(), sink_name.c_str());
        return pipeline;
    }
	
	GstElement* create_processing_pipeline(const std::string& config,
                                           const std::string& initial_source,
                                           const std::string& tracker_lib,
                                           const std::string& tracker_config,
                                           int tracker_width,
                                           int tracker_height) {
        GstElement* pipeline = gst_pipeline_new("processing-pipeline");
        if (!pipeline) { g_printerr("Failed to create processing pipeline\n"); return nullptr; }
        
        GstElement* src     = gst_element_factory_make("interpipesrc",   "interpipesrc");
        GstElement* mux     = gst_element_factory_make("nvstreammux",    "mux");
        GstElement* pgie    = gst_element_factory_make("nvinfer",        "pgie");
        GstElement* tracker = gst_element_factory_make("nvtracker",      "tracker");
        GstElement* conv    = gst_element_factory_make("nvvideoconvert", "nvvidconv");
        GstElement* queue   = gst_element_factory_make("queue",          "queue");
        GstElement* sink    = gst_element_factory_make("interpipesink",  "processing_output");
        
        if (!src || !mux || !pgie || !tracker || !conv || !queue || !sink) {
            g_printerr("Failed to create one or more processing pipeline elements\n");
            gst_object_unref(pipeline);
            return nullptr;
        }

        gst_bin_add_many(GST_BIN(pipeline), src, mux, pgie, tracker, queue, sink, NULL);
        
        // FIX #3: Set explicit caps on interpipesrc so it matches the emitter
        // exactly. Without this, interpipesrc may silently drop buffers if caps
        // negotiation fails (e.g. framerate or colorimetry mismatch), resulting
        // in no data reaching nvstreammux and batch_meta always being NULL.
        GstCaps* src_caps = gst_caps_new_simple("video/x-raw",
            "format", G_TYPE_STRING, "NV12",
            "width",  G_TYPE_INT,    infer_width,
            "height", G_TYPE_INT,    infer_height,
            NULL);
        GstCapsFeatures* f = gst_caps_features_new("memory:NVMM", NULL);
        gst_caps_set_features(src_caps, 0, f);

        g_object_set(G_OBJECT(src),
            "listen-to",           initial_source.c_str(),
            "is-live",             TRUE,
            "stream-sync",         0,     // pass buffers without sync
            "allow-renegotiation", TRUE,
            "caps",                src_caps,   // FIX #3: explicit caps
            NULL);
        gst_caps_unref(src_caps);

        // live-source=TRUE: required when input arrives via interpipe.
        // Without it nvstreammux waits for a live clock that never arrives.
        g_object_set(G_OBJECT(mux),
            "width",                infer_width,
            "height",               infer_height,
            "batch-size",           batch_size,
            "batched-push-timeout", 40000,
            "input-meta-queue",		FALSE,
			"live-source",          TRUE,
            "gpu-id",               0,
            NULL);
        
        g_object_set(G_OBJECT(pgie),
            "config-file-path", config.c_str(),
            "batch-size",       batch_size,
            NULL);
        
        g_object_set(G_OBJECT(tracker),
            "tracker-width",        tracker_width,
            "tracker-height",       tracker_height,
            "ll-lib-file",          tracker_lib.c_str(),
            "ll-config-file",       tracker_config.c_str(),
            "enable-batch-process", TRUE,
            "gpu-id",               0,
            "enable-past-frame",    TRUE,
            NULL);
        
        g_object_set(G_OBJECT(queue),
            "leaky",            2,
            "max-size-buffers", 10,
            "max-size-bytes",   0,
            "max-size-time",    0,
            NULL);
        
        // forward-eos=TRUE: propagate EOS to all listener pipelines when source ends.
        g_object_set(G_OBJECT(sink),
            "sync",        FALSE,
            "async",       FALSE,
            "forward-eos", TRUE,
            NULL);
        
        // interpipesrc has a STATIC src pad — link directly to nvstreammux.
        // The input pipeline already emits NvBufSurface-backed NVMM buffers
        // so no conversion is needed here.
        GstPad* src_pad      = gst_element_get_static_pad(src, "src");
        GstPad* mux_sink_pad = gst_element_get_request_pad(mux, "sink_0");

        if (!src_pad || !mux_sink_pad) {
            g_printerr("Failed to get pads for interpipesrc -> nvstreammux\n");
            if (src_pad)      gst_object_unref(src_pad);
            if (mux_sink_pad) gst_object_unref(mux_sink_pad);
            gst_object_unref(pipeline);
            return nullptr;
        }

        if (gst_pad_link(src_pad, mux_sink_pad) != GST_PAD_LINK_OK) {
            g_printerr("Failed to link interpipesrc -> nvstreammux\n");
            gst_object_unref(src_pad);
            gst_object_unref(mux_sink_pad);
            gst_object_unref(pipeline);
            return nullptr;
        }
        g_print("[LINK] interpipesrc -> nvstreammux sink_0 linked\n");

        // Probe to confirm buffers are flowing into the mux
        gst_pad_add_probe(mux_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
            [](GstPad*, GstPadProbeInfo*, gpointer) -> GstPadProbeReturn {
                static int count = 0;
                if (++count % 30 == 0)
                    g_print("[PROBE] mux sink_0: %d buffers received\n", count);
                return GST_PAD_PROBE_OK;
            }, NULL, NULL);

        gst_object_unref(src_pad);
        gst_object_unref(mux_sink_pad);

        if (!gst_element_link_many(mux, pgie, tracker, queue, sink, NULL)) {
            g_printerr("Failed to link mux -> pgie -> tracker -> queue -> sink\n");
            gst_object_unref(pipeline);
            return nullptr;
        }
        
		current_source = initial_source;
		g_print("Created processing pipeline: %s → processing_output\n", initial_source.c_str());
		g_print("  Config: %s\n", config.c_str());
		g_print("  Tracker: %dx%d\n", tracker_width, tracker_height);
		return pipeline;
	}
	
    GstElement* create_display_pipeline(bool sync = false) {
    	std::string pipe_desc = 
            "interpipesrc name=src listen-to=processing_output"
            " is-live=true stream-sync=0 allow-renegotiation=true ! "
			"nvvideoconvert ! "
        	"video/x-raw(memory:NVMM),format=RGBA ! "
            "nvdsosd name=osd ! "
            "nveglglessink name=sink sync=" + std::string(sync ? "true" : "false") +
            " async=false";
    	GstElement* pipeline = create_pipeline(pipe_desc);
        if (pipeline) g_print("Created display pipeline: processing_output → screen\n");
        return pipeline;
    }

    GstElement* create_headless_pipeline() {
        std::string pipe_desc = 
            "interpipesrc name=src listen-to=processing_output"
            " is-live=true stream-sync=0 allow-renegotiation=true ! "
            "fakesink name=sink sync=false async=false";
        GstElement* pipeline = create_pipeline(pipe_desc);
        if (pipeline) g_print("Created headless pipeline: processing_output → fakesink\n");
        return pipeline;
    }

    GstElement* create_recorder_pipeline(const std::string& output_file,
                                         int bitrate = 8000000) {
        std::string pipe_desc = 
            "interpipesrc name=src listen-to=processing_output"
            " is-live=true stream-sync=0 allow-renegotiation=true ! "
            "nvvideoconvert ! "
            "nvv4l2h264enc name=encoder bitrate=" + std::to_string(bitrate) + " ! "
            "h264parse ! mp4mux ! "
            "filesink location=" + output_file + " sync=false";
        GstElement* pipeline = create_pipeline(pipe_desc);
        if (pipeline)
            g_print("Created recorder pipeline: processing_output → %s\n", output_file.c_str());
        return pipeline;
    }
    
    GstElement* create_metadata_pipeline(GCallback callback, gpointer user_data) {
    	std::string pipe_desc = 
            "interpipesrc name=src listen-to=processing_output"
            " is-live=true stream-sync=0 allow-renegotiation=true ! "
            "appsink name=meta_sink emit-signals=true max-buffers=1 drop=true sync=false";
		GstElement* pipeline = create_pipeline(pipe_desc);
		if (!pipeline) return nullptr;
		GstElement* appsink = gst_bin_get_by_name(GST_BIN(pipeline), "meta_sink");
        if (appsink) {
        	g_signal_connect(appsink, "new-sample", callback, user_data);
        	gst_object_unref(appsink);
			g_print("Created metadata pipeline: processing_output → appsink\n");
        }
        return pipeline;
    }
    
    bool switch_source(const std::string& new_source) {
    	GstElement* proc = pipelines["processing"];
        if (!proc) { g_printerr("Processing pipeline not found\n"); return false; }
        GstElement* isrc = gst_bin_get_by_name(GST_BIN(proc), "interpipesrc");
        if (isrc) {
            g_object_set(G_OBJECT(isrc), "listen-to", new_source.c_str(), NULL);
            current_source = new_source;
			g_print("Switched to source: %s\n", new_source.c_str());
            gst_object_unref(isrc);
            return true;
        }
		g_printerr("interpipesrc element not found in processing pipeline\n");
        return false;
    }
    
    void add_pipeline(const std::string& name, GstElement* pipeline) {
		if (pipeline) pipelines[name] = pipeline;
    }
    
    GstElement* get_pipeline(const std::string& name) {
		auto itr = pipelines.find(name);
		return (itr != pipelines.end()) ? itr->second : nullptr;
	}

    bool start_pipeline(const std::string& name) {
		GstElement* pipeline = get_pipeline(name);
		if (!pipeline) { g_printerr("Pipeline not found: %s\n", name.c_str()); return false; }
		GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
		if (ret == GST_STATE_CHANGE_FAILURE) {
			g_printerr("Failed to start pipeline: %s\n", name.c_str()); return false;
		}
		g_print("Started pipeline: %s\n", name.c_str());
		return true;
	}

    // Start in dependency order: input → processing → outputs.
    // Block on input and processing until confirmed PLAYING so that
    // interpipesrc finds a live emitter when it connects.
	void start_all() {
		// Start processing and output pipelines FIRST so interpipesrc is
		// already listening before the input pipeline begins emitting frames.
		// Then start input last.
		const std::vector<std::string> order = {
		    "processing", "display", "headless", "recorder", "metadata", "input"
		};
		for (const auto& name : order) {
		    auto it = pipelines.find(name);
		    if (it == pipelines.end()) continue;
		    GstStateChangeReturn ret = gst_element_set_state(it->second, GST_STATE_PLAYING);
		    if (ret == GST_STATE_CHANGE_FAILURE) {
		        g_printerr("Failed to start pipeline: %s\n", name.c_str()); continue;
		    }
		    g_print("Started pipeline: %s\n", name.c_str());
		    // Block until PLAYING is confirmed before proceeding to the next pipeline
		    if (name == "processing") {
		        GstState state;
		        GstStateChangeReturn sr = gst_element_get_state(
		            it->second, &state, nullptr, 10 * GST_SECOND);
		        g_print("  processing state: %s (change: %d)\n",
		                gst_element_state_get_name(state), sr);
		    }
		    // Small settle gap so interpipesrc completes its listen-to connection
		    // before the emitter starts pushing buffers
		    if (name == "processing") {
		        g_usleep(200000);  // 200ms
		    }
		}
	}
    
    // Stop in reverse order: outputs first, then processing, then inputs.
    // Disconnect each interpipesrc before going to NULL to avoid assertion errors.
    void stop_all() {
    	const std::vector<std::string> order = {
    		"display", "headless", "recorder", "metadata", "processing", "input"
    	};
    	auto stop_one = [&](const std::string& name) {
    		auto it = pipelines.find(name);
    		if (it == pipelines.end()) return;
    		// Detach listener from emitter before NULL transition
    		GstElement* isrc = gst_bin_get_by_name(GST_BIN(it->second), "src");
    		if (!isrc) isrc = gst_bin_get_by_name(GST_BIN(it->second), "interpipesrc");
    		if (isrc) {
    			g_object_set(G_OBJECT(isrc), "listen-to", "", NULL);
    			gst_object_unref(isrc);
    		}
    		g_print("Stopping pipeline: %s\n", name.c_str());
    		gst_element_set_state(it->second, GST_STATE_NULL);
    		gst_object_unref(it->second);
    		pipelines.erase(it);
    	};
    	for (const auto& name : order) stop_one(name);
    	for (auto& [name, pipeline] : pipelines) {
    		g_print("Stopping pipeline: %s\n", name.c_str());
    		gst_element_set_state(pipeline, GST_STATE_NULL);
    		gst_object_unref(pipeline);
    	}
    	pipelines.clear();
		g_print("All pipelines stopped and cleaned up\n");
    }

	void print_status() {
		g_print("\n=== InterpipeTracker Status ===\n");
		g_print("Pipelines: %zu\n", pipelines.size());
		for (auto& [name, pipeline] : pipelines) {
            GstState state, pending;
            gst_element_get_state(pipeline, &state, &pending, 0);
            g_print("  %s: %s\n", name.c_str(), gst_element_state_get_name(state));
        }
        g_print("Current source: %s\n", current_source.c_str());
        g_print("===============================\n\n");
	}
    
private:
	static GstFlowReturn metadata_callback(GstElement* sink, gpointer user_data) {
		GstSample* sample = nullptr;
		g_signal_emit_by_name(sink, "pull-sample", &sample);
		if (sample) {
			GstBuffer* buffer = gst_sample_get_buffer(sample);
			NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
			if (batch_meta) {
				for (NvDsMetaList* l_frame = batch_meta->frame_meta_list;
					l_frame != NULL; l_frame = l_frame->next) {
					NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)(l_frame->data);
					int n = 0;
					for (NvDsMetaList* l = frame_meta->obj_meta_list; l; l = l->next) n++;
					if (n > 0) g_print("Frame %d: %d objects\n", frame_meta->frame_num, n);
				}
			}
			gst_sample_unref(sample);
		}
		return GST_FLOW_OK;
	}
};
