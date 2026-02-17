
/*

	A GstInterpipe implementation of track.cpp.
	
	Tentative architecture:

	┌──────────────────────────────────────────────────────────────────┐
	│                    Input Pipelines (Independent)                 │
	├──────────────────────────────────────────────────────────────────┤
	│ Pipeline 1: uridecodebin → nvvideoconvert → capsfilter →         │
	│             interpipesink name=source1                           │
	│                                                                  │
	│ Pipeline 2: v4l2src → nvvideoconvert → capsfilter →              │
	│             interpipesink name=source2                           │
	│                                                                  │
	│ Pipeline 3: rtspsrc → rtph264depay → h264parse → nvv4l2decoder → │
	│             nvvideoconvert → capsfilter → interpipesink name=rtsp│
	└──────────────────────────────────────────────────────────────────┘

	┌──────────────────────────────────────────────────────────────────┐
	│                   Processing Pipeline (Dynamic)                  │
	├──────────────────────────────────────────────────────────────────┤
	│ interpipesrc listen-to=source1 → nvstreammux → nvinfer →         │
	│ nvtracker → nvvideoconvert → queue → interpipesink name=output   │
	└──────────────────────────────────────────────────────────────────┘

	┌──────────────────────────────────────────────────────────────────┐
	│                    Output Pipelines (Multiple)                   │
	├──────────────────────────────────────────────────────────────────┤
	│ Pipeline 1: interpipesrc listen-to=output → nvdsosd →            │
	│             nveglglessink                                        │
	│                                                                  │
	│ Pipeline 2: interpipesrc listen-to=output → nvv4l2h264enc →      │
	│             h264parse → mp4mux → filesink                        │
	│                                                                  │
	│ Pipeline 3: interpipesrc listen-to=output → appsink              │
	│             (for metadata extraction)                            │
	└──────────────────────────────────────────────────────────────────┘
	
*/

#pragma once

#include <gst/gst.h>
#include <string>
#include <map>


class InterpipeTracker {
private:
	std::map<std::string, GstElement*> pipelines;
	std::string current_source;
	
public:
	
	// Create a pipeline with description
	GstElement* create_pipeline(const std::string& pipe_desc) {
		GError* error = nullptr;
		GstElement* pipeline = gst_parse_launch(pipe_desc.c_str(), &error);
		if (error) {
			g_printerr("Failed to create input pipeline: %s\n", error->message);
			g_error_free(error);
			return nullptr;
		}
		return pipeline;
	}
	
	// Create input pipeline for file source
	GstElement* create_file_input_pipeline(const std::string& uri, 
                                           const std::string& sink_name) {
                                           
		std::string pipe_desc = 
            "rtspsrc location=" + location + " latency=100 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvideoconvert ! "
            "video/x-raw(memory:NVMM),format=NV12,width=800,height=608 ! "
            "queue leaky=2 max-size-buffers=10 ! "
            "interpipesink name=" + sink_name + " sync=false async=false";
		
		return create_pipeline(pipe_desc);
	}

	// Create RTSP input pipeline
    GstElement* create_rtsp_input_pipeline(const std::string& location,
                                           const std::string& sink_name) {
        std::string pipe_desc = 
            "rtspsrc location=" + location + " latency=100 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvideoconvert ! "
            "video/x-raw(memory:NVMM),format=NV12,width=800,height=608 ! "
            "queue leaky=2 max-size-buffers=10 ! "
            "interpipesink name=" + sink_name + " sync=false async=false";
            
		return create_pipeline(pipe_desc);
	}
	
	// Create main processing pipeline
	GstElement* create_processing_pipeline(const std::string& config,
										   const std::string& initial_source) {
		std::string pipe_desc = 
            "interpipesrc listen-to=" + initial_source + " "
            "stream-sync=2 allow-renegotiation=false "  // stream-sync=2 for compensation
            "caps=\"video/x-raw(memory:NVMM),format=NV12,width=800,height=608\" ! "
            "nvstreammux name=mux width=800 height=608 batch-size=1 "
            "batched-push-timeout=40000 ! "
            "nvinfer name=pgie config-file-path=" + config + " ! "
            "nvtracker name=tracker tracker-width=800 tracker-height=608 "
            "ll-lib-file=/opt/nvidia/deepstream/deepstream-6.0/lib/libnvds_nvmultiobjecttracker.so "
            "ll-config-file=/opt/nvidia/deepstream/deepstream-6.0/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml ! "
            "nvvideoconvert ! "
            "queue leaky=2 max-size-buffers=10 ! "
            "interpipesink name=processing_output sync=false async=false";
            
		current_source = initial_source;
		return create_pipeline(pipe_desc);
	}
	
	// Create display output pipeline
    GstElement* create_display_pipeline() {
    	std::string pipe_desc = 
            "interpipesrc listen-to=processing_output stream-sync=0 "  // stream-sync=0 for display
            "allow-renegotiation=false ! "
            "nvdsosd ! "
            "nveglglessink sync=false async=false";
            
    	return create_pipeline(pipe_desc);
    }
    
    // Create metadata extraction pipeline
    GstElement* create_metadata_pipeline() {
    	std::string pipe_desc = 
            "interpipesrc listen-to=processing_output stream-sync=2 "
            "allow-renegotiation=false ! "
            "appsink name=meta_sink emit-signals=true sync=false";
            
		GstElement* pipeline = create_pipeline(pipe_desc);
		
		// Connect to appsink signal for metadata extraction
		GstElement* appsink = gst_bin_get_by_name(GST_BIN(pipeline), "meta_sink");
        if (appsink) {
        	g_signal_connect(appsink, "new-sample", G_CALLBACK(metadata_callback), this);
        	gst_object_unref(appsink);
        }
        
        return pipeline;
    }
    
    // Switch input source dynamically
    bool switch_source(const std::string& new_source) {
    	GstElement* processing_pipe = pipelines["processing"];
        if (!processing_pipe) {
            g_printerr("Processing pipeline not found\n");
            return false;
        }
        
        GstElement* interpipesrc = gst_bin_get_by_name(GST_BIN(processing_pipe), "interpipesrc0");
        
        if (interpipesrc) {
            g_object_set(G_OBJECT(interpipesrc), "listen-to", new_source.c_str(), NULL);
            current_source = new_source;
            gst_object_unref(interpipesrc);
            return true;
        }
        
        return false;
    
    }
    
    // Add pipeline to manager
    void add_pipeline(const std::string& name, GstElement* pipeline) {
        pipelines[name] = pipeline;
    }
    
    // Start all pipelines
    void start_all() {
    	for (auto& [name, pipeline] : pipelines) {
    		g_print("Starting pipeline: %s\n", name.c_str());
    		gst_element_set_state(pipeline, GST_STATE_PLAYING);
    	}
    }
    
	// Stop all pipelines
    void stop_all() {
    	for (auto& [name, pipeline] : pipelines) {
    		g_print("Stoping pipeline: %s\n", name.c_str());
    		gst_element_set_state(pipeline, GST_STATE_NULL);
    		gst_object_unref(pipeline);
    	}
    	pipelines.clear();
    }
    
private:

	static GstFlowReturn metadata_callback(GstElement* sink, gpointer user_data) {
	
		GstSample* sample = nullptr;
		g_signal_emit_by_name(sink, "pull-sample", &sample);
		
		if (sample) {
			GstBuffer* buffer = gst_sample_get_buffer(sample);
			
			// Extract DeepStream metadata
			NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
			if (batch_meta) {
			
				// Process metadata here
			
			}
			
			gst_sample_unref(sample);
		}
		
		return GST_FLOW_OK;
	
	}
      
};












