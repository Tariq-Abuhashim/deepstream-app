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

pipeline:
uridecodebin → nvstreammux → nvinfer (DETR) → nvtracker → nvvideoconvert → nvdsosd → nveglglessink

run:
./build/deepstream_detr_app file:///home/mrt/src/ByteTrack/videos/

June, 19, 2025
*/


#include <gst/gst.h>
#include <nvdsmeta.h>
#include <gstnvdsmeta.h>
#include <nvds_meta.h>  // Core metadata definitions

#include <glib.h>
#include <iostream>
#include <string>
#include <cstring>

//#include <nvds_meta.h>
#include <nvds_infer.h>
#include <nvdsinfer_custom_impl.h>
#include <nvds_tracker_meta.h>

#define PGIE_CONFIG_FILE "config_infer_primary_detr.txt"
//#define TRACKER_CONFIG_FILE "tracker_config.txt"

static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) {
        g_print("Warning: Failed to get batch meta from buffer\n");
        return GST_PAD_PROBE_OK;
    }
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        int person_count = 0;
        int vehicle_count = 0;
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
            if (obj_meta->class_id == 0) person_count++;
            if (obj_meta->class_id == 2) vehicle_count++;
            std::cout << "Object ID: "   << obj_meta->object_id 
                      << " Class ID: "   << obj_meta->class_id 
                      << " Confidence: " << obj_meta->confidence 
                      << std::endl;
        }
        std::cout << "Frame " << frame_meta->frame_num
                  << " | Persons: " << person_count
                  << " | Vehicles: " << vehicle_count << std::endl;
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


int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <uri>" << std::endl;
        return -1;
    }
    
	/* Initializes the GStreamer library
    */
    gst_init(&argc, &argv);

	/* Element factory 
    Elements: source, streammux, pgie, tracker, nvvidconv, nvosd and sink 
    */
    GstElement *pipeline = gst_pipeline_new("deepstream-pipeline");
    GstElement *source = gst_element_factory_make("uridecodebin", "src");
    GstElement *streammux = gst_element_factory_make("nvstreammux", "mux");
    GstElement *pgie = gst_element_factory_make("nvinfer", "pgie");
    GstElement *tracker = gst_element_factory_make("nvtracker", "tracker");
    GstElement *nvvidconv = gst_element_factory_make("nvvideoconvert", "convert");
    GstElement *nvosd = gst_element_factory_make("nvdsosd", "osd");
    GstElement *sink = gst_element_factory_make("nveglglessink", "sink");

    if (!pipeline || !source || !streammux || !pgie || !tracker || !nvvidconv || !nvosd || !sink) {
        std::cerr << "One element could not be created. Exiting." << std::endl;
        return -1;
    }

	/* GObject Property sitter 
	*/
    g_object_set(G_OBJECT(source), "uri", argv[1], NULL);
    g_object_set(G_OBJECT(pgie), "config-file-path", PGIE_CONFIG_FILE, NULL);
    g_object_set(G_OBJECT(tracker),
             "tracker-width", 640,
             "tracker-height", 384,
             "ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
             "ll-config-file", "/opt/nvidia/deepstream/deepstream-6.0/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml",
             "enable-batch-process", TRUE,
             "gpu-id", 0,
             NULL);
    g_object_set(G_OBJECT(streammux), "width", 1280, "height", 720, "batch-size", 1, "batched-push-timeout", 40000, NULL);
    g_object_set(G_OBJECT(sink), "sync", FALSE, NULL);

	/* Add multiple elements into a GstBin at once
    */
    gst_bin_add_many(GST_BIN(pipeline), source, streammux, pgie, tracker, nvvidconv, nvosd, sink, NULL);

	/* Hanble dynamic pads
    */
    g_signal_connect(source, "pad-added", G_CALLBACK(pad_added_handler), streammux);

	/* Attempts to link each element’s src pad to the next element’s sink pad in order
    streammux → pgie → tracker → nvvidconv → nvosd → sink
    Pads must exist already.
    Elements wth static pads, src and sink pads exist after creation.
    Elements with dynamic pads, src and sink pads must be created first. 
    */
    if (!gst_element_link_many(streammux, pgie, tracker, nvvidconv, nvosd, sink, NULL)) {
        std::cerr << "Elements could not be linked." << std::endl;
        return -1;
    }

	/* Adding a probe to a GStreamer pad to inspect or modify buffers, Examples:
    Getting the NvDsBatchMeta (DeepStream metadata)
    Drawing on frames
    Logging statistics
    */
    GstPad *osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, osd_sink_pad_buffer_probe, NULL, NULL);
    gst_object_unref(osd_sink_pad);

	/* set your pipeline to PLAYING state.
    */
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    std::cout << "Running DeepStream pipeline..." << std::endl;

	/* The GstBus is where the pipeline posts messages
    gst_bus_timed_pop_filtered blocks forever (GST_CLOCK_TIME_NONE) until:
    An ERROR message arrives
    An EOS (end-of-stream) message arrives
    */
    GstBus *bus = gst_element_get_bus(pipeline);
    GstMessage *msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
                                                 (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    if (msg != NULL) {
        GError *err;
        gchar *debug_info;
        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_ERROR:
                gst_message_parse_error(msg, &err, &debug_info);
                std::cerr << "Error received: " << err->message << std::endl;
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
    gst_object_unref(pipeline); // free pipeline
    gst_object_unref(bus); // free bus

    return 0;
}

