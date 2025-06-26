#!/usr/bin/env python3

import sys
import os
import gi # GObject Introspection (used to access GStreamer's Python bindings).
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib

# Initialize GStreamer
Gst.init(None)

# This class encapsulates the entire video processing pipeline.
class DeepStreamApp:
    def __init__(self):
        self.config = {
            "input_video": "/home/dv/src/ByteTrack/videos/palace.mp4",
            "output_file": "output_people_detect.mkv",
            "yolo_engine": "/home/dv/src/yolov5/yolov5s.engine",
            "config_infer": "/home/dv/src/DeepStream-Yolo/config_infer_primary_yoloV5.txt",
            "tracker_config": "/opt/nvidia/deepstream/deepstream-6.3/sources/apps/sample_apps/deepstream-test2/dstest2_tracker_config.txt",
            "streammux": { # Batch processing settings (width, height, batch size, timeout).
                "width": 640,
                "height": 384,
                "batch_size": 1,
                "timeout": 4000000
            },
            "encoder": { # Video encoding settings (bitrate)
                "bitrate": 4000000 # 4 Mbps
            }
        }
        self.pipeline = None # Will hold the GStreamer pipeline
        self.loop = None # Main event loop (GLib)
        self.elements = {} # Stores GStreamer elements (source, demux, decoder, etc.)
        self.requested_pads = [] # Tracks dynamically requested pads (for linking elements)

    # Checks if all required files exist
    def validate_config(self):
        """Validate configuration paths"""
        required_files = [
            self.config["input_video"],
            self.config["yolo_engine"],
            self.config["config_infer"],
            self.config["tracker_config"]
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")

    # Creates a GStreamer element; sets properties if provided, adds the element to the pipeline
    def create_element(self, factory_name, element_name, properties=None):
        """Create and configure a GStreamer element"""
        element = Gst.ElementFactory.make(factory_name, element_name)
        if not element:
            raise RuntimeError(f"Unable to create {element_name} ({factory_name})")
        
        if properties:
            for prop, value in properties.items():
                if element.find_property(prop) is not None:
                    element.set_property(prop, value)
                else:
                    print(f"Warning: Property '{prop}' not found in {element_name}")
        
        self.pipeline.add(element)
        return element

    # Called when qtdemux (demuxer) dynamically creates a new pad (connection points for video/audio streams)
    # Checks if the pad carries video (name.startswith("video/"))
    # Links the demuxer's video pad to the h264parse element.
    def on_pad_added(self, demux, pad):
        """Handle dynamic pad from demuxer"""
        caps = pad.get_current_caps()
        if caps:
            struct = caps.get_structure(0)
            name = struct.get_name()
            if name.startswith("video/"): # if streams contain a video stream
                sink_pad = self.elements["parser"].get_static_pad("sink") # link the video stream to the next element
                if not sink_pad.is_linked():
                    pad.link(sink_pad)

    # Attached to the OSD (On-Screen Display) element to inspect frames.
    def osd_sink_pad_buffer_probe(self, pad, info):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        # Correct way to get batch metadata in DeepStream 6.0+
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        # Iterate through frames in the batch
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                # Contains frame-level info (frame number, timestamp)
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            print(f"Frame {frame_meta.frame_num} @ PTS {frame_meta.buf_pts / Gst.SECOND:.2f}s")
  
            # Iterate through objects in the frame
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    # Contains detected object info (class ID, confidence)
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                print(f"  Object class_id: {obj_meta.class_id}, confidence: {obj_meta.confidence:.2f}")

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    # Handles GStreamer bus messages
    def bus_call(self, bus, message):
        """Handle bus messages"""
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-Of-Stream reached.")
            self.loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
            self.loop.quit()
        return True

    def build_pipeline(self):
        """Build and link the GStreamer pipeline"""
        self.pipeline = Gst.Pipeline.new("deepstream-pipeline")
        self.loop = GLib.MainLoop()

        # Create elements
        self.elements = {
            "source": self.create_element("filesrc", "file-source", { # filesrc: Reads input video
                                          "location": self.config["input_video"]
                                          }),
            "demux": self.create_element("qtdemux", "demux"), # qtdemux: Demuxes video streams, (splits the video into video, audio, subtitles)
            "parser": self.create_element("h264parse", "parser"), # 	h264parse: Parses H.264 video
                                                 # ideo streams in containers (like MP4) may have headers in a format that decoders don’t like.
                                                 # h264parse fixes this so the decoder can process it properly.
            "decoder": self.create_element("nvv4l2decoder", "nvv4l2-decoder", { # nvv4l2decoder: Decodes the H.264 video using NVIDIA GPU acceleration.
                                           "disable-dpb": True # Disables internal frame buffering (helps with performance)
                                           }),
            "capsfilter": self.create_element("capsfilter", "capsfilter", { # capsfilter: Ensures correct video format (NVIDIA Memory Management)
                                              "caps": Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12") # YUV color format that GPUs process efficiently.
                                              }),
            "mux_caps": self.create_element("capsfilter", "mux-caps", {
                                            "caps": Gst.Caps.from_string("video/x-h264,stream-format=byte-stream")
                                            }),
            "nvvidconv1": self.create_element("nvvidconv", "nvvidconv1"), # nvvidconv1: Converts the decoded video to the resolution expected by nvstreammux (640x384)
            "streammux": self.create_element("nvstreammux", "stream-muxer", { # nvstreammux: Batches frames for processing
                                             "width": self.config["streammux"]["width"],
                                             "height": self.config["streammux"]["height"],
                                             "batch-size": self.config["streammux"]["batch_size"],
                                             "batched-push-timeout": self.config["streammux"]["timeout"]
                                             }),
            "pgie": self.create_element("nvinfer", "primary-inference", { # nvinfer: Runs YOLOv5 inference
                                        "config-file-path": self.config["config_infer"]
                                        }),
            "tracker": self.create_element("nvtracker", "tracker", { # nvtracker: Tracks objects across frames
                                           "ll-config-file": self.config["tracker_config"],
                                           "ll-lib-file": "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
                                           }),
            "nvvidconv2": self.create_element("nvvidconv", "nvvidconv2"), # nvvidconv2: After tracking, frames may need reformatting before OSD (on-screen display).
            "osd": self.create_element("nvdsosd", "onscreendisplay"), # nvdsosd: Draws bounding boxes, labels, and other overlays on the video.
            "encoder": self.create_element("nvv4l2h264enc", "encoder", { # nvv4l2h264enc: Encodes the video back to H.264 for storage.
                                           "bitrate": self.config["encoder"]["bitrate"], # Mbps
                                           "preset-level": 1, # Speed/quality tradeoff
                                           "insert-sps-pps": 1, # Adds headers for playback compatibility
                                           "bufapi-version": 1,
                                           "iframeinterval": 30, # Keyframe interval
                                           "insert-vui": 1
                                           }),
            "muxer": self.create_element("qtmux", "muxer"), # qtmux: Muxes video into MKV
            "sink": self.create_element("filesink", "filesink", { # filesink: Saves the final video to disk.
                                        "location": self.config["output_file"],
                                        "sync": 0, # Disables sync (for maximum speed)
                                        "async": 0 # Disables async (for stability)
                                         })
        }

        # Link elements
        self.elements["source"].link(self.elements["demux"]) # Source → Demuxer
        self.elements["demux"].connect("pad-added", self.on_pad_added) # Demuxer → Parser (dynamic pad handling)
        self.elements["parser"].link(self.elements["decoder"]) # Parser → Decoder → Capfilter → NVVIDCONV1
        self.elements["decoder"].link(self.elements["capsfilter"])
        self.elements["capsfilter"].link(self.elements["nvvidconv1"])

        # Request pad for streammux (NVVIDCONV1 → StreamMux (request pad))
        sink_pad = self.elements["streammux"].get_request_pad("sink_0")
        self.requested_pads.append(sink_pad)  # Keep track for cleanup
        src_pad = self.elements["nvvidconv1"].get_static_pad("src")
        if src_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Failed to link nvvidconv1 to streammux")

        # Link the rest of the pipeline
        # StreamMux → Inference → Tracker → NVVIDCONV2 → OSD → Encoder → Muxer → Sink
        self.elements["streammux"].link(self.elements["pgie"])
        self.elements["pgie"].link(self.elements["tracker"])
        self.elements["tracker"].link(self.elements["nvvidconv2"])
        self.elements["nvvidconv2"].link(self.elements["osd"])
        self.elements["osd"].link(self.elements["encoder"])
        self.elements["encoder"].link(self.elements["mux_caps"])
        self.elements["mux_caps"].link(self.elements["muxer"])
        self.elements["muxer"].link(self.elements["sink"])

        # Add probe to OSD sink pad
        osd_sink_pad = self.elements["osd"].get_static_pad("sink")
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe)

        # Set up bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call)

    def run(self):
        """Run the application"""
        try:
            self.validate_config()
            self.build_pipeline()
            
            # Start pipeline
            self.pipeline.set_state(Gst.State.PLAYING)
            print("Pipeline running...")
            
            try:
                self.loop.run()
            except KeyboardInterrupt:
                print("\nInterrupt received, stopping...")
            
        except Exception as e:
            print(f"Error: {e}")
        finally:
            # Cleanup
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            for pad in self.requested_pads:
                self.elements["streammux"].release_request_pad(pad)

if __name__ == "__main__":
    try:
        import pyds  # DeepStream Python bindings
        #print(f"DeepStream Python bindings version: {pyds.__version__}")
        app = DeepStreamApp()
        app.run()
    except ImportError:
        print("Error: pyds module not found. Please ensure DeepStream is properly installed.")
        sys.exit(1)
