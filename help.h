#pragma once

#include <iostream>
#include <fstream>

// Print detailed usage information
inline void print_usage() {
    std::cerr << R"(
usage
    deepstream-orbslam <options>
    deepstream-orbslam <uri> <settings> [options]

unnamed options
    uri                  URI of the camera (e.g., file:///path/to/video.mp4)
    settings             Settings file for the tracker (YAML format)

options
    -c,--config=<path>   Path to model config (default: config_infer_primary_detr.txt)
    -h,--help            Print help and exit
    --no-stdout          Do not output to stdout
    -v,--verbose         Verbose output
    --vocabulary=<path>  Path to ORB vocabulary (default: ORBvoc.txt)
    --headless           Run without visualization

description
    Pipeline for running camera, tracking objects and running ORB-SLAM3.

)" << std::endl;
}

// Print short help message
inline void print_help() {
    std::cerr <<
    "Usage: app <uri> <ORBvoc.txt> <settings.yaml> [options]\n"
    "\n"
    "Options:\n"
    "  --help            Show this help message\n"
    "  --headless        Run without visualization\n"
    "  --orbslam         Enable ORB-SLAM processing\n"
    "  --no-stdout       Supress stdout detections\n";
}

// Print README.md contents
inline void print_readme() {
    std::ifstream f("README.md");
    if (!f.is_open()) {
        std::cerr << "ERROR: README.md not found\n";
        return;
    }
    std::cerr << f.rdbuf();
}
