cmake_minimum_required(VERSION 3.10)
project(deepstream_detr_app)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")

# Verify this path matches your DeepStream installation
set(DS_PATH "/opt/nvidia/deepstream/deepstream-6.0")

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0>=1.14)
pkg_check_modules(GST_VIDEO REQUIRED gstreamer-video-1.0)
pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(GOBJECT REQUIRED gobject-2.0)

# Include directories
include_directories(
    ${DS_PATH}/sources/includes
    ${DS_PATH}/sources/apps/apps-common/includes
    ${DS_PATH}/sources/gst-plugins/gst-nvdsosd
    ${DS_PATH}/sources/gst-plugins/gst-nvinfer
    ${GST_INCLUDE_DIRS}
    ${GST_VIDEO_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR}/apps-common
)

# Library directories
link_directories(
    ${DS_PATH}/lib
    ${GST_LIBRARY_DIRS}
    ${GST_VIDEO_LIBRARY_DIRS}
    /usr/local/cuda-11.4/lib64  # Adjust CUDA version if needed
)

# Custom parser library
add_library(nvds_infercustomparser_detr SHARED nvdsinfer_customparser_detr.cpp)
target_include_directories(nvds_infercustomparser_detr PRIVATE 
    ${DS_PATH}/sources/includes 
    /usr/local/cuda-11.4/include
)
target_link_libraries(nvds_infercustomparser_detr nvds_infer)

# Main application
add_executable(deepstream_detr_app main.cpp)
target_link_libraries(deepstream_detr_app
    nvds_infercustomparser_detr
    nvdsgst_meta
    nvds_meta
    nvds_infer
    nvds_utils
    nvbufsurface
    nvbufsurftransform
    gstreamer-1.0
    gstvideo-1.0
    glib-2.0
    gobject-2.0
    cudart
)

# Install rules (optional)
install(TARGETS deepstream_detr_app DESTINATION bin)

#sudo apt-get install libgstrtspserver-1.0-dev


