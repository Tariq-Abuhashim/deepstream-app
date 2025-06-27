cmake_minimum_required(VERSION 3.10)
project(deepstream_app)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")

# Find DeepStream and other required packages
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")
find_package(DeepStream REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0>=1.14)
pkg_check_modules(GST_VIDEO REQUIRED gstreamer-video-1.0)
pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(GOBJECT REQUIRED gobject-2.0)

# Include directories
include_directories(
    ${DeepStream_INCLUDE_DIRS}
    ${GST_INCLUDE_DIRS}
    ${GST_VIDEO_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR}/apps-common
)

# Library directories (DeepStream libs already found via DeepStream_LIBRARIES)
link_directories(
    ${GST_LIBRARY_DIRS}
    ${GST_VIDEO_LIBRARY_DIRS}
    /usr/local/cuda-11.4/lib64  # Adjust if needed
)

# Custom parser library
add_library(nvds_infercustomparser_detr SHARED nvdsinfer_customparser_detr.cpp)
target_include_directories(nvds_infercustomparser_detr PRIVATE 
    ${DeepStream_INCLUDE_DIRS}
    /usr/local/cuda-11.4/include
)
target_link_libraries(nvds_infercustomparser_detr DeepStream::nvds_infer)

# Main application
add_executable(deepstream_app main.cpp)
target_link_libraries(deepstream_app
    nvds_infercustomparser_detr
    DeepStream::nvdsgst_meta
    DeepStream::nvds_meta
    DeepStream::nvds_infer
    DeepStream::nvbufsurface
    DeepStream::nvbufsurftransform
    ${GST_LIBRARIES}
    ${GST_VIDEO_LIBRARIES}
    ${GLIB_LIBRARIES}
    ${GOBJECT_LIBRARIES}
    cudart
)

# Install rule (optional)
install(TARGETS deepstream_app DESTINATION bin)

