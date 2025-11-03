cmake_minimum_required(VERSION 3.10)
project(deepstream-app)

set(CMAKE_CXX_STANDARD 17)

# Use With Caution
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
    -Wno-deprecated-declarations \
    -Wno-attributes \
    -Wno-cpp \
    -O3")

# Python3
find_package(Python3 3.8 EXACT REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3_EXECUTABLE = ${Python3_EXECUTABLE}")
message(STATUS "Python3_INCLUDE_DIRS = ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES = ${Python3_LIBRARIES}")

# Pangolin
find_package(Pangolin REQUIRED)
message(STATUS "Pangolin_INCLUDE_DIRS = ${Pangolin_INCLUDE_DIRS}")
message(STATUS "Pangolin_LIBRARIES = ${Pangolin_LIBRARIES}")

# Eigen3
find_package(Eigen3 3.1.0 REQUIRED)
message(STATUS "Eigen3 found: ${Eigen3_FOUND}")
message(STATUS "Eigen3 include dirs: ${EIGEN3_INCLUDE_DIRS}")

# DeepStream + GStreamer
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")
find_package(DeepStream REQUIRED)

# PkgConfig
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0>=1.14)
pkg_check_modules(GST_VIDEO REQUIRED gstreamer-video-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)
pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(GOBJECT REQUIRED gobject-2.0)

# OpenCV
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)

# ORB-SLAM3 paths - VERIFY THESE
set(ORB_SLAM3_DIR "/home/mrt/dev/window-tracker/ORB_SLAM3")
set(ORB_SLAM3_INCLUDE_DIRS
    ${ORB_SLAM3_DIR}
    ${ORB_SLAM3_DIR}/include
    ${ORB_SLAM3_DIR}/include/CameraModels
    ${ORB_SLAM3_DIR}/Thirdparty/Sophus
    ${ORB_SLAM3_DIR}/Thirdparty/DBoW2
    ${ORB_SLAM3_DIR}/Thirdparty/g2o
)

# Check if ORB-SLAM3 library exists
if(NOT EXISTS "${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so")
    message(WARNING "ORB-SLAM3 library not found at: ${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so")
endif()

# Include directories
include_directories(
    ${DeepStream_INCLUDE_DIRS}
    ${GST_INCLUDE_DIRS}
    ${GST_VIDEO_INCLUDE_DIRS}
    ${GST_APP_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${GOBJECT_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR}/apps-common
    ${Python3_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${ORB_SLAM3_INCLUDE_DIRS}
)

# Library search paths
link_directories(
    ${GST_LIBRARY_DIRS}
    ${GST_VIDEO_LIBRARY_DIRS}
    ${GST_APP_LIBRARY_DIRS}
    /usr/local/cuda-11.4/lib64
    ${ORB_SLAM3_DIR}/lib
)

# Custom parser
add_library(nvds_infercustomparser_detr SHARED nvdsinfer_customparser_detr.cpp)
target_include_directories(nvds_infercustomparser_detr PRIVATE 
    ${DeepStream_INCLUDE_DIRS}
    /usr/local/cuda-11.4/include
)
target_link_libraries(nvds_infercustomparser_detr DeepStream::nvds_infer)

# Find ORB-SLAM3 dependencies
find_library(DBOW2_LIB NAMES DBoW2 PATHS ${ORB_SLAM3_DIR}/Thirdparty/DBoW2/lib)
find_library(G2O_LIB NAMES g2o PATHS ${ORB_SLAM3_DIR}/Thirdparty/g2o/lib)

# ORB-SLAM3 dependencies (typical - adjust based on your build)
set(ORB_SLAM3_DEPENDENCIES
    ${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so
    ${DBOW2_LIB}
    ${G2O_LIB}
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    ${Python3_LIBRARIES}
)

# DeepStream + ORB-SLAM3 app
add_executable(deepstream-orbslam orbslam.cpp)
target_link_libraries(deepstream-orbslam
    # DeepStream libraries first
    nvds_infercustomparser_detr
    DeepStream::nvdsgst_meta
    DeepStream::nvds_meta
    DeepStream::nvds_infer
    DeepStream::nvbufsurface
    DeepStream::nvbufsurftransform
    
    # GStreamer libraries
    ${GST_LIBRARIES}
    ${GST_VIDEO_LIBRARIES}
    ${GST_APP_LIBRARIES}
    ${GLIB_LIBRARIES}
    ${GOBJECT_LIBRARIES}
    
    # CUDA
    cudart
    
    # ORB-SLAM3 and dependencies - IN CORRECT ORDER
    ${ORB_SLAM3_DEPENDENCIES}
    Eigen3::Eigen
)

if(UNIX)
    target_link_options(deepstream-orbslam PRIVATE
        "-Wl,--exclude-libs,ALL"  # Hide all symbols from static libraries
    )
endif()

# Add compile definitions if needed
target_compile_definitions(deepstream-orbslam PRIVATE 
    HAVE_CXX11=1
    COMPILEDWITHC11=1
)

# DeepStream-only app
add_executable(deepstream-track track.cpp)
target_link_libraries(deepstream-track
    nvds_infercustomparser_detr
    DeepStream::nvdsgst_meta
    DeepStream::nvds_meta
    DeepStream::nvds_infer
    DeepStream::nvbufsurface
    DeepStream::nvbufsurftransform
    ${GST_LIBRARIES}
    ${GST_VIDEO_LIBRARIES}
    ${GST_APP_LIBRARIES}
    ${GLIB_LIBRARIES}
    ${GOBJECT_LIBRARIES}
    cudart
)

# Install rule (optional)
install(TARGETS deepstream-track deepstream-orbslam DESTINATION bin)

