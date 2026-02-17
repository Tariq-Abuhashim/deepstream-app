#pragma once

#include <opencv2/opencv.hpp>
#include <gst/gst.h>
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include <cstring>
#include <iostream>

/* Fixed helper to make a CPU NvBufSurface and copy with NvBufSurfTransform
	Handle color format (33 = NV12 in NVMM)
	NV12 is a YUV color format that saves space by storing color information at lower 
	resolution than brightness:
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
			
	Human eyes are more sensitive to brightness than color, 
	so this saves ~50% memory without visible quality loss
*/


// Copy NVIDIA buffer to CPU and extract grayscale (Y plane only from NV12)
inline bool copyNvToCpuAndMakeGRAY(NvBufSurface *src_surf, cv::Mat &gray) {
    if (!src_surf || src_surf->numFilled < 1) {
        return false;
    }

    NvBufSurfaceParams *params = &src_surf->surfaceList[0];
    int width = params->width;
    int height = params->height;
    
#ifdef __aarch64__
    // Jetson path - use SURFACE_ARRAY
    if (src_surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
        NvBufSurface *cpu_surf = nullptr;
        NvBufSurfaceCreateParams create_params;
        memset(&create_params, 0, sizeof(create_params));
        
        create_params.gpuId = src_surf->gpuId;
        create_params.width = width;
        create_params.height = height;
        create_params.colorFormat = params->colorFormat;
        create_params.layout = NVBUF_LAYOUT_PITCH;
        create_params.memType = NVBUF_MEM_SURFACE_ARRAY;
        
        if (NvBufSurfaceCreate(&cpu_surf, 1, &create_params) != 0) {
            return false;
        }
        
        // Try copy first, fallback to transform
        int copy_result = NvBufSurfaceCopy(src_surf, cpu_surf);
        if (copy_result != 0) {
            NvBufSurfTransformParams xform_params;
            memset(&xform_params, 0, sizeof(xform_params));
            xform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
            xform_params.transform_filter = NvBufSurfTransformInter_Nearest;
            
            if (NvBufSurfTransform(src_surf, cpu_surf, &xform_params) != 0) {
                NvBufSurfaceDestroy(cpu_surf);
                return false;
            }
        }
        
        if (NvBufSurfaceMap(cpu_surf, 0, -1, NVBUF_MAP_READ) != 0) {
            NvBufSurfaceDestroy(cpu_surf);
            return false;
        }
        
        NvBufSurfaceSyncForCpu(cpu_surf, 0, -1);
        
        NvBufSurfaceParams *cpu_params = &cpu_surf->surfaceList[0];
        uint8_t *y_src = (uint8_t *)(cpu_params->mappedAddr.addr[0] ? 
                                      cpu_params->mappedAddr.addr[0] : 
                                      cpu_params->dataPtr);
        
        if (!y_src) {
            NvBufSurfaceUnMap(cpu_surf, 0, -1);
            NvBufSurfaceDestroy(cpu_surf);
            return false;
        }
        
        // Copy Y plane only (grayscale)
        gray.create(height, width, CV_8UC1);
        for (int i = 0; i < height; i++) {
            memcpy(gray.ptr(i), y_src + i * cpu_params->pitch, width);
        }

        NvBufSurfaceUnMap(cpu_surf, 0, -1);
        NvBufSurfaceDestroy(cpu_surf);
        return !gray.empty();
    }
#endif

    // Fallback for other memory types (x86, CUDA memory, etc.)
    NvBufSurface *dst_surf = nullptr;
    NvBufSurfaceCreateParams create_params;
    memset(&create_params, 0, sizeof(create_params));
    
    create_params.gpuId = src_surf->gpuId;
    create_params.width = width;
    create_params.height = height;
    create_params.colorFormat = params->colorFormat;
    create_params.layout = NVBUF_LAYOUT_PITCH;
    create_params.isContiguous = 1;
    
#ifdef __aarch64__
    create_params.memType = NVBUF_MEM_CUDA_UNIFIED;
#else
    create_params.memType = NVBUF_MEM_CUDA_PINNED;
#endif

    if (NvBufSurfaceCreate(&dst_surf, 1, &create_params) != 0) {
        return false;
    }

    NvBufSurfTransformParams xform_params;
    memset(&xform_params, 0, sizeof(xform_params));
    xform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
    xform_params.transform_filter = NvBufSurfTransformInter_Nearest;

    if (NvBufSurfTransform(src_surf, dst_surf, &xform_params) != 0) {
        NvBufSurfaceDestroy(dst_surf);
        return false;
    }
    
    if (NvBufSurfaceMap(dst_surf, 0, 0, NVBUF_MAP_READ) != 0) {
        NvBufSurfaceDestroy(dst_surf);
        return false;
    }

    NvBufSurfaceSyncForCpu(dst_surf, 0, 0);
    
    NvBufSurfaceParams *dst_params = &dst_surf->surfaceList[0];
    uint8_t *y_src = (uint8_t *)(dst_params->mappedAddr.addr[0] ?
                                  dst_params->mappedAddr.addr[0] :
                                  dst_params->dataPtr);
    
    // Copy Y plane only
    gray.create(height, width, CV_8UC1);
    for (int i = 0; i < height; i++) {
        memcpy(gray.ptr(i), y_src + i * dst_params->pitch, width);
    }
    
    NvBufSurfaceUnMap(dst_surf, 0, 0);
    NvBufSurfaceDestroy(dst_surf);

    return !gray.empty();
}

// Copy NVIDIA buffer to CPU and convert to BGR
inline bool copyNvToCpuAndMakeBGR(NvBufSurface *src_surf, cv::Mat &bgr) {
    if (!src_surf || src_surf->numFilled < 1) {
        return false;
    }

    NvBufSurfaceParams *params = &src_surf->surfaceList[0];
    int width = params->width;
    int height = params->height;
    
#ifdef __aarch64__
    if (src_surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
        NvBufSurface *cpu_surf = nullptr;
        NvBufSurfaceCreateParams create_params;
        memset(&create_params, 0, sizeof(create_params));
        
        create_params.gpuId = src_surf->gpuId;
        create_params.width = width;
        create_params.height = height;
        create_params.colorFormat = params->colorFormat;
        create_params.layout = NVBUF_LAYOUT_PITCH;
        create_params.memType = NVBUF_MEM_SURFACE_ARRAY;
        
        if (NvBufSurfaceCreate(&cpu_surf, 1, &create_params) != 0) {
            return false;
        }
        
        int copy_result = NvBufSurfaceCopy(src_surf, cpu_surf);
        if (copy_result != 0) {
            NvBufSurfTransformParams xform_params;
            memset(&xform_params, 0, sizeof(xform_params));
            xform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
            xform_params.transform_filter = NvBufSurfTransformInter_Nearest;
            
            if (NvBufSurfTransform(src_surf, cpu_surf, &xform_params) != 0) {
                NvBufSurfaceDestroy(cpu_surf);
                return false;
            }
        }
        
        if (NvBufSurfaceMap(cpu_surf, 0, -1, NVBUF_MAP_READ) != 0) {
            NvBufSurfaceDestroy(cpu_surf);
            return false;
        }
        
        NvBufSurfaceSyncForCpu(cpu_surf, 0, -1);
        
        NvBufSurfaceParams *cpu_params = &cpu_surf->surfaceList[0];
        uint8_t *y_data = (uint8_t *)cpu_params->mappedAddr.addr[0];
        if (!y_data) y_data = (uint8_t *)cpu_params->dataPtr;
        
        uint8_t *uv_data = nullptr;
        if (cpu_params->planeParams.num_planes > 1) {
            uv_data = (uint8_t *)cpu_params->mappedAddr.addr[1];
            if (!uv_data) {
                uv_data = y_data + (cpu_params->planeParams.pitch[0] * cpu_params->planeParams.height[0]);
            }
        } else {
            uv_data = y_data + (cpu_params->pitch * cpu_params->height);
        }
        
        if (!y_data) {
            NvBufSurfaceUnMap(cpu_surf, 0, -1);
            NvBufSurfaceDestroy(cpu_surf);
            return false;
        }
        
        // Create NV12 Mat and copy both planes
        cv::Mat nv12(height * 3 / 2, width, CV_8UC1);
        
        for (int i = 0; i < height; i++) {
            memcpy(nv12.data + i * width, y_data + i * cpu_params->pitch, width);
        }
        
        for (int i = 0; i < height / 2; i++) {
            memcpy(nv12.data + height * width + i * width, 
                   uv_data + i * cpu_params->pitch, width);
        }
        
        cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
        
        NvBufSurfaceUnMap(cpu_surf, 0, -1);
        NvBufSurfaceDestroy(cpu_surf);
        return !bgr.empty();
    }
#endif

    // Fallback path
    NvBufSurface *dst_surf = nullptr;
    NvBufSurfaceCreateParams create_params;
    memset(&create_params, 0, sizeof(create_params));
    
    create_params.gpuId = src_surf->gpuId;
    create_params.width = width;
    create_params.height = height;
    create_params.colorFormat = params->colorFormat;
    create_params.layout = NVBUF_LAYOUT_PITCH;
    create_params.isContiguous = 1;
    
#ifdef __aarch64__
    create_params.memType = NVBUF_MEM_CUDA_UNIFIED;
#else
    create_params.memType = NVBUF_MEM_CUDA_PINNED;
#endif

    if (NvBufSurfaceCreate(&dst_surf, 1, &create_params) != 0) {
        return false;
    }

    NvBufSurfTransformParams xform_params;
    memset(&xform_params, 0, sizeof(xform_params));
    xform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
    xform_params.transform_filter = NvBufSurfTransformInter_Algo2;

    if (NvBufSurfTransform(src_surf, dst_surf, &xform_params) != 0) {
        NvBufSurfaceDestroy(dst_surf);
        return false;
    }

    if (NvBufSurfaceMap(dst_surf, 0, 0, NVBUF_MAP_READ) != 0) {
        NvBufSurfaceDestroy(dst_surf);
        return false;
    }
    
    NvBufSurfaceSyncForCpu(dst_surf, 0, 0);
    
    NvBufSurfaceParams *dst_params = &dst_surf->surfaceList[0];
    uint8_t *y_data = (uint8_t *)(dst_params->mappedAddr.addr[0] ? 
                                   dst_params->mappedAddr.addr[0] : dst_params->dataPtr);
    
    if (y_data) {
        uint8_t *uv_data = y_data + (dst_params->pitch * dst_params->height);
        
        cv::Mat nv12(height * 3 / 2, width, CV_8UC1);
        
        for (int i = 0; i < height; i++) {
            memcpy(nv12.data + i * width, y_data + i * dst_params->pitch, width);
        }
        
        for (int i = 0; i < height / 2; i++) {
            memcpy(nv12.data + height * width + i * width, 
                   uv_data + i * dst_params->pitch, width);
        }
        
        cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
    }
    
    NvBufSurfaceUnMap(dst_surf, 0, 0);
    NvBufSurfaceDestroy(dst_surf);
    
    return !bgr.empty();
}
