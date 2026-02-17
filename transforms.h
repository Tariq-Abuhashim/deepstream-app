#pragma once

#include "types.h"
#include <iostream>
#include <algorithm>

// Calculate letterbox/pillarbox parameters
inline void calculate_letterbox_params(int src_w, int src_h, 
                                int dst_w, int dst_h,
                                TransformParams &params) {
    float src_aspect = (float)src_w / src_h;
    float dst_aspect = (float)dst_w / dst_h;
    
    params.original_w = src_w;
    params.original_h = src_h;
    params.model_w = dst_w;
    params.model_h = dst_h;
    
    if (src_aspect > dst_aspect) {
        int scaled_h = (int)(dst_w / src_aspect);
        params.scale_x = (float)src_w / dst_w;
        params.scale_y = (float)src_h / scaled_h;
        params.offset_x = 0;
        params.offset_y = (dst_h - scaled_h) / 2;
        std::cerr << "[TRANSFORM] Letterbox mode (bars top/bottom)\n";
    } else {
        int scaled_w = (int)(dst_h * src_aspect);
        params.scale_x = (float)src_w / scaled_w;
        params.scale_y = (float)src_h / dst_h;
        params.offset_x = (dst_w - scaled_w) / 2;
        params.offset_y = 0;
        std::cerr << "[TRANSFORM] Pillarbox mode (bars left/right)\n";
    }
    
    std::cerr << "[TRANSFORM] Source: " << src_w << "x" << src_h 
              << " -> Model: " << dst_w << "x" << dst_h << "\n";
    std::cerr << "[TRANSFORM] Scale: " << params.scale_x << "x" << params.scale_y 
              << " Offset: (" << params.offset_x << "," << params.offset_y << ")\n";
}

// Transform detection coordinates from model space to original space
inline void transform_detection(DetectedObject &obj, const TransformParams &params) {
    float adj_left = obj.left - params.offset_x;
    float adj_top = obj.top - params.offset_y;
    
    obj.left = adj_left * params.scale_x;
    obj.top = adj_top * params.scale_y;
    obj.width = obj.width * params.scale_x;
    obj.height = obj.height * params.scale_y;
    
    // Clamp to image boundaries
    obj.left = std::max(0.0f, std::min(obj.left, (float)params.original_w - 1));
    obj.top = std::max(0.0f, std::min(obj.top, (float)params.original_h - 1));
    obj.width = std::max(1.0f, std::min(obj.width, (float)params.original_w - obj.left));
    obj.height = std::max(1.0f, std::min(obj.height, (float)params.original_h - obj.top));
}

// Scale camera intrinsics
inline ScaledIntrinsics scale_intrinsics (float orig_fx, float orig_fy, float orig_cx, float orig_cy,
									int orig_width, int orig_height, int new_width, int new_height) {
	ScaledIntrinsics scaled;
	float scale_x = (float)new_width / orig_width;
	float scale_y = (float)new_height / orig_height;
	
	scaled.fx = scale_x * orig_fx;
	scaled.fy = scale_y * orig_fy;
	scaled.cx = scale_x * orig_cx;
	scaled.cy = scale_y * orig_cy;
	scaled.width = new_width;
	scaled.height = new_height;
	
	std::cerr << "[INTRINSICS] Scaling from " << orig_width << "x" << orig_height 
              << " to " << new_width << "x" << new_height << "\n";
    std::cerr << "  Original: fx=" << orig_fx << " fy=" << orig_fy 
              << " cx=" << orig_cx << " cy=" << orig_cy << "\n";
    std::cerr << "  Scaled:   fx=" << scaled.fx << " fy=" << scaled.fy 
              << " cx=" << scaled.cx << " cy=" << scaled.cy << "\n";
	
	return scaled;

}

// Round up to nearest multiple of 32
inline int round_to_multiple_of_32(int value) {
    return ((value + 31) / 32) * 32;
}
