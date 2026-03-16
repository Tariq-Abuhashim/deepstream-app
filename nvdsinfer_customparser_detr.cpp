#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "nvdsinfer_custom_impl.h"

// Softmax function for converting logits to probabilities
void softmax(const float* input, float* output, int length) {
    float max_val = *std::max_element(input, input + length);
    float sum = 0.0f;
    
    for (int i = 0; i < length; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    for (int i = 0; i < length; ++i) {
        output[i] /= sum;
    }
}

extern "C" bool NvDsInferParseCustomDETR(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    static int call_count = 0;
    call_count++;
    
    if (outputLayersInfo.size() < 2) {
        std::cerr << "[PARSER ERROR] Expected at least 2 output layers, got " 
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    const NvDsInferLayerInfo* logitsLayer = &outputLayersInfo[0];
    const NvDsInferLayerInfo* boxesLayer = &outputLayersInfo[1];

    const float *logits = static_cast<const float *>(logitsLayer->buffer);
    const float *boxes = static_cast<const float *>(boxesLayer->buffer);

    int num_queries = logitsLayer->inferDims.d[0];
    int num_classes = logitsLayer->inferDims.d[1];

    /*if (call_count <= 3) {
        std::cerr << "[PARSER] num_queries=" << num_queries 
                  << " num_classes=" << num_classes << std::endl;
    }*/

    // Sanity check
    if (boxesLayer->inferDims.d[0] != num_queries ||
        boxesLayer->inferDims.d[1] != 4) {
        std::cerr << "[PARSER ERROR] Unexpected DETR output dimensions." << std::endl;
        return false;
    }

    int detections_found = 0;
    float max_confidence_seen = -1000.0f;
    float max_raw_logit_seen = -1000.0f;
    int best_class_seen = -1;
    
    std::vector<float> probs(num_classes);
    
    for (int i = 0; i < num_queries; ++i) {
        // Apply softmax to convert logits to probabilities
        softmax(&logits[i * num_classes], probs.data(), num_classes);
        
        // Find class with highest probability
        int maxClass = -1;
        float maxProb = -1.0f;
        float maxRawLogit = -1000.0f;
        
        for (int c = 0; c < num_classes; ++c) {
            if (probs[c] > maxProb) {
                maxProb = probs[c];
                maxClass = c;
                maxRawLogit = logits[i * num_classes + c];
            }
        }

        // Track highest confidence across all queries for debugging
        if (maxProb > max_confidence_seen) {
            max_confidence_seen = maxProb;
            max_raw_logit_seen = maxRawLogit;
            best_class_seen = maxClass;
        }

        // Print first few detections for debugging
        /*
        if (call_count <= 3 && i < 5) {
            std::cerr << "[PARSER] Query " << i 
                      << " class=" << maxClass 
                      << " raw_logit=" << maxRawLogit
                      << " prob=" << maxProb << std::endl;
        }
        */

        // Get threshold for this class
        float threshold = (maxClass < detectionParams.numClassesConfigured) 
                         ? detectionParams.perClassPreclusterThreshold[maxClass]
                         : detectionParams.perClassPreclusterThreshold[0];

        // Only add detection if above threshold and not background class
        if (maxProb >= threshold && maxClass >= 0 && maxClass < (num_classes - 1)) {
            
            NvDsInferObjectDetectionInfo obj;
            obj.classId = maxClass;
            obj.detectionConfidence = maxProb;

            const float cx = boxes[i * 4 + 0] * networkInfo.width;
            const float cy = boxes[i * 4 + 1] * networkInfo.height;
            const float w  = boxes[i * 4 + 2] * networkInfo.width;
            const float h  = boxes[i * 4 + 3] * networkInfo.height;

            obj.left = cx - w / 2.0f;
            obj.top = cy - h / 2.0f;
            obj.width = w;
            obj.height = h;

            // Clamp to valid bounds
            if (obj.left < 0) obj.left = 0;
            if (obj.top < 0) obj.top = 0;
            if (obj.left + obj.width > networkInfo.width) 
                obj.width = networkInfo.width - obj.left;
            if (obj.top + obj.height > networkInfo.height) 
                obj.height = networkInfo.height - obj.top;

            if (obj.width > 0 && obj.height > 0) {
                objectList.push_back(obj);
                detections_found++;
            }
        }
    }
/*
    if (call_count <= 5) {
        std::cerr << "[PARSER] Frame " << call_count 
                  << " - Found " << detections_found << " detections above threshold " 
                  << detectionParams.perClassPreclusterThreshold[0] << std::endl;
        std::cerr << "[PARSER] Frame " << call_count 
                  << " - Max raw logit: " << max_raw_logit_seen << std::endl;
        std::cerr << "[PARSER] Frame " << call_count 
                  << " - Max probability (after softmax): " << max_confidence_seen 
                  << " (class " << best_class_seen << ")" << std::endl;
    }
*/

    return true;
}

/* Called by nvinfer when custom parser .so is loaded.
*/
extern "C" bool NvDsInferInitialize(void *handle) {
    return true;
}

/* Called when DeepStream unloads .so or shuts down the pipeline.
*/
extern "C" void NvDsInferDeInitialize(void *handle) {
    // No-op
}

/* Called by nvinfer only if the user did NOT override thresholds in the config file.
 	num-detected-classes=3
 	pre-cluster-threshold=0.3;0.25;0.4
*/
extern "C" NvDsInferParseDetectionParams NvDsInferGetDefaultParseDetectionParams() {
    int num_classes = 3;  // Adjust to your actual number of classes
    NvDsInferParseDetectionParams params;
    params.numClassesConfigured = num_classes;
    for (int i = 0; i < num_classes; ++i)
        params.perClassPreclusterThreshold[i] = 0.2f;
    return params;
}
