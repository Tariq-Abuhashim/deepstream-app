#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>
#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseCustomDETR(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    //const float kBboxNorm = 100.0f;
    const NvDsInferLayerInfo *bboxLayer = nullptr;
    const NvDsInferLayerInfo *logitsLayer = nullptr;

    for (const auto &layer : outputLayersInfo) {
        if (strcmp(layer.layerName, "boxes") == 0) {
            bboxLayer = &layer;
        } else if (strcmp(layer.layerName, "logits") == 0) {
            logitsLayer = &layer;
        }
    }

    if (!bboxLayer || !logitsLayer) {
        std::cerr << "Missing expected output layers (boxes, logits)" << std::endl;
        return false;
    }

    const float *boxes = static_cast<const float *>(bboxLayer->buffer);
    const float *logits = static_cast<const float *>(logitsLayer->buffer);

    int num_queries = logitsLayer->inferDims.d[0];
    int num_classes = logitsLayer->inferDims.d[1];
    //std::cout << "num_queries=" << num_queries << ", num_classes=" << num_classes << std::endl;

    for (int i = 0; i < num_queries; ++i) {
        int maxClass = -1;
        float maxProb = detectionParams.perClassPreclusterThreshold[0];

        for (int c = 0; c < num_classes; ++c) {
            float prob = logits[i * num_classes + c];
            if (prob > maxProb) {
                maxProb = prob;
                maxClass = c;
            }
        }
        
        //std::cout << "Query " << i << " maxClass=" << maxClass << " maxProb=" << maxProb << std::endl;

        if (maxClass == 1) { // Only person class
            NvDsInferObjectDetectionInfo obj;
            obj.classId = maxClass;
            obj.detectionConfidence = maxProb;

            const float cx = boxes[i * 4 + 0] * networkInfo.width;
            const float cy = boxes[i * 4 + 1] * networkInfo.height;
            const float w  = boxes[i * 4 + 2] * networkInfo.width;
            const float h  = boxes[i * 4 + 3] * networkInfo.height;

            obj.left = cx - w / 2;
            obj.top = cy - h / 2;
            obj.width = w;
            obj.height = h;

            objectList.push_back(obj);
        }
    }

    return true;
}

extern "C" bool NvDsInferInitialize(void *handle) {
    return true;
}

extern "C" void NvDsInferDeInitialize(void *handle) {
    // No-op
}

extern "C" NvDsInferParseDetectionParams NvDsInferGetDefaultParseDetectionParams() {
    NvDsInferParseDetectionParams params;
    params.numClassesConfigured = 91;
    for (int i = 0; i < 91; ++i)
        params.perClassPreclusterThreshold[i] = 0.2f;
    return params;
}

