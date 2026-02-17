#pragma once

#include "types.h"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <utility>
#include <opencv2/opencv.hpp>

// Helper: Get value after '=' in argument
inline std::string get_value(const std::string& arg) {
    auto pos = arg.find('=');
    if (pos == std::string::npos) return "";
    return arg.substr(pos + 1);
}
// Helper: Check if file exists
inline bool file_exists(const std::string& path) {
    return std::ifstream(path).good();
}

// Helper: Parse key=value line
inline std::pair<std::string, std::string> parse_key_value(const std::string& line) {
    size_t delim_pos = line.find('=');
    if (delim_pos == std::string::npos) {
        delim_pos = line.find(':');
    }
    if (delim_pos == std::string::npos) {
        return {"", ""};
    }
    
    std::string key = line.substr(0, delim_pos);
    std::string value = line.substr(delim_pos + 1);
    
    // Trim whitespace
    auto trim = [](std::string& s) {
        s.erase(0, s.find_first_not_of(" \t\r\n"));
        s.erase(s.find_last_not_of(" \t\r\n") + 1);
    };
    
    trim(key);
    trim(value);
    
    return {key, value};
}

// Helper: Parse "3;512;1384" format
inline bool parse_infer_dims(const std::string& value, int& height, int& width) {
    std::string dims = value;
    for (char& c : dims) {
        if (c == ';') c = ' ';
    }
    
    std::istringstream iss(dims);
    int channels, h, w;
    
    if (iss >> channels >> h >> w) {
        height = h;
        width = w;
        return true;
    }
    
    return false;
}

// Helper: Load model configuration from file
inline bool load_model_config(const std::string& config_path, ModelConfig& config) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open config file: " << config_path << "\n";
        return false;
    }
    
    config.config_file_path = config_path;
    config.batch_size = 1;
    config.confidence_threshold = 0.5f;
    config.maintain_aspect_ratio = true;
    config.model_name = "Unknown Model";
    
    std::string line;
    bool found_width = false;
    bool found_height = false;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;
        }
        
        auto [key, value] = parse_key_value(line);
        if (key.empty()) continue;
        
        if (key == "infer-dims" || key == "network-input-shape") {
            if (parse_infer_dims(value, config.infer_height, config.infer_width)) {
                found_width = true;
                found_height = true;
            }
        }
        else if (key == "model-engine-file" || key == "model-file") {
            config.model_engine = value;
        }
        else if (key == "labelfile-path") {
            config.label_file_path = value;
        }
        else if (key == "batch-size") {
            try { config.batch_size = std::stoi(value); } 
            catch (...) { std::cerr << "[WARN] Invalid batch-size: " << value << "\n"; }
        }
        else if (key == "model-name" || key == "model_name") {
            config.model_name = value;
        }
        else if (key == "input-width" || key == "infer_width") {
            try { 
                config.infer_width = std::stoi(value);
                found_width = true;
            } catch (...) { std::cerr << "[WARN] Invalid input-width: " << value << "\n"; }
        }
        else if (key == "input-height" || key == "infer_height") {
            try { 
                config.infer_height = std::stoi(value);
                found_height = true;
            } catch (...) { std::cerr << "[WARN] Invalid input-height: " << value << "\n"; }
        }
        else if (key == "confidence-threshold" || key == "confidence_threshold") {
            try { config.confidence_threshold = std::stof(value); } 
            catch (...) { std::cerr << "[WARN] Invalid confidence-threshold: " << value << "\n"; }
        }
        else if (key == "maintain-aspect-ratio" || key == "maintain_aspect_ratio") {
            std::string v = value;
            std::transform(v.begin(), v.end(), v.begin(), ::tolower);
            config.maintain_aspect_ratio = (v == "1" || v == "true" || v == "yes");
        }
    }
    
    file.close();
    
    // Validation
    if (!found_width || !found_height || config.infer_width <= 0 || config.infer_height <= 0) {
        std::cerr << "[ERROR] Config validation failed:\n";
        std::cerr << "  Width found: " << found_width << " (value: " << config.infer_width << ")\n";
        std::cerr << "  Height found: " << found_height << " (value: " << config.infer_height << ")\n";
        std::cerr << "  Please add either:\n";
        std::cerr << "    1. infer-dims=3;HEIGHT;WIDTH\n";
        std::cerr << "    OR\n";
        std::cerr << "    2. input-width=WIDTH and input-height=HEIGHT\n";
        return false;
    }
    
    return true;
}

// Helper: Detect video dimensions from URI
inline bool get_video_dimensions(const std::string& uri, int& width, int& height) {
    if (uri.find("file://") == 0) {
        std::string path = uri.substr(7);
        cv::VideoCapture cap(path);
        if (cap.isOpened()) {
            width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
            height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            cap.release();
            return (width > 0 && height > 0);
        }
    }
    return false;
}

