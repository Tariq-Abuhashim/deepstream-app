#pragma once

#include <map>
#include <deque>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <nvdsmeta.h>

// Define detected objects
struct DetectedObject {
    int id;           // DeepStream tracker ID
    int class_id;     // COCO class (e.g., 0 person, 2 car)
    float confidence;
    float left;       // bbox in pixels
    float top;
    float width;
    float height;
};

// Define frame packet
struct FramePacket {
    cv::Mat frame;                  // Image for SLAM
    double timestamp;                   // seconds
    std::vector<DetectedObject> objs;   // 2D detections for this frame

    ~FramePacket() {
		frame.release();
    }
};

// Coordinate transformation parameters
struct TransformParams {
    float scale_x, scale_y;
    int offset_x, offset_y;
    int original_w, original_h;
    int model_w, model_h;
};

// Model configuration
struct ModelConfig {
    std::string config_file_path;
    int infer_width;
    int infer_height;
    std::string model_name;
    std::string model_engine;
    std::string label_file_path;
    int batch_size;
    float confidence_threshold;
    bool maintain_aspect_ratio;
    
    inline bool is_valid() const {
        return infer_width > 0 && infer_height > 0;
    }
    
    inline void print() const {
        std::cerr << "[MODEL CONFIG]\n"
                  << "  Name: " << model_name << "\n"
                  << "  Config file: " << config_file_path << "\n"
                  << "  Engine file: " << model_engine << "\n"
                  << "  Label file: " << label_file_path << "\n"
                  << "  Infer dimensions: " << infer_width << "x" << infer_height << "\n"
                  << "  Batch size: " << batch_size << "\n"
                  << "  Confidence threshold: " << confidence_threshold << "\n"
                  << "  Maintain aspect ratio: " << (maintain_aspect_ratio ? "yes" : "no") << "\n";
    }
};

// Scale camera intrinsics
struct ScaledIntrinsics {
	float fx, fy, cx, cy;
	int width, height;
};


/**
 * Kalman Filter for 2D bounding box tracking
 * 
 * State vector: [cx, cy, w, h, vx, vy, vw, vh]
 *   cx, cy: center position
 *   w, h: width, height
 *   vx, vy: velocity of center
 *   vw, vh: rate of change of size
 */

class BBoxKalmanFilter {

public:

	BBoxKalmanFilter () {

		// Initialize state (position + velocity)
		// z = [cx, cy, w, h, vx, vy, vw, vh]
		x = Eigen::VectorXd::Zero(8);
		    
		// State transition matrix (constant velocity model)
		F = Eigen::MatrixXd::Identity(8, 8);
		F(0, 4) = 1.0; // cx += vx * dt (dt=1 frame)
		F(1, 5) = 1.0; // cy += vy * dt
		F(2, 6) = 1.0; // w += vx * dt
		F(3, 7) = 1.0; // h += vh * dt
		
		// Measurement model matrix
		// z = [cx, cy, w, h]
		H = Eigen::MatrixXd::Zero(4, 8); // FIXME Identity(4, 8) ?
		H(0, 0) = 1.0;
		H(1, 1) = 1.0;
		H(2, 2) = 1.0;
		H(3, 3) = 1.0;
		
		// Process noise covariance (how much we trust the motion model)
		Q = Eigen::MatrixXd::Identity(8, 8);
		Q.block<4,4>(0,0) *= 1.0;	// Low noise in position
		Q.block<4,4>(4,4) *= 10.0;	// Higher noise in velocity
		 
		// Measurement noise covariance (how much we trust detections)
		R = Eigen::MatrixXd::Identity(4, 4);
		R *= 10.0;
		
		// Error covariance matrix
		P = Eigen::MatrixXd::Identity(8, 8) * 100.0;

		initialised = false;
		age = 0;
	}

	// Initialize filter with first detection
	void init(const NvOSD_RectParams& bbox) {

		x	<<	bbox.left + bbox.width / 2.0f,
				bbox.top + bbox.height / 2.0f,
				bbox.width, bbox.height,
				0, 0, 0, 0;
				
		initialised = true;
		
		age = 1;
			
	}

	// Predict next state (called every frame)
	void predict () {
		if (!initialised) return;

		// predict state
		x = F * x;
		
		// Predict error covariance
		P = F * P * F.transpose() + Q;
		
		age++;
	}

	// Update with new measurement (called when detection available)
	void update (const NvOSD_RectParams& bbox) {
		if (!initialised) {
			init(bbox);
			return;
		}

		// Measurement
		Eigen::VectorXd z(4);
		z	<<	bbox.left + bbox.width / 2.0f,
				bbox.top + bbox.height / 2.0f,
				bbox.width,
				bbox.height;
				
		// Innovation vector
		Eigen::VectorXd y = z - H * x;
		
		// Innovation covariance
		Eigen::MatrixXd S = H * P * H.transpose() + R;
		
		// Kalman gain
		Eigen::MatrixXd K = P * H.transpose() * S.inverse();
		
		// Update state estimate
		x = x + K * y;
		
		// Update error covariance
		Eigen::MatrixXd I = Eigen::MatrixXd::Identity(8, 8);
		P = (I - K * H) * P;
		
		age = 0; // Reset age since we got a measurement

	}
	
	// Get current state as bounding box
	NvOSD_RectParams getBBox() const {
		NvOSD_RectParams bbox;
		bbox.left = x(0) - x(2) / 2.0f;
		bbox.top = x(1) - x(3) / 2.0f;
		bbox.width = x(2);
		bbox.height = x(3);
		
		bbox.border_width = 4;
		bbox.border_color.red = 0.0;
		bbox.border_color.green = 1.0;
		bbox.border_color.blue = 0.0;
		bbox.border_color.alpha = 1.0;
				
		return bbox;
	}

	// Get velocity magnitude
	float getVelocity() const {
		return std::sqrt(x(4)*x(4) + x(5)*x(5));
	}
	
	// Get age (frames since last measurement)
	int getAge () const {
		return age;
	}
	
private:

	Eigen::VectorXd x;	// State vector
	Eigen::MatrixXd F;	// State transition matrix
	Eigen::MatrixXd H;	// Measurement (model) matrix
	Eigen::MatrixXd Q;	// Process noise covariance
	Eigen::MatrixXd R;	// Measurement noise covariance
	Eigen::MatrixXd P;	// Error covariance

	bool initialised;
	int age;			// Number of frames since last measurement

};


// history of object tracking
struct BBoxHistory {
	BBoxKalmanFilter kf;				// Kalman filter for smoothing
    std::deque<NvOSD_RectParams> boxes;	// Recent bounding boxes
    int frame_count;					// Total frames this object has been tracked
    bool currently_visible;				// Is object still being detected
    int last_seen_frame;				// Last frame number where object was detected
    
    // This gets called automatically when map creates new entry
    // if g_object_histories[obj_id] doesn't exist and is called.
    BBoxHistory() : frame_count(0), currently_visible(false), last_seen_frame(0) {}
};

