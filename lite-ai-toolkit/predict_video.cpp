#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>

#include "lite/lite.h"

int main() {
    auto *detector = new lite::cv::face::detect::SCRFD("../model/scrfd_500m_bnkps_shape640x640.onnx");

    std::string video_path = "../data/videov.mp4";
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    auto start_time_predict = std::chrono::steady_clock::now();
    std::string output_path = "../results/output_cpp_video.mp4";
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::VideoWriter out(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 20, cv::Size(frame_width, frame_height));

    cv::Mat frame;
    int n_frame = 0;
    while (cap.read(frame)) {
        // Break the loop when counting frames reaches 1000
        if (n_frame == 1000) {
            break;
        }
        n_frame++;
        auto start_time = std::chrono::steady_clock::now();

        std::vector<lite::types::BoxfWithLandmarks> detected_boxes_kps;
        detector->detect(frame, detected_boxes_kps, 0.3f);

        for (const auto &box_kps : detected_boxes_kps) {
            lite::utils::draw_boxes_with_landmarks_inplace(frame, detected_boxes_kps);
        }
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        double fps = 1000.0 / duration;
        cv::putText(frame, std::to_string(fps), cv::Point(7, 70), cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(100, 255, 0), 3, cv::LINE_AA);

        out.write(frame);
    }

    cap.release();
    out.release();
    auto end_time_predict = std::chrono::steady_clock::now();
    auto duration_predict = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_predict - start_time_predict).count();
    std::cout << "Execution time: " << duration_predict << " ms\n";
    return 0;
}