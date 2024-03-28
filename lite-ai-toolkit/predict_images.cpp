#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>

#include "lite/lite.h"

int main() {
    auto *detector = new lite::cv::face::detect::SCRFD("../model/scrfd_500m_bnkps_shape640x640.onnx");
    std::vector<std::string> img_paths = {"../data/image.jpg"};

    for (const auto &img_path : img_paths) {
        cv::Mat img = cv::imread(img_path);
        std::cout << "Input img size: " << img.size() << std::endl;

        for (int i = 0; i < 1; ++i) {
            auto ta = std::chrono::steady_clock::now();

            std::vector<lite::types::BoxfWithLandmarks> detected_boxes_kps;
            detector->detect(img, detected_boxes_kps, 0.3f);
            
            auto tb = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tb - ta).count();
            std::cout << "all cost: " << duration << " ms\n";

            lite::utils::draw_boxes_with_landmarks_inplace(img, detected_boxes_kps);

            std::size_t found = img_path.find_last_of("/\\");
            std::string filename = img_path.substr(found+1);
            std::string path_output = "../results/output_cpp_" + filename;
            std::cout << "output:" << path_output << std::endl;
            cv::imwrite(path_output, img);
        }
    }
    return 0;
}
