#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <vector>
#include <chrono>
#include <map>
#include <numeric>
#include <glob.h>
#include <sys/stat.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <bits/stdc++.h>
#include <torch/script.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torchvision/vision.h>

using namespace std;
using namespace cv;
using std::map;
using std::vector;

// For string delimiter
vector<string> split(string s, string delimiter)
{
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != string::npos)
    {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res;
}

// List all files in folder
vector<string> globVector(const string &pattern)
{
    glob_t glob_result;
    glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    vector<string> files;
    for (unsigned int i = 0; i < glob_result.gl_pathc; ++i)
    {
        files.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return files;
}
// Convert image to tensor
torch::IValue get_script_inputs(Mat &img, torch::DeviceType device)
{
    const int height = img.rows;
    const int width = img.cols;
    const int channels = 3;

    torch::Tensor input = torch::from_blob(img.data, {height, width, channels}, torch::kUInt8);
    input = input.to(device, torch::kFloat).permute({2, 0, 1}).contiguous();

    return input;
}

void draw_bbox(int i, Mat frame, at::Tensor bbox, at::Tensor pred_classes, at::Tensor scores, vector<string> class_names)
{
    Scalar colors[3] = {(0, 0, 255),
                        (0, 255, 0),
                        (255, 0, 0)};
    int box_x = bbox[i][0].item<int>();
    int box_y = bbox[i][1].item<int>();
    int box_width = bbox[i][2].item<int>();
    int box_height = bbox[i][3].item<int>();
    string class_name = class_names[pred_classes[i].item<int>()];
    string result = to_string(static_cast<int>(scores[i].item<float>() * 100));
    if (class_name == "no_mask")
    {
        rectangle(frame, Point(box_x, box_y), Point(box_width, box_height), Scalar(255, 0, 0), 2);
        putText(frame, class_name + " " + result + "%", Point(box_x, box_y - 5), FONT_HERSHEY_DUPLEX, 0.5, Scalar(255, 0, 0), 2);
    }
    else if (class_name == "mask")
    {
        rectangle(frame, Point(box_x, box_y), Point(box_width, box_height), Scalar(0, 255, 0), 2);
        putText(frame, class_name + " " + result + "%", Point(box_x, box_y - 5), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 255, 0), 2);
    }
    else if (class_name == "incorrect_mask")
    {
        rectangle(frame, Point(box_x, box_y), Point(box_width, box_height), Scalar(0, 0, 255), 2);
        putText(frame, class_name + " " + result + "%", Point(box_x, box_y - 5), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 255), 2);
    }
}

// Custom class for deepcopy
class DeepCopy
{
private:
    at::Tensor *bbox_copy;
    at::Tensor *pred_classes_copy;
    at::Tensor *scores_copy;

public:
    DeepCopy()
    {
        bbox_copy = new at::Tensor;
        pred_classes_copy = new at::Tensor;
        scores_copy = new at::Tensor;
    }
    void set_copy(at::Tensor bbox, at::Tensor pred_classes, at::Tensor scores)
    {
        *bbox_copy = bbox;
        *pred_classes_copy = pred_classes;
        *scores_copy = scores;
    }
    vector<at::Tensor> get_copy()
    {
        return {*bbox_copy, *pred_classes_copy, *scores_copy};
    }
    DeepCopy(DeepCopy &sample)
    {
        bbox_copy = new at::Tensor;
        *bbox_copy = *(sample.bbox_copy);
        pred_classes_copy = new at::Tensor;
        *pred_classes_copy = *(sample.pred_classes_copy);
        scores_copy = new at::Tensor;
        *scores_copy = *(sample.scores_copy);
    }
};

int main(int argc, const char *argv[])
{
    if (argc != 5)
        throw runtime_error("Usage: ./build/d2 <model_path> <input_path> <output_path> <fps_predict (Default is 0)>");
    cout
        << "Loading model..." << endl;

    // Class of tear break-up pattern
    vector<string> class_names = {"no_mask",
                                  "mask",
                                  "incorrect_mask"};

    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        torch::jit::getBailoutDepth() = 1;
        torch::autograd::AutoGradMode guard(false);
        torch::DeviceType device = torch::kCPU;
        torch::jit::script::Module module;

        // Load input parameters
        module = torch::jit::load(argv[1]);
        assert(module.buffers().size() > 0);
        string input_path = argv[2];
        string output_path = argv[3];
        string fps_predict = argv[4];

        // Create output folder
        const char *dir = (output_path).c_str();
        mkdir(dir, 0777);

        // Declare objects
        DeepCopy objs;
        DeepCopy objs_copy;
        // Declare variables
        string delimiter = "/";
        vector<string> file_name = split(input_path, delimiter);
        auto start_time = chrono::high_resolution_clock::now();
        if (stoi(fps_predict) == 0)
        {
            cout << "--------------IMAGE-PROCESSING---------------" << endl;
            Mat image = cv::imread(input_path);
            if (image.empty())
            {
                cout << "Failed imread(): image not found" << endl;
                // don't let the execution continue, else imshow() will crash.
            }
            cout << "Width : " << image.size[1] << endl;
            cout << "Height: " << image.size[0] << endl;
            cout << "File name: " << file_name.back() << endl;
            cout << "Image save: " << output_path << file_name.back() << endl;

            at::Tensor bbox, pred_classes, scores;
            auto inputs = get_script_inputs(image, device);

            // Run the model
            auto output = module.forward({inputs});
            auto outputs = output.toTuple()->elements();
            bbox = outputs[0].toTensor();
            pred_classes = outputs[1].toTensor();
            scores = outputs[2].toTensor();

            // Write bounding box into image
            for (int i = 0; i < pred_classes.size(-1); i++)
            {
                draw_bbox(i, image, bbox, pred_classes, scores, class_names);
            }
            cv::imwrite(output_path + file_name.back(), image);
            auto end_time = chrono::high_resolution_clock::now();
            auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time)
                          .count();

            cout << "\n🔚 Total time predict: "
                 << int(ms * 1.0 / 1e6) % 60 << " seconds" << endl;
        }
        else if (stoi(fps_predict) >= 1)
        {
            cout << "--------------VIDEO-PROCESSING---------------" << endl;
            VideoCapture cap(input_path);

            // Check if video opened successfully
            if (!cap.isOpened())
            {
                cout << "Error opening video" << endl;
                return -1;
            }
            int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
            double fps = cap.get(cv::CAP_PROP_FPS);
            int num_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);

            cout << "Processing with video name: " << file_name.back() << endl;
            VideoWriter video(output_path + file_name.back(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(frame_width, frame_height));
            cout << "Metadata | W: " << frame_width << " | H: " << frame_height << " | FPS: " << round(fps) << " | F: " << num_frames << endl;
            cout << "Video save: " << output_path << file_name.back() << endl;

            // Loop each frame of video
            for (int f = 0; f < num_frames; f++)
            {
                // Read each frame
                Mat frame;
                cap >> frame;
                at::Tensor bbox, pred_classes, scores;

                // Do with normal target
                if (stoi(fps_predict) == 1)
                {
                    auto inputs = get_script_inputs(frame, device);

                    // Run the model
                    auto output = module.forward({inputs});
                    auto outputs = output.toTuple()->elements();
                    bbox = outputs[0].toTensor();
                    pred_classes = outputs[1].toTensor();
                    scores = outputs[2].toTensor();

                    // Write bounding box into video
                    for (int i = 0; i < pred_classes.size(-1); i++)
                    {
                        draw_bbox(i, frame, bbox, pred_classes, scores, class_names);
                    }
                    video.write(frame);
                }

                // Do with downsampling
                else if (stoi(fps_predict) > 1)
                {
                    if (f % int(round(fps) / stoi(fps_predict)) == 0)
                    {
                        auto inputs = get_script_inputs(frame, device);

                        // Run the model
                        auto output = module.forward({inputs});
                        auto outputs = output.toTuple()->elements();
                        bbox = outputs[0].toTensor();
                        pred_classes = outputs[1].toTensor();
                        scores = outputs[2].toTensor();
                        objs.set_copy(bbox, pred_classes, scores);

                        // Write bounding box into video
                        for (int i = 0; i < pred_classes.size(-1); i++)
                        {
                            draw_bbox(i, frame, bbox, pred_classes, scores, class_names);
                        }
                    }
                    else
                    {
                        vector<at::Tensor> predicts = objs_copy.get_copy();
                        // Write bounding box into video
                        for (int i = 0; i < predicts[1].size(-1); i++)
                        {
                            draw_bbox(i, frame, predicts[0], predicts[1], predicts[2], class_names);
                        }
                    }

                    objs_copy = objs;
                    video.write(frame);
                }
                cout << "\r🔜 Processing... [" << ceil(f + 1) << '/' << num_frames << "]" << flush;
            }

            // When everything done, release the video capture and write object
            cap.release();
            video.release();
            auto end_time = chrono::high_resolution_clock::now();
            auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time)
                          .count();

            cout << "\n🔚 Total time predict: "
                 << int(ms * 1.0 / 1e6) / 60 << " minutes and " << int(ms * 1.0 / 1e6) % 60 << " seconds" << endl;

            cout << "Time per frame: " << ms * 1.0 / (1e6 * num_frames) << "(s) with " << num_frames << "(frames)" << endl;
        }
        else
        {
            cout << "Follow fps in description" << endl;
        }

        cout << "--------------------------------------------------\n"
             << endl;
        cout << "DONE ✅" << endl;
        return 0;
    }

    // Avoid run programming without model
    catch (const c10::Error &e)
    {
        cout << "Error loading the model\n"
             << e.what() << endl;
        return -1;
    }
}
