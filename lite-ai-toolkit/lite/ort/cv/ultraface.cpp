//
// Created by DefTruth on 2021/3/14.
//

#include "ultraface.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::UltraFace;

Ort::Value UltraFace::transform(const cv::Mat &mat)
{

  cv::Mat canvas;
  cv::cvtColor(mat, canvas, cv::COLOR_BGR2RGB);
  cv::resize(canvas, canvas, cv::Size(input_node_dims.at(3),
                                      input_node_dims.at(2)));
  // (640,480) | (320,240) | (w,h) 1xCXHXW

  ortcv::utils::transform::normalize_inplace(canvas, mean_val, scale_val); // float32
  return ortcv::utils::transform::create_tensor(
      canvas, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void UltraFace::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                       float score_threshold, float iou_threshold, unsigned int topk,
                       unsigned int nms_type)
{
  if (mat.empty()) return;
  // this->transform(mat);
  float img_height = static_cast<float>(mat.rows);
  float img_width = static_cast<float>(mat.cols);

  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat);
  // 2. inference scores & boxes.
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  // 3. rescale & exclude.
  std::vector<types::Boxf> bbox_collection;
  this->generate_bboxes(bbox_collection, output_tensors, score_threshold, img_height, img_width);
  // 4. hard|blend nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

void UltraFace::generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                                std::vector<Ort::Value> &output_tensors,
                                float score_threshold, float img_height,
                                float img_width)
{

  Ort::Value &scores = output_tensors.at(0);
  Ort::Value &boxes = output_tensors.at(1);
  auto scores_dims = output_node_dims.at(0); // (1,n,2)
  const unsigned int num_anchors = scores_dims.at(1); // n = 17640 (640x480)

  bbox_collection.clear();
  unsigned int count = 0;
  for (unsigned int i = 0; i < num_anchors; ++i)
  {
    float confidence = scores.At<float>({0, i, 1});
    if (confidence < score_threshold) continue;
    types::Boxf box;
    box.x1 = boxes.At<float>({0, i, 0}) * img_width;
    box.y1 = boxes.At<float>({0, i, 1}) * img_height;
    box.x2 = boxes.At<float>({0, i, 2}) * img_width;
    box.y2 = boxes.At<float>({0, i, 3}) * img_height;
    box.score = confidence;
    box.label_text = "face";
    box.label = 1;
    box.flag = true;
    bbox_collection.push_back(box);

    count += 1; // limit boxes for nms.
    if (count > max_nms)
      break;
  }
#if LITEORT_DEBUG
  std::cout << "detected num_anchors: " << num_anchors << "\n";
  std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif
}

void UltraFace::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                    float iou_threshold, unsigned int topk, unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
  else lite::utils::hard_nms(input, output, iou_threshold, topk);
}
























