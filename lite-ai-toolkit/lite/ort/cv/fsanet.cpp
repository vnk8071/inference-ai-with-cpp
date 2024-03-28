//
// Created by DefTruth on 2021/3/14.
//

#include "fsanet.h"
#include "lite/ort/core/ort_utils.h"

using ortcv::FSANet;

Ort::Value FSANet::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  // 0. padding
  const int h = mat.rows;
  const int w = mat.cols;
  const int nh = static_cast<int>((static_cast<float>(h) + pad * static_cast<float>(h)));
  const int nw = static_cast<int>((static_cast<float>(w) + pad * static_cast<float>(w)));

  const int nx1 = std::max(0, static_cast<int>((nw - w) / 2));
  const int ny1 = std::max(0, static_cast<int>((nh - h) / 2));

  canvas = cv::Mat(nh, nw, CV_8UC3, cv::Scalar(0, 0, 0));
  mat.copyTo(canvas(cv::Rect(nx1, ny1, w, h)));

  cv::resize(canvas, canvas, cv::Size(input_width, input_height));
  ortcv::utils::transform::normalize_inplace(canvas, 127.5, 1.f / 127.5f);

  return ortcv::utils::transform::create_tensor(
      canvas, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void FSANet::detect(const cv::Mat &mat, types::EulerAngles &euler_angles)
{

  Ort::Value input_tensor = this->transform(mat);

  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), 1
  );

  const float *angles_ptr = output_tensors.front().GetTensorMutableData<float>();

  euler_angles.yaw = angles_ptr[0];
  euler_angles.pitch = angles_ptr[1];
  euler_angles.roll = angles_ptr[2];
  euler_angles.flag = true;
}


















































