//
// Created by DefTruth on 2021/10/18.
//

#include "tnn_yolop.h"
#include "lite/utils.h"

using tnncv::TNNYOLOP;

TNNYOLOP::TNNYOLOP(const std::string &_proto_path,
                   const std::string &_model_path,
                   unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}

void TNNYOLOP::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                              int target_height, int target_width,
                              YOLOPScaleParams &scale_params)
{
  if (mat.empty()) return;
  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);

  mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
                   cv::Scalar(114, 114, 114));
  // scale ratio (new / old) new_shape(h,w)
  float w_r = (float) target_width / (float) img_width;
  float h_r = (float) target_height / (float) img_height;
  float r = std::min(w_r, h_r);
  // compute padding
  int new_unpad_w = static_cast<int>((float) img_width * r); // floor
  int new_unpad_h = static_cast<int>((float) img_height * r); // floor
  int pad_w = target_width - new_unpad_w; // >=0
  int pad_h = target_height - new_unpad_h; // >=0

  int dw = pad_w / 2;
  int dh = pad_h / 2;

  // resize with unscaling
  cv::Mat new_unpad_mat = mat.clone();
  cv::resize(new_unpad_mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
  new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

  // record scale params.
  scale_params.r = r;
  scale_params.dw = dw;
  scale_params.dh = dh;
  scale_params.new_unpad_w = new_unpad_w;
  scale_params.new_unpad_h = new_unpad_h;
  scale_params.flag = true;
}

void TNNYOLOP::transform(const cv::Mat &mat_rs)
{
  // push into input_mat
  input_mat = std::make_shared<tnn::Mat>(input_device_type, tnn::N8UC3,
                                         input_shape, (void *) mat_rs.data);
  if (!input_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNYOLOP::detect(const cv::Mat &mat,
                      std::vector<types::Boxf> &detected_boxes,
                      types::SegmentContent &da_seg_content,
                      types::SegmentContent &ll_seg_content,
                      float score_threshold, float iou_threshold,
                      unsigned int topk, unsigned int nms_type)
{
  if (mat.empty()) return;
  float img_height = static_cast<float>(mat.rows);
  float img_width = static_cast<float>(mat.cols);

  // resize & unscale
  cv::Mat mat_rs;
  YOLOPScaleParams scale_params;
  this->resize_unscale(mat, mat_rs, input_height, input_width, scale_params);

  if ((!scale_params.flag) || mat_rs.empty()) return;
  // 1. make input mat
  cv::Mat mat_rs_;
  cv::cvtColor(mat_rs, mat_rs_, cv::COLOR_BGR2RGB);
  this->transform(mat_rs_);
  // 2. set input_mat
  tnn::MatConvertParam input_cvt_param;
  input_cvt_param.scale = scale_vals;
  input_cvt_param.bias = bias_vals;

  tnn::Status status;
  status = instance->SetInputMat(input_mat, input_cvt_param);
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->SetInputMat failed!:"
              << status.description().c_str() << "\n";
#endif
    return;
  }
  // 3. forward
  status = instance->Forward();
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->Forward failed!:"
              << status.description().c_str() << "\n";
#endif
    return;
  }
  // 4. rescale & fetch da|ll seg.
  std::vector<types::Boxf> bbox_collection;
  this->generate_bboxes_da_ll(scale_params, instance, bbox_collection,
                              da_seg_content, ll_seg_content, score_threshold,
                              img_height, img_width);
  // 5. hard|blend nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

void TNNYOLOP::generate_bboxes_da_ll(const YOLOPScaleParams &scale_params,
                                     std::shared_ptr<tnn::Instance> &_instance,
                                     std::vector<types::Boxf> &bbox_collection,
                                     types::SegmentContent &da_seg_content,
                                     types::SegmentContent &ll_seg_content,
                                     float score_threshold, float img_height,
                                     float img_width)
{
  std::shared_ptr<tnn::Mat> det_out_mat;
  std::shared_ptr<tnn::Mat> da_seg_out_mat;
  std::shared_ptr<tnn::Mat> ll_seg_out_mat;
  tnn::MatConvertParam cvt_param;
  tnn::Status status_det_out;
  tnn::Status status_da_seg_out;
  tnn::Status status_ll_seg_out;

  // (1,n,6=5+1=cxcy+cwch+obj_conf+cls_conf) (1,2,640,640) (1,2,640,640)
  status_det_out    = _instance->GetOutputMat(det_out_mat, cvt_param, "det_out", output_device_type);
  status_da_seg_out = _instance->GetOutputMat(da_seg_out_mat, cvt_param, "drive_area_seg", output_device_type);
  status_ll_seg_out = _instance->GetOutputMat(ll_seg_out_mat, cvt_param, "lane_line_seg", output_device_type);

  if (status_det_out != tnn::TNN_OK || status_da_seg_out != tnn::TNN_OK
      || status_ll_seg_out != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->GetOutputMat failed!:"
              << status_det_out.description().c_str() << ": "
              << status_ll_seg_out.description().c_str() << ": "
              << status_da_seg_out.description().c_str() << "\n";
#endif
    return;
  }

  auto det_dims = det_out_mat->GetDims();
  const unsigned int num_anchors = det_dims.at(1); // n = ?

  float r = scale_params.r;
  int dw = scale_params.dw;
  int dh = scale_params.dh;
  int new_unpad_w = scale_params.new_unpad_w;
  int new_unpad_h = scale_params.new_unpad_h;

  // generate bounding boxes.
  bbox_collection.clear();
  unsigned int count = 0;
  for (unsigned int i = 0; i < num_anchors; ++i)
  {
    const float *offset_obj_cls_ptr = (float *) det_out_mat->GetData() + (i * 6);
    float obj_conf = offset_obj_cls_ptr[4];
    if (obj_conf < score_threshold) continue; // filter first.

    unsigned int label = 1;  // 1 class only
    float cls_conf = offset_obj_cls_ptr[5];
    float conf = obj_conf * cls_conf; // cls_conf (0.,1.)
    if (conf < score_threshold) continue; // filter

    float cx = offset_obj_cls_ptr[0];
    float cy = offset_obj_cls_ptr[1];
    float w = offset_obj_cls_ptr[2];
    float h = offset_obj_cls_ptr[3];
    float x1 = ((cx - w / 2.f) - (float) dw) / r;
    float y1 = ((cy - h / 2.f) - (float) dh) / r;
    float x2 = ((cx + w / 2.f) - (float) dw) / r;
    float y2 = ((cy + h / 2.f) - (float) dh) / r;

    types::Boxf box;
    // de-padding & rescaling
    box.x1 = std::max(0.f, x1);
    box.y1 = std::max(0.f, y1);
    box.x2 = std::min(x2, (float) img_width);
    box.y2 = std::min(y2, (float) img_height);
    box.score = conf;
    box.label = label;
    box.label_text = "traffic car";
    box.flag = true;
    bbox_collection.push_back(box);

    count += 1; // limit boxes for nms.
    if (count > max_nms)
      break;
  }
#if LITETNN_DEBUG
  std::cout << "detected num_anchors: " << num_anchors << "\n";
  std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif

  // generate da && ll seg.
  da_seg_content.names_map.clear();
  da_seg_content.class_mat = cv::Mat(new_unpad_h, new_unpad_w, CV_8UC1, cv::Scalar(0));
  da_seg_content.color_mat = cv::Mat(new_unpad_h, new_unpad_w, CV_8UC3, cv::Scalar(0, 0, 0));
  ll_seg_content.names_map.clear();
  ll_seg_content.class_mat = cv::Mat(new_unpad_h, new_unpad_w, CV_8UC1, cv::Scalar(0));
  ll_seg_content.color_mat = cv::Mat(new_unpad_h, new_unpad_w, CV_8UC3, cv::Scalar(0, 0, 0));

  const unsigned int channel_step = input_height * input_width;
  const float *da_seg_bg_ptr = (float *) da_seg_out_mat->GetData(); // background
  const float *da_seg_fg_ptr = (float *) da_seg_out_mat->GetData() + channel_step; // foreground
  const float *ll_seg_bg_ptr = (float *) ll_seg_out_mat->GetData(); // background
  const float *ll_seg_fg_ptr = (float *) ll_seg_out_mat->GetData() + channel_step; // foreground

  for (int i = dh; i < dh + new_unpad_h; ++i)
  {
    // row ptr.
    uchar *da_p_class = da_seg_content.class_mat.ptr<uchar>(i - dh);
    uchar *ll_p_class = ll_seg_content.class_mat.ptr<uchar>(i - dh);
    cv::Vec3b *da_p_color = da_seg_content.color_mat.ptr<cv::Vec3b>(i - dh);
    cv::Vec3b *ll_p_color = ll_seg_content.color_mat.ptr<cv::Vec3b>(i - dh);

    for (int j = dw; j < dw + new_unpad_w; ++j)
    {
      // argmax
      float da_bg_prob = da_seg_bg_ptr[i * input_height + j];
      float da_fg_prob = da_seg_fg_ptr[i * input_height + j];
      float ll_bg_prob = ll_seg_bg_ptr[i * input_height + j];
      float ll_fg_prob = ll_seg_fg_ptr[i * input_height + j];
      unsigned int da_label = da_bg_prob < da_fg_prob ? 1 : 0;
      unsigned int ll_label = ll_bg_prob < ll_fg_prob ? 1 : 0;

      if (da_label == 1)
      {
        // assign label for pixel(i,j)
        da_p_class[j - dw] = 1 * 255;  // 255 indicate drivable area, for post resize
        // assign color for detected class at pixel(i,j).
        da_p_color[j - dw][0] = 0;
        da_p_color[j - dw][1] = 255;  // green
        da_p_color[j - dw][2] = 0;
        // assign names map
        da_seg_content.names_map[255] = "drivable area";
      }

      if (ll_label == 1)
      {
        // assign label for pixel(i,j)
        ll_p_class[j - dw] = 1 * 255;  // 255 indicate lane line, for post resize
        // assign color for detected class at pixel(i,j).
        ll_p_color[j - dw][0] = 0;
        ll_p_color[j - dw][1] = 0;
        ll_p_color[j - dw][2] = 255;  // red
        // assign names map
        ll_seg_content.names_map[255] = "lane line";
      }

    }
  }
  // resize to original size.
  const unsigned int img_h = static_cast<unsigned int>(img_height);
  const unsigned int img_w = static_cast<unsigned int>(img_width);
  // da_seg_mask 255 or 0
  cv::resize(da_seg_content.class_mat, da_seg_content.class_mat,
             cv::Size(img_w, img_h), cv::INTER_LINEAR);
  cv::resize(da_seg_content.color_mat, da_seg_content.color_mat,
             cv::Size(img_w, img_h), cv::INTER_LINEAR);
  // ll_seg_mask 255 or 0
  cv::resize(ll_seg_content.class_mat, ll_seg_content.class_mat,
             cv::Size(img_w, img_h), cv::INTER_LINEAR);
  cv::resize(ll_seg_content.color_mat, ll_seg_content.color_mat,
             cv::Size(img_w, img_h), cv::INTER_LINEAR);

  da_seg_content.flag = true;
  ll_seg_content.flag = true;
}

void TNNYOLOP::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                   float iou_threshold, unsigned int topk, unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
  else lite::utils::hard_nms(input, output, iou_threshold, topk);
}










