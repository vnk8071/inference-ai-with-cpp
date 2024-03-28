//
// Created by DefTruth on 2021/3/30.
//
#include "ort_handler.h"

#ifdef LITE_WIN32
#include "lite/utils.h"
#endif

using core::BasicOrtHandler;
using core::BasicMultiOrtHandler;

//**************************************BasicOrtHandler************************************/
BasicOrtHandler::BasicOrtHandler(
    const std::string &_onnx_path, unsigned int _num_threads) :
    log_id(_onnx_path.data()), num_threads(_num_threads)
{
#ifdef LITE_WIN32
  std::wstring _w_onnx_path(lite::utils::to_wstring(_onnx_path));
  onnx_path = _w_onnx_path.data();
#else
  onnx_path = _onnx_path.data();
#endif
  initialize_handler();
}

void BasicOrtHandler::initialize_handler()
{
  ort_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, log_id);
  // 0. session options
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(num_threads);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options.SetLogSeverityLevel(4);

  // GPU compatiable.
  // OrtCUDAProviderOptions provider_options;
  // session_options.AppendExecutionProvider_CUDA(provider_options);
#ifdef USE_CUDA
  OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0); // C API stable.
#endif
  // 1. session
  ort_session = new Ort::Session(ort_env, onnx_path, session_options);

  Ort::AllocatorWithDefaultOptions allocator;
  // 2. input name & input dims
  input_name = ort_session->GetInputName(0, allocator);
  input_node_names.resize(1);
  input_node_names[0] = input_name;
  // 3. type info.
  Ort::TypeInfo type_info = ort_session->GetInputTypeInfo(0);
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  input_tensor_size = 1;
  input_node_dims = tensor_info.GetShape();
  for (unsigned int i = 0; i < input_node_dims.size(); ++i)
    input_tensor_size *= input_node_dims.at(i);
  input_values_handler.resize(input_tensor_size);
  // 4. output names & output dimms
  num_outputs = ort_session->GetOutputCount();
  output_node_names.resize(num_outputs);
  for (unsigned int i = 0; i < num_outputs; ++i)
  {
    output_node_names[i] = ort_session->GetOutputName(i, allocator);
    Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_node_dims.push_back(output_dims);
  }
#if LITEORT_DEBUG
  this->print_debug_string();
#endif
}

BasicOrtHandler::~BasicOrtHandler()
{
  if (ort_session)
    delete ort_session;
  ort_session = nullptr;
}

void BasicOrtHandler::print_debug_string()
{
  std::cout << "LITEORT_DEBUG LogId: " << onnx_path << "\n";
  std::cout << "=============== Input-Dims ==============\n";
  for (unsigned int i = 0; i < input_node_dims.size(); ++i)
    std::cout << "input_node_dims: " << input_node_dims.at(i) << "\n";
  std::cout << "=============== Output-Dims ==============\n";
  for (unsigned int i = 0; i < num_outputs; ++i)
    for (unsigned int j = 0; j < output_node_dims.at(i).size(); ++j)
      std::cout << "Output: " << i << " Name: "
                << output_node_names.at(i) << " Dim: " << j << " :"
                << output_node_dims.at(i).at(j) << std::endl;
  std::cout << "========================================\n";
}

//***********************************BasicMultiOrtHandler**********************************/
BasicMultiOrtHandler::BasicMultiOrtHandler(
    const std::string &_onnx_path, unsigned int _num_threads) :
    log_id(_onnx_path.data()), num_threads(_num_threads)
{
#ifdef LITE_WIN32
  std::wstring _w_onnx_path(lite::utils::to_wstring(_onnx_path));
  onnx_path = _w_onnx_path.data();
#else
  onnx_path = _onnx_path.data();
#endif
  initialize_handler();
}

void BasicMultiOrtHandler::initialize_handler()
{
  ort_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, log_id);
  // 0. session options
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(num_threads);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options.SetLogSeverityLevel(4);

  // 1. session
  // GPU compatiable.
  // OrtCUDAProviderOptions provider_options;
  // session_options.AppendExecutionProvider_CUDA(provider_options);
#ifdef USE_CUDA
  OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0); // C API stable.
#endif
  ort_session = new Ort::Session(ort_env, onnx_path, session_options);
  Ort::AllocatorWithDefaultOptions allocator;
  // 2. input name & input dims
  num_inputs = ort_session->GetInputCount();
  input_node_names.resize(num_inputs);
  for (unsigned int i = 0; i < num_inputs; ++i)
  {
    input_node_names[i] = ort_session->GetInputName(i, allocator);
    Ort::TypeInfo type_info = ort_session->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto input_dims = tensor_info.GetShape();
    input_node_dims.push_back(input_dims);
    size_t tensor_size = 1;
    for (unsigned int j = 0; j < input_dims.size(); ++j)
      tensor_size *= input_dims.at(j);
    input_tensor_sizes.push_back(tensor_size);
    input_values_handlers.push_back(std::vector<float>(tensor_size));
  }
  // 4. output names & output dimms
  num_outputs = ort_session->GetOutputCount();
  output_node_names.resize(num_outputs);
  for (unsigned int i = 0; i < num_outputs; ++i)
  {
    output_node_names[i] = ort_session->GetOutputName(i, allocator);
    Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_node_dims.push_back(output_dims);
  }
#if LITEORT_DEBUG
  this->print_debug_string();
#endif
}

BasicMultiOrtHandler::~BasicMultiOrtHandler()
{
  if (ort_session)
    delete ort_session;
  ort_session = nullptr;
}

void BasicMultiOrtHandler::print_debug_string()
{
  std::cout << "LITEORT_DEBUG LogId: " << onnx_path << "\n";
  std::cout << "=============== Input-Dims ==============\n";
  for (unsigned int i = 0; i < num_inputs; ++i)
    for (unsigned int j = 0; j < input_node_dims.at(i).size(); ++j)
      std::cout << "Input: " << i << " Name: "
                << input_node_names.at(i) << " Dim: " << j << " :"
                << input_node_dims.at(i).at(j) << std::endl;
  std::cout << "=============== Output-Dims ==============\n";
  for (unsigned int i = 0; i < num_outputs; ++i)
    for (unsigned int j = 0; j < output_node_dims.at(i).size(); ++j)
      std::cout << "Output: " << i << " Name: "
                << output_node_names.at(i) << " Dim: " << j << " :"
                << output_node_dims.at(i).at(j) << std::endl;
  std::cout << "========================================\n";
}