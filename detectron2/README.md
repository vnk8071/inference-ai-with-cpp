# AI-on-Cpp

The goal of these projects is to improve model training time and give optimal inference results.

Use CMake to configure on Linux system for these projects.

## Inference Detectron2 on C++

```bash
git clone https://github.com/vnk8071/AI-on-Cpp.git
cd Detectron2-Cpp
```

### Create virtual environment
```bash
conda create -n d2detector python=3.8
conda activate d2detector
```

### Install independences
1. Libtorch & Torchvision

Binary distributions of all headers, libraries and CMake configuration files required to depend on PyTorch.
Website: https://pytorch.org/cppdocs/installing.html

```bash
bash libtorch-setup.sh
```

2. Opencv in C++

Follow instruction: https://thecodinginterface.com/blog/opencv-cpp-vscode/


### Download model tracing
```bash
bash get_model.sh
```
- Model saved at model/model_tracing.ts

- Model trained from project and convert to tracing format. 

Link: https://github.com/vnk8071/detectron2-object-detection

### Compile project:

```bash
bash main-project.sh
```

### Run inference
```bash
./build/d2 model/model_tracing.ts sample/1052.jpg inference/ 0
```
With 4 arguments:
- arg[1] - Path model
- arg[2] - Sample of image or video
- arg[3] - Output of inference folder
- arg[4] - frame per second
    - 0 is image
    - 1 is normal video
    - greater than 1 is the number of frame per second to predict

Check output of inference to see the result.

## Try your best ^^