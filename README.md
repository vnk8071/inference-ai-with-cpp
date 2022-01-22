# AI-on-Cpp

The goal of these projects is to improve model training time and give optimal inference results.

Use CMake to configure on Linux system for these projects.

## Logistic Regression on C++

*Reference*: https://github.com/coding-ai/machine_learning_cpp

### Install independences
1. Boost

Boost Libraries are set of peer-reviewed and mostly header-only libraries used by many projects and applications. They are regarded as an extension of the C++ standard library and even many features from the C++ standard come from Boost. Boost provides many facilities for numerical computing; parsers; template metaprogramming; network sockets TCP/IP and UDP; inter process communication; shared memory and so on.

Website: https://www.boost.org/

```bash
sudo apt-get install libboost-all-dev
```

2. Eigen

Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.

```bash
bash eigen-setup.sh
```

It's take a lot of your minutes.

3. Pre-processing (Optional with python)

Because in C++ it is very difficult to deal with table data preprocessing. Instead, we preprocessed the data with the file <stroke-data-processed.csv> or the user can run the script below to perform this preprocessing step.

```bash
conda create -n strokecpp python=3.7
conda activate strokecpp
pip install -r requirements.txt
```

### Compile project:

*Change ${PWD}/eigen* to the eigen path of the project.

```bash
bash main-project.sh
```

### Run training
```bash
./build/stroke ./dataset/stroke-data-processed.csv "," true
```

With arg:
- arg[1] - Path to dataset
- arg[2] - Delimiter of csv file (Default: ",")
- arg[3] - Header of csv file

## Try your best ^^
