#ifndef ETL_h
#define ETL_h

#include <iostream>
#include <fstream>
#include <vector>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

class ETL
{
    std::string dataset;
    std::string delimiter;
    bool header;

public:
    ETL(std::string data, std::string separator, bool head) : dataset(data), delimiter(separator), header(head)
    {
    }

    vector<std::vector<std::string>> readCSV();
    MatrixXd CSVtoEigen(vector<std::vector<std::string>> dataset, int rows, int cols);

    MatrixXd Normalize(MatrixXd data, bool normalizeTarget);
    auto Mean(MatrixXd data) -> decltype(data.colwise().mean());
    auto Std(MatrixXd data) -> decltype(((data.array().square().colwise().sum()) / (data.rows() - 1)).sqrt());

    std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> TrainTestSplit(MatrixXd data, float train_size);
};

#endif