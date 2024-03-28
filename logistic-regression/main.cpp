#include "./ETL/ETL.h"
#include "./LogisticRegression/LogisticRegression.h"
#include <vector>
#include <string>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <list>

using namespace std;
using namespace Eigen;

// Parameters
double lambda = 0.0;
bool log_cost = true;
double learning_rate = 0.01;
int num_iter = 20000;

int main(int argc, char *argv[])
{
    ETL etl(argv[1], argv[2], argv[3]);
    std::vector<std::vector<std::string>> dataset = etl.readCSV();
    int rows = dataset.size();
    int cols = dataset[0].size();
    MatrixXd dataMat = etl.CSVtoEigen(dataset, rows, cols);
    MatrixXd X_train, y_train, X_test, y_test;
    tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> split_data = etl.TrainTestSplit(dataMat, 0.8);
    tie(X_train, y_train, X_test, y_test) = split_data;

    LogisticRegression lr;

    int dims = X_train.cols();
    MatrixXd W = VectorXd::Zero(dims);
    double b = 0.0;

    MatrixXd dw;
    double db;
    list<double> costs;
    tuple<MatrixXd, double, MatrixXd, double, list<double>> optimize = lr.Optimize(W, b, X_train, y_train, num_iter, learning_rate, lambda, log_cost);
    tie(W, b, dw, db, costs) = optimize;

    MatrixXd y_pred_test = lr.Predict(W, b, X_test);
    MatrixXd y_pred_train = lr.Predict(W, b, X_train);

    auto train_acc = (100 - (y_pred_train - y_train).cwiseAbs().mean() * 100);
    auto test_acc = (100 - (y_pred_test - y_test).cwiseAbs().mean() * 100);

    cout << "Train Accuracy: " << train_acc << endl;
    cout << "Test Accuracy: " << test_acc << endl;

    return EXIT_SUCCESS;
}