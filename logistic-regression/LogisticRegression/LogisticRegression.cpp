#include "LogisticRegression.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <map>
#include <list>

using namespace std;
using namespace Eigen;

MatrixXd LogisticRegression::Sigmoid(MatrixXd Z)
{
    return 1 / (1 + (-Z.array()).exp());
}

tuple<MatrixXd, double, double> LogisticRegression::Propagate(MatrixXd W, double b, MatrixXd X, MatrixXd y, double lambda)
{

    int m = y.rows();

    MatrixXd Z = (W.transpose() * X.transpose()).array() + b;
    MatrixXd A = Sigmoid(Z);

    auto cross_entropy = -(y.transpose() * (VectorXd)A.array().log().transpose() + ((VectorXd)(1 - y.array())).transpose() * (VectorXd)(1 - A.array()).log().transpose()) / m;

    double l2_reg_cost = W.array().pow(2).sum() * (lambda / (2 * m));

    double cost = static_cast<const double>((cross_entropy.array()[0])) + l2_reg_cost;

    MatrixXd dw = (MatrixXd)(((MatrixXd)(A - y.transpose()) * X) / m) + ((MatrixXd)(lambda / m * W)).transpose();

    double db = (A - y.transpose()).array().sum() / m;

    return make_tuple(dw, db, cost);
}

tuple<MatrixXd, double, MatrixXd, double, list<double>> LogisticRegression::Optimize(MatrixXd W, double b, MatrixXd X, MatrixXd y, int num_iter, double learning_rate, double lambda, bool log_cost)
{

    list<double> costsList;
    MatrixXd dw;
    double db, cost;

    for (int i = 0; i < num_iter; i++)
    {
        tuple<MatrixXd, double, double> propagate = Propagate(W, b, X, y, lambda);
        tie(dw, db, cost) = propagate;

        W = W - (learning_rate * dw).transpose();
        b = b - (learning_rate * db);

        if (i % 100 == 0)
        {
            costsList.push_back(cost);
        }

        if (log_cost && i % 500 == 0)
        {
            cout << "Cost after iteration " << i << ": " << cost << endl;
        }
    }

    return make_tuple(W, b, dw, db, costsList);
}

MatrixXd LogisticRegression::Predict(MatrixXd W, double b, MatrixXd X)
{

    int m = X.rows();

    MatrixXd y_pred = VectorXd::Zero(m).transpose();

    MatrixXd Z = (W.transpose() * X.transpose()).array() + b;
    MatrixXd A = Sigmoid(Z);

    for (int i = 0; i < A.cols(); i++)
    {
        if (A(0, i) <= 0.5)
        {
            y_pred(0, i) = 0;
        }
        else
        {
            y_pred(0, i) = 1;
        }
    }

    return y_pred.transpose();
}