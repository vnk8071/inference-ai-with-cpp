#ifndef LogisticRegression_h
#define LogisticRegression_h

#include <eigen3/Eigen/Dense>
#include <list>

using namespace std;
using namespace Eigen;

class LogisticRegression
{
public:
    LogisticRegression()
    {
    }

    MatrixXd Sigmoid(MatrixXd Z);

    tuple<MatrixXd, double, double> Propagate(MatrixXd W, double b, MatrixXd X, MatrixXd y, double lambda);
    tuple<MatrixXd, double, MatrixXd, double, list<double>> Optimize(MatrixXd W, double b, MatrixXd X, MatrixXd y, int num_iter, double learning_rate, double lambda, bool log_cost);
    MatrixXd Predict(MatrixXd W, double b, MatrixXd X);
};

#endif