#ifndef NN_CLASSIFIER_H
#define NN_CLASSIFIER_H

#include "Matrix.h"

namespace nn {

Matrix forward(const Matrix &X, const Matrix &W);

Matrix classify(const Matrix &X, const Matrix &W);

Matrix train(const Matrix &X, const Matrix &Y, int Iterations, double LR);

void accuracy(const Matrix &Y, const Matrix &Ypred);

} // namespace nn

#endif // NN_CLASSIFIER_H
