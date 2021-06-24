#include "Classifier.h"
#include "Matrix.h"

#include <iostream>
#include <string>
#include <utility>

namespace nn {
Matrix forward(const Matrix &X, const Matrix &W) { return sigmoid(mult(X, W)); }

Matrix classify(const Matrix &X, const Matrix &W) {
  return round(forward(X, W));
}

double loss(const Matrix &X, const Matrix &Y, const Matrix &W) {
  const auto Ypred = forward(X, W);
  const auto Ones = createMatrix(Y, 1.0);
  const auto Pos = multElements(Y, log(Ypred));
  const auto Neg = multElements(sub(Ones, Y), log(sub(Ones, Ypred)));
  return -sum(add(Pos, Neg)) / X.size();
}

Matrix gradient(const Matrix &X, const Matrix &Y, const Matrix &W) {
  return mult(1.0 / X.size(), mult(transpose(X), sub(forward(X, W), Y)));
}

Matrix train(const Matrix &X, const Matrix &Y, int Iterations, double LR) {
  auto W = createMatrix(X[0].size(), Y[0].size());
  for (size_t It = 0; It < Iterations; ++It) {
    double CurrentLoss = loss(X, Y, W);
    std::cout << "Iteration " << It << " Loss: " << CurrentLoss << "\n";
    W = sub(W, mult(LR, gradient(X, Y, W)));
  }
  return W;
}

void accuracy(const Matrix &Y, const Matrix &Ypred) {
  int CorrectResults = 0;
  for (size_t Idx = 0; Idx < Y.size(); ++Idx) {
    CorrectResults += Ypred[Idx][0] == Y[Idx][0];
  }
  std::cout << "Accuracy: " << (CorrectResults * 100 / Y.size()) << "%";
}

} // namespace nn
