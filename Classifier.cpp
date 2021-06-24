#include "Matrix.h"
#include "Mnist.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>

namespace nn {
Matrix forward(const Matrix &X, const Matrix &W) { return sigmoid(mult(X, W)); }

Matrix classify(const Matrix &X, const Matrix &W) {
  auto Ypred = nn::forward(X, W);
  auto Labels = createMatrix(Ypred.size(), 1);
  for (size_t Row = 0; Row < Ypred.size(); ++Row) {
    Labels[Row][0] = std::max_element(Ypred[Row].begin(), Ypred[Row].end()) -
                     Ypred[Row].begin();
  }
  return Labels;
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

Matrix oneHotEncodeMnist(const Matrix &Labels) {
  auto Encoded = createMatrix(Labels.size(), 10);
  for (size_t Row = 0; Row < Labels.size(); ++Row) {
    Encoded[Row][static_cast<int>(Labels[Row][0])] = 1;
  }
  return Encoded;
}

} // namespace nn

int main() {
  auto TrainImages = nn::readImageFile("train-images.idx3-ubyte");
  auto TrainLabels = nn::readLabelFile("train-labels.idx1-ubyte");
  auto EncodedLabels = nn::oneHotEncodeMnist(TrainLabels);
  for (size_t Idx = 0; Idx < 5; ++Idx) {
    nn::printImageAndLabel(TrainImages, TrainLabels, Idx);
  }

  const auto W = nn::train(TrainImages, EncodedLabels, 20, 0.00001);

  auto TestImages = nn::readImageFile("t10k-images.idx3-ubyte");
  auto TestLabels = nn::readLabelFile("t10k-labels.idx1-ubyte");
  nn::accuracy(TestLabels, nn::classify(TestImages, W));

  return 0;
}
