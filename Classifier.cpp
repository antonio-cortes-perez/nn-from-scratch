#include "Matrix.h"
#include "Mnist.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>

namespace nn {
std::pair<Matrix, Matrix> forward(const Matrix &X, const Matrix &W1,
                                  const Matrix &W2) {
  auto Hidden = sigmoid(mult(X, W1));
  auto Ypred = softmax(mult(Hidden, W2));
  return {Ypred, Hidden};
}

Matrix classify(const Matrix &X, const Matrix &W1, const Matrix &W2) {
  auto Ypred = forward(X, W1, W2).first;
  auto Labels = createMatrix(Ypred.size(), 1);
  for (size_t Row = 0; Row < Ypred.size(); ++Row) {
    Labels[Row][0] = std::max_element(Ypred[Row].begin(), Ypred[Row].end()) -
                     Ypred[Row].begin();
  }
  return Labels;
}

double loss(const Matrix &Y, const Matrix &Ypred) {
  return -sum(multElements(Y, log(Ypred))) / Y.size(); // cross-entropy loss
}

std::pair<Matrix, Matrix> backward(const Matrix &X, const Matrix &Y,
                                   const Matrix &Ypred, const Matrix &W2,
                                   const Matrix &Hidden) {
  auto W2Gradient =
      mult(1.0 / X.size(), mult(transpose(Hidden), sub(Ypred, Y)));
  auto Ones = createMatrix(Hidden, 1.0);
  auto W1Gradient =
      mult(1.0 / X.size(),
           mult(transpose(X),
                multElements(mult(sub(Ypred, Y), transpose(W2)),
                             multElements(Hidden, sub(Ones, Hidden)))));
  return {W1Gradient, W2Gradient};
}

std::pair<Matrix, Matrix> train(const Matrix &X, const Matrix &Y,
                                int NumHiddenNodes, int Iterations, double LR) {
  auto W1 = createMatrix(X[0].size(), NumHiddenNodes);
  fillRandom(W1);
  auto W2 = createMatrix(NumHiddenNodes, Y[0].size());
  fillRandom(W2);
  for (size_t It = 0; It < Iterations; ++It) {
    auto [Ypred, Hidden] = forward(X, W1, W2);
    auto [W1Gradient, W2Gradient] = backward(X, Y, Ypred, W2, Hidden);
    double CurrentLoss = loss(Y, Ypred);
    std::cout << "Iteration " << It << " Loss: " << CurrentLoss << "\n";
    W1 = sub(W1, mult(LR, W1Gradient));
    W2 = sub(W2, mult(LR, W2Gradient));
  }
  return {W1, W2};
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

  auto [W1, W2] = nn::train(TrainImages, EncodedLabels, 128, 100, 0.01);

  auto TestImages = nn::readImageFile("t10k-images.idx3-ubyte");
  auto TestLabels = nn::readLabelFile("t10k-labels.idx1-ubyte");
  nn::accuracy(TestLabels, nn::classify(TestImages, W1, W2));

  return 0;
}
