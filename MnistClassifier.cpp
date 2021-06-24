#include "Classifier.h"
#include "Matrix.h"
#include "Mnist.h"

#include <algorithm>

namespace nn {

Matrix oneHotEncodeMnist(const Matrix &Labels) {
  auto Encoded = createMatrix(Labels.size(), 10);
  for (size_t Row = 0; Row < Labels.size(); ++Row) {
    Encoded[Row][static_cast<int>(Labels[Row][0])] = 1;
  }
  return Encoded;
}

Matrix classifyMnist(const Matrix &X, const Matrix &W) {
  auto Ypred = nn::forward(X, W);
  auto Labels = createMatrix(Ypred.size(), 1);
  for (size_t Row = 0; Row < Ypred.size(); ++Row) {
    Labels[Row][0] = std::max_element(Ypred[Row].begin(), Ypred[Row].end()) -
                     Ypred[Row].begin();
  }
  return Labels;
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
  nn::accuracy(TestLabels, nn::classifyMnist(TestImages, W));

  return 0;
}
