#ifndef NN_MNIST_H
#define NN_MNIST_H

#include "Matrix.h"

#include <string>

namespace nn {

Matrix readImageFile(const std::string &Filename);

Matrix readLabelFile(const std::string &Filename);

void printImageAndLabel(const Matrix &Images, const Matrix &Labels, int Idx);

} // namespace nn

#endif // NN_MNIST_H
