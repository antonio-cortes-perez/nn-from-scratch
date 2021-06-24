// Routines to load MNIST data (http://yann.lecun.com/exdb/mnist/)

#include "Matrix.h"

#include <fstream>
#include <iostream>
#include <string>

namespace nn {

uint32_t readUint32(std::ifstream &Input) {
  uint32_t N = 0;
  for (size_t Idx = 0; Idx < 4; ++Idx) {
    N <<= 8;
    N |= Input.get();
  }
  return N;
}

Matrix readImageFile(const std::string &Filename) {
  std::ifstream Input(Filename.c_str(), std::ios::in | std::ios::binary);
  uint32_t MagicNumber = readUint32(Input);
  uint32_t NumRows = readUint32(Input);
  uint32_t NumCols = readUint32(Input) * readUint32(Input) + 1; // + 1 => bias
  auto M = createMatrix(NumRows, NumCols);
  for (size_t Row = 0; Row < NumRows; ++Row) {
    M[Row][0] = 1;
    for (size_t Col = 1; Col < NumCols; ++Col) {
      M[Row][Col] = Input.get();
    }
  }
  Input.close();
  return M;
}

Matrix readLabelFile(const std::string &Filename) {
  std::ifstream Input(Filename.c_str(), std::ios::in | std::ios::binary);
  uint32_t MagicNumber = readUint32(Input);
  uint32_t NumRows = readUint32(Input);
  auto M = createMatrix(NumRows, 1);
  for (size_t Row = 0; Row < NumRows; ++Row) {
    M[Row][0] = Input.get();
  }
  Input.close();
  return M;
}

void printImageAndLabel(const Matrix &Images, const Matrix &Labels, int Idx) {
  std::cout << "Label: " << Labels[Idx][0] << "\n";

  for (size_t I = 0; I < 28; ++I) {
    for (size_t J = 0; J < 28; ++J) {
      std::cout << ((Images[Idx][I * 28 + J + 1] > 127) ? "*" : " ");
    }
    std::cout << "\n";
  }
}

} // namespace nn
