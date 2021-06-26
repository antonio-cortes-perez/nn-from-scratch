#ifndef NN_MATRIX_H
#define NN_MATRIX_H

#include <fstream>
#include <vector>

namespace nn {

using Matrix = std::vector<std::vector<double>>;

Matrix createMatrix(int R, int C, double V = 0.0);

Matrix createMatrix(const Matrix &M, double V = 0.0);

Matrix mult(const Matrix &A, const Matrix &B);

Matrix mult(double V, const Matrix &A);

Matrix add(const Matrix &A, const Matrix &B);

Matrix sub(const Matrix &A, const Matrix &B);

Matrix multElements(const Matrix &A, const Matrix &B);

Matrix squared(const Matrix &A);

double average(const Matrix &A);

double sum(const Matrix &A);

Matrix transpose(const Matrix &A);

Matrix sigmoid(const Matrix &A);

Matrix softmax(const Matrix &A);

Matrix round(const Matrix &A);

Matrix log(const Matrix &A);

void fillRandom(Matrix &A);

void print(const Matrix &A, std::string name, int firstNRows, int firstNCols);

} // namespace nn

std::ostream &operator<<(std::ostream &O, const nn::Matrix &M);

#endif // NN_MATRIX_H
