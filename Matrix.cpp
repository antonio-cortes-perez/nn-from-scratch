#include "Matrix.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>

namespace nn {

using Matrix = std::vector<std::vector<double>>;

Matrix createMatrix(int R, int C, double V) {
  return Matrix(R, std::vector<double>(C, V));
}

Matrix createMatrix(const Matrix &M, double V) {
  return createMatrix(M.size(), M[0].size(), V);
}

Matrix mult(const Matrix &A, const Matrix &B) {
  assert(A[0].size() == B.size());
  auto C = createMatrix(A.size(), B[0].size());
  for (size_t Row = 0; Row < A.size(); ++Row) {
    for (size_t Col = 0; Col < B[0].size(); ++Col) {
      double Sum = 0.0;
      for (size_t Idx = 0; Idx < A[0].size(); ++Idx) {
        Sum += A[Row][Idx] * B[Idx][Col];
      }
      C[Row][Col] = Sum;
    }
  }
  return C;
}

Matrix mult(double D, const Matrix &A) {
  auto B = createMatrix(A);
  for (size_t Row = 0; Row < A.size(); ++Row) {
    for (size_t Col = 0; Col < A[0].size(); ++Col) {
      B[Row][Col] = D * A[Row][Col];
    }
  }
  return B;
}

Matrix add(const Matrix &A, const Matrix &B) {
  assert(A.size() == B.size());
  assert(A[0].size() == B[0].size());
  auto C = createMatrix(A);
  for (size_t Row = 0; Row < A.size(); ++Row) {
    for (size_t Col = 0; Col < A[0].size(); ++Col) {
      C[Row][Col] = A[Row][Col] + B[Row][Col];
    }
  }
  return C;
}

Matrix sub(const Matrix &A, const Matrix &B) {
  assert(A.size() == B.size());
  assert(A[0].size() == B[0].size());
  auto C = createMatrix(A);
  for (size_t Row = 0; Row < A.size(); ++Row) {
    for (size_t Col = 0; Col < A[0].size(); ++Col) {
      C[Row][Col] = A[Row][Col] - B[Row][Col];
    }
  }
  return C;
}

Matrix multElements(const Matrix &A, const Matrix &B) {
  assert(A.size() == B.size());
  assert(A[0].size() == B[0].size());
  auto C = createMatrix(A);
  for (size_t Row = 0; Row < A.size(); ++Row) {
    for (size_t Col = 0; Col < A[0].size(); ++Col) {
      C[Row][Col] = A[Row][Col] * B[Row][Col];
    }
  }
  return C;
}

Matrix squared(const Matrix &A) {
  auto B = createMatrix(A);
  for (size_t Row = 0; Row < A.size(); ++Row) {
    for (size_t Col = 0; Col < A[0].size(); ++Col) {
      B[Row][Col] = A[Row][Col] * A[Row][Col];
    }
  }
  return B;
}

double average(const Matrix &A) {
  assert(A[0].size() == 1);
  double Total = 0.0;
  for (const auto &Row : A) {
    Total += Row[0];
  }
  return Total / A.size();
}

double sum(const Matrix &A) {
  double Total = 0.0;
  for (size_t Row = 0; Row < A.size(); ++Row) {
    for (size_t Col = 0; Col < A[0].size(); ++Col) {
      Total += A[Row][Col];
    }
  }
  return Total;
}

Matrix transpose(const Matrix &A) {
  auto T = createMatrix(A[0].size(), A.size());
  for (size_t Row = 0; Row < A.size(); ++Row) {
    for (size_t Col = 0; Col < A[0].size(); ++Col) {
      T[Col][Row] = A[Row][Col];
    }
  }
  return T;
}

Matrix sigmoid(const Matrix &A) {
  auto B = createMatrix(A);
  for (size_t Row = 0; Row < A.size(); ++Row) {
    for (size_t Col = 0; Col < A[0].size(); ++Col) {
      B[Row][Col] = 1.0 / (1.0 + exp(-A[Row][Col]));
    }
  }
  return B;
}

Matrix softmax(const Matrix &A) {
  auto B = createMatrix(A);
  for (size_t Row = 0; Row < A.size(); ++Row) {
    double Total = 0.0;
    for (size_t Col = 0; Col < A[0].size(); ++Col) {
      B[Row][Col] = exp(A[Row][Col]);
      Total += B[Row][Col];
    }
    for (size_t Col = 0; Col < A[0].size(); ++Col) {
      B[Row][Col] /= Total;
    }
  }
  return B;
}

Matrix round(const Matrix &A) {
  auto B = createMatrix(A);
  for (size_t Row = 0; Row < A.size(); ++Row) {
    for (size_t Col = 0; Col < A[0].size(); ++Col) {
      B[Row][Col] = ::round(A[Row][Col]);
    }
  }
  return B;
}

Matrix log(const Matrix &A) {
  auto B = createMatrix(A);
  for (size_t Row = 0; Row < A.size(); ++Row) {
    for (size_t Col = 0; Col < A[0].size(); ++Col) {
      B[Row][Col] = ::log(A[Row][Col]);
    }
  }
  return B;
}

void fillRandom(Matrix &A) {
  srand(time(NULL));
  for (size_t Row = 0; Row < A.size(); ++Row) {
    for (size_t Col = 0; Col < A[0].size(); ++Col) {
      A[Row][Col] = static_cast<double>(rand()) / RAND_MAX;
    }
  }
}

void print(const Matrix &A, std::string name, int firstNRows, int firstNCols) {
  std::cout << name << " (" << A.size() << "," << A[0].size() << ")\n";
  int Rows = firstNRows == -1 ? A.size() : firstNRows;
  int Cols = firstNCols == -1 ? A[0].size() : firstNCols;
  for (size_t Row = 0; Row < Rows; ++Row) {
    for (size_t Col = 0; Col < Cols; ++Col) {
      std::cout << A[Row][Col] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

} // namespace nn

std::ostream &operator<<(std::ostream &O, const nn::Matrix &M) {
  for (size_t Row = 0; Row < M.size(); ++Row) {
    for (size_t Col = 0; Col < M[0].size(); ++Col) {
      O << M[Row][Col] << " ";
    }
    O << "\n";
  }
  O << "\n";
  return O;
}
