#include <Algebra.hpp>
#include <Matrix.hpp>
#include <Tensor3.hpp>
#include <stdexcept>

template <typename T>
Matrix<T> cross(Matrix<T> m1, Matrix<T> m2) {
  if (m1.getNumCols() != m2.getNumRows()) {
    throw std::invalid_argument(
        "Number of colums of the first matrix doesn't match the number of rows of the second");
  }
  Matrix<T> result = Matrix<T>(m2.getNumCols(), m1.getNumRows());
  for (size_t y = 0; y < m1.getNumRows(); y++) {
    for (size_t x = 0; x < m1.getNumCols(); x++) {
      for (size_t x2 = 0; x2 < m2.getNumCols(); x2++) {
        T v = result.getValue(x2, y);
        v += m1.getValue(x, y) * m2.getValue(x2, x);
        result.setValue(x2, y, v);
      }
    }
  }
  return result;
}

template <typename T>
Matrix<T> transpose(Matrix<T> m) {
  Matrix<T> result = Matrix<T>(m.getNumRows(), m.getNumCols());
  for (size_t y = 0; y < m.getNumRows(); y++) {
    for (size_t x = 0; x < m.getNumCols(); x++) {
      result.setValue(y, x, m.getValue(x, y));
    }
  }
  return result;
}

template <typename T>
Matrix<T> apply(Matrix<T> m, T (*function)(T)) {
  Matrix<T> result = Matrix<T>(m.getNumCols(), m.getNumRows());
  for (size_t y = 0; y < m.getNumRows(); y++) {
    for (size_t x = 0; x < m.getNumCols(); x++) {
      result.setValue(x, y, function(m.getValue(x, y)));
    }
  }
  return result;
}

template <typename T>
Matrix<T> im2col(Tensor3<T> input, int filterSize, int filterDepth) {
  int slidesW = input.getWidth() - filterSize + 1;
  int slidesH = input.getHeight() - filterSize + 1;
  Matrix<T> flatInput(filterSize * filterSize * filterDepth, slidesH * slidesW);
  for (size_t c = 0; c < filterDepth; c++) {
    int rowBase = c * filterSize * filterSize;
    for (size_t y = 0; y < slidesH; y++) {
      for (size_t x = 0; x < slidesW; x++) {
        int col = y * slidesW + x;
        for (size_t filY = 0; filY < filterSize; filY++) {
          int rowBaseY = rowBase + filY * filterSize;
          for (size_t filX = 0; filX < filterSize; filX++) {
            T val = input.getValue(x + filX, y + filY, c);
            flatInput.setValue(rowBaseY + filX, col, val);
          }
        }
      }
    }
  }

  return flatInput;
}

// Explicit template instantiations
template Matrix<int> cross(Matrix<int> m1, Matrix<int> m2);
template Matrix<float> cross(Matrix<float> m1, Matrix<float> m2);
template Matrix<double> cross(Matrix<double> m1, Matrix<double> m2);
template Matrix<int> transpose(Matrix<int> m);
template Matrix<float> transpose(Matrix<float> m);
template Matrix<double> transpose(Matrix<double> m);
template Matrix<int> apply(Matrix<int> m, int (*function)(int));
template Matrix<float> apply(Matrix<float> m, float (*function)(float));
template Matrix<double> apply(Matrix<double> m, double (*function)(double));
template Matrix<int> im2col(Tensor3<int> input, int filterSize, int filterDepth);
template Matrix<float> im2col(Tensor3<float> input, int filterSize, int filterDepth);
template Matrix<double> im2col(Tensor3<double> input, int filterSize, int filterDepth);