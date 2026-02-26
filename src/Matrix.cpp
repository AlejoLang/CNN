#include <Matrix.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

template <typename T>
Matrix<T>::Matrix() {
  this->numRows = 0;
  this->numCols = 0;
  this->values = nullptr;
}

template <typename T>
Matrix<T>::Matrix(int c, int r) {
  this->numRows = r;
  this->numCols = c;
  this->values = new T[r * c]();
}

template <typename T>
Matrix<T>::Matrix(std::vector<T> vals, int cols) {
  if (vals.size() % cols != 0) {
    throw std::invalid_argument(
        "Cant make a matrix with the specified vector size and column size");
  }
  this->numCols = cols;
  this->numRows = vals.size() / cols;
  this->values = new T[this->numCols * this->numRows];
  for (size_t y = 0; y < this->numRows; y++) {
    for (size_t x = 0; x < this->numCols; x++) {
      this->values[(this->numCols * y) + x] = vals[(this->numCols * y) + x];
    }
  }
};

template <typename T>
Matrix<T>::Matrix(const Matrix<T>& other) {
  this->numRows = other.numRows;
  this->numCols = other.numCols;
  this->values = new T[this->numRows * this->numCols];
  for (size_t i = 0; i < this->numRows * this->numCols; i++) {
    this->values[i] = other.values[i];
  }
}

template <typename T>
Matrix<T>::Matrix(Matrix<T>&& other) noexcept {
  this->numRows = other.numRows;
  this->numCols = other.numCols;
  this->values = other.values;
  other.values = nullptr;
  other.numRows = 0;
  other.numCols = 0;
}

template <typename T>
int Matrix<T>::getNumCols() {
  return this->numCols;
}

template <typename T>
int Matrix<T>::getNumRows() {
  return this->numRows;
}

template <typename T>
T* Matrix<T>::getValues() {
  return this->values;
}

template <typename T>
T Matrix<T>::getValue(int x, int y) {
  if (x >= this->numCols || y >= this->numRows) {
    throw std::invalid_argument("Coordinates out of bounds");
  }
  return this->values[(y * this->numCols) + x];
}

template <typename T>
void Matrix<T>::setValue(int x, int y, T value) {
  if (x >= this->numCols || y >= this->numRows) {
    throw std::invalid_argument("Coordinates out of bounds");
  }
  this->values[(y * this->numCols) + x] = value;
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& m) {
  if (this != &m) {
    delete[] this->values;
    this->numRows = m.numRows;
    this->numCols = m.numCols;
    this->values = new T[this->numRows * this->numCols];
    for (size_t i = 0; i < this->numRows * this->numCols; i++) {
      this->values[i] = m.values[i];
    }
  }
  return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& m) {
  if (this->numCols != m.numCols || this->numRows != m.numRows) {
    throw std::invalid_argument("Matrices dimensions don't match");
  }
  int n = this->numRows * this->numCols;
  Matrix<T> result = Matrix<T>(this->numCols, this->numRows);
  for (int i = 0; i < n; i++) {
    result.values[i] = this->values[i] + m.values[i];
  }
  return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& m) {
  if (this->numCols != m.numCols || this->numRows != m.numRows) {
    throw std::invalid_argument("Matrices dimensions don't match");
  }
  int n = this->numRows * this->numCols;
  Matrix<T> result = Matrix<T>(this->numCols, this->numRows);
  for (int i = 0; i < n; i++) {
    result.values[i] = this->values[i] - m.values[i];
  }
  return result;
}

template <typename T>
template <typename U>
Matrix<T> Matrix<T>::operator*(const U& num) {
  int n = this->numRows * this->numCols;
  T factor = static_cast<T>(num);
  Matrix<T> result = Matrix<T>(this->numCols, this->numRows);
  for (int i = 0; i < n; i++) {
    result.values[i] = this->values[i] * factor;
  }
  return result;
}

template <typename T>
template <typename U>
Matrix<T> Matrix<T>::operator/(const U& num) {
  if (static_cast<T>(num) == static_cast<T>(0)) {
    throw std::invalid_argument("Potential division by 0");
  }
  int n = this->numRows * this->numCols;
  T factor = static_cast<T>(num);
  Matrix<T> result = Matrix<T>(this->numCols, this->numRows);
  for (int i = 0; i < n; i++) {
    result.values[i] = this->values[i] / factor;
  }
  return result;
}

template <typename T>
Matrix<T>::~Matrix() {
  if (this->values != nullptr) {
    delete[] this->values;
  }
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& m) {
  os << "[";
  for (size_t y = 0; y < m.numRows; y++) {
    if (y > 0)
      os << " ";
    os << "[";
    for (size_t x = 0; x < m.numCols; x++) {
      os << m.values[(y * m.numCols) + x];
      if (x < m.numCols - 1)
        os << ", ";
    }
    os << "]";
    if (y < m.numRows - 1)
      os << ",\n";
  }
  os << "]";
  return os;
}

template class Matrix<float>;

template std::ostream& operator<<(std::ostream& os, const Matrix<float>& m);

template Matrix<float> Matrix<float>::operator* <int>(const int&);
template Matrix<float> Matrix<float>::operator* <float>(const float&);
template Matrix<float> Matrix<float>::operator* <double>(const double&);
template Matrix<float> Matrix<float>::operator/ <int>(const int&);
template Matrix<float> Matrix<float>::operator/ <float>(const float&);
template Matrix<float> Matrix<float>::operator/ <double>(const double&);