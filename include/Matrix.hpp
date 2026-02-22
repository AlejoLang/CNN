#pragma once
#include <iostream>
#include <vector>

template <typename T>
class Matrix {
private:
  T* values;
  int numRows;
  int numCols;

public:
  Matrix(int c, int r);
  Matrix(std::vector<T> vals, int cols);
  Matrix(const Matrix<T>& other);
  int getNumCols();
  int getNumRows();
  T* getValues();
  T getValue(int x, int y);
  void setValue(int x, int y, T value);
  Matrix<T>& operator=(const Matrix<T>& m);
  Matrix<T> operator+(const Matrix<T>& m);
  Matrix<T> operator-(const Matrix<T>& m);
  template <typename U>
  Matrix<T> operator*(const U& num);
  template <typename U>
  Matrix<T> operator/(const U& num);
  ~Matrix();

  template <typename U>
  friend std::ostream& operator<<(std::ostream& os, const Matrix<U>& m);
};
