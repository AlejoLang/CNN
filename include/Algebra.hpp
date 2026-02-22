#pragma once
#include <Matrix.hpp>

// Computes the cross product of two matrices
template <typename T>
Matrix<T> cross(Matrix<T> m1, Matrix<T> m2);

// Computes the transpose of a matrix
template <typename T>
Matrix<T> transpose(Matrix<T> m);

// Apply a function to every element of a matrix
template <typename T>
Matrix<T> apply(Matrix<T> m, T (*function)(T));
