#include <Activations.hpp>
#include <Algebra.hpp>
#include <ConvolutionalLayer.hpp>
#include <Matrix.hpp>
#include <cmath>

ConvolutionalLayer::ConvolutionalLayer(int filterSize, int filterDepth, int filterCount,
                                       ActivationFunction activation)
    : Layer(activation) {
  this->activation = activation;
  this->filterSize = filterSize;
  this->filterDepth = filterDepth;
  this->filterCount = filterCount;
  for (size_t i = 0; i < filterCount; i++) {
    Tensor3<float> newT = Tensor3<float>(filterSize, filterSize, filterDepth);
    filters.push_back(newT);
  }
  this->biases = Tensor3<float>(1, 1, filterCount);
}

Tensor3<float> ConvolutionalLayer::forward(Tensor3<float> input) {
  Matrix<float> flatInput = im2col(input, this->filterSize, this->filterDepth);
  Matrix<float> flatFilters =
      Matrix<float>(this->filterCount, this->filterSize * this->filterSize * this->filterDepth);
  for (size_t f = 0; f < filterCount; f++) {
    for (size_t c = 0; c < filterDepth; c++) {
      int channel = c * this->filterSize * this->filterSize;
      for (size_t y = 0; y < filterSize; y++) {
        int row = y * this->filterSize;
        for (size_t x = 0; x < filterSize; x++) {
          float val = this->filters[f].getValue(x, y, c);
          flatFilters.setValue(f, channel + row + x, val);
        }
      }
    }
  }

  Matrix<float> featureMat = cross(flatInput, flatFilters);
  int slidesW = input.getWidth() - this->filterSize + 1;
  int slidesH = input.getHeight() - this->filterSize + 1;
  Tensor3<float> featureTens = Tensor3<float>(slidesW, slidesH, this->filterCount);
  for (size_t y = 0; y < featureMat.getNumRows(); y++) {
    int row = std::floor(y / slidesW);
    int col = y % slidesW;
    for (size_t x = 0; x < featureMat.getNumCols(); x++) {
      float value = featureMat.getValue(x, y) + this->biases.getValue(col, row, x);
      if (this->activation == RELU) {
        value = relu(value);
      } else if (this->activation == SIGMOID) {
        value = sigmoid(value);
      }
      featureTens.setValue(col, row, x, value);
    }
  }
  return featureTens;
}