#include <Activations.hpp>
#include <Algebra.hpp>
#include <ConvolutionalLayer.hpp>
#include <Matrix.hpp>
#include <cmath>
#include <random>

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
  Matrix<float> flatInput = im2col<float>(input, this->filterSize, this->filterDepth);
  this->flatLastInput = flatInput;
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
  this->flatActivations = Matrix<float>(featureMat.getNumCols(), featureMat.getNumRows());
  int slidesW = input.getWidth() - this->filterSize + 1;
  int slidesH = input.getHeight() - this->filterSize + 1;
  Tensor3<float> featureTens = Tensor3<float>(slidesW, slidesH, this->filterCount);
  for (size_t y = 0; y < featureMat.getNumRows(); y++) {
    int row = std::floor(y / slidesW);
    int col = y % slidesW;
    for (size_t x = 0; x < featureMat.getNumCols(); x++) {
      float value = featureMat.getValue(x, y) + this->biases.getValue(0, 0, x);
      this->flatActivations.setValue(x, y, value);
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

Tensor3<float> ConvolutionalLayer::backwards(Tensor3<float> prevLayerDeltas) {
  Matrix<float> flatDeltas = im2col<float>(prevLayerDeltas, 1, this->filterCount);
  this->deltas =
      hadamard(flatDeltas, apply(this->flatActivations,
                                 this->activation == RELU ? reluDerivative : sigmoidDerivative));
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
  Matrix<float> prevDeltas = cross(this->deltas, transpose(flatFilters));
  int inputW = prevLayerDeltas.getWidth() + this->filterSize - 1;
  int inputH = prevLayerDeltas.getHeight() + this->filterSize - 1;
  Tensor3<float> result = Tensor3<float>(inputW, inputH, this->filterDepth);
  for (size_t col = 0; col < prevDeltas.getNumRows(); col++) {
    int outY = col / inputW;
    int outX = col % inputW;
    for (size_t c = 0; c < (size_t)this->filterDepth; c++) {
      int channelBase = c * this->filterSize * this->filterSize;
      for (size_t fy = 0; fy < (size_t)this->filterSize; fy++) {
        for (size_t fx = 0; fx < (size_t)this->filterSize; fx++) {
          int inY = outY - (int)fy;
          int inX = outX - (int)fx;
          if (inY >= 0 && inX >= 0 && inY < inputH && inX < inputW) {
            float v = result.getValue(inX, inY, c);
            v += prevDeltas.getValue(channelBase + fy * this->filterSize + fx, col);
            result.setValue(inX, inY, c, v);
          }
        }
      }
    }
  }
  return result;
}

void ConvolutionalLayer::update(float learningRate) {
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
  Matrix<float> weightDeltas = cross(transpose(this->flatLastInput), this->deltas);
  for (size_t f = 0; f < filterCount; f++) {
    for (size_t c = 0; c < filterDepth; c++) {
      int channel = c * this->filterSize * this->filterSize;
      for (size_t y = 0; y < filterSize; y++) {
        int row = y * this->filterSize;
        for (size_t x = 0; x < filterSize; x++) {
          float val = flatFilters.getValue(f, channel + row + x) -
                      learningRate * weightDeltas.getValue(f, channel + row + x);
          this->filters[f].setValue(x, y, c, val);
        }
      }
    }
  }
  for (size_t f = 0; f < filterCount; f++) {
    float biasVal = this->biases.getValue(0, 0, f) - learningRate * this->deltas.getValue(f, 0);
    this->biases.setValue(0, 0, f, biasVal);
  }
}

void ConvolutionalLayer::initWeights() {
  std::mt19937 rng(std::random_device{}());
  int fan_in = this->filterSize * this->filterSize * this->filterDepth;
  // He init for ReLU: stddev = sqrt(2 / fan_in)
  // Xavier init for Sigmoid/None: stddev = sqrt(1 / fan_in)
  float stddev = (this->activation == RELU) ? std::sqrt(2.0f / fan_in) : std::sqrt(1.0f / fan_in);
  std::normal_distribution<float> dist(0.0f, stddev);
  for (size_t f = 0; f < (size_t)this->filterCount; f++) {
    for (size_t c = 0; c < (size_t)this->filterDepth; c++) {
      for (size_t y = 0; y < (size_t)this->filterSize; y++) {
        for (size_t x = 0; x < (size_t)this->filterSize; x++) {
          this->filters[f].setValue(x, y, c, dist(rng));
        }
      }
    }
  }
  // biases initialized to zero
  for (size_t f = 0; f < (size_t)this->filterCount; f++) {
    this->biases.setValue(0, 0, f, 0.0f);
  }
}