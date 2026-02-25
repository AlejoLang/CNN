#include <Activations.hpp>
#include <Algebra.hpp>
#include <DenseLayer.hpp>

DenseLayer::DenseLayer(int inputSize, int outputSize, ActivationFunction activation)
    : Layer(activation) {
  this->inputSize = inputSize;
  this->outputSize = outputSize;
  this->activation = activation;
  this->weights = Matrix<float>(inputSize, outputSize);
  this->biases = Matrix<float>(1, outputSize);
}

Tensor3<float> DenseLayer::forward(Tensor3<float> input) {
  Matrix<float> inputMat = Matrix<float>(1, this->inputSize);
  for (size_t i = 0; i < input.getWidth(); i++) {
    inputMat.setValue(0, i, input.getValue(i, 0, 0));
  }

  Matrix<float> outputMatrix = cross(this->weights, inputMat);
  for (size_t x = 0; x < outputMatrix.getNumCols(); x++) {
    for (size_t y = 0; y < outputMatrix.getNumRows(); y++) {
      float val = outputMatrix.getValue(x, y) + this->biases.getValue(0, y);
      if (this->activation == RELU) {
        val = relu(val);
      } else if (this->activation == SIGMOID) {
        val = sigmoid(val);
      }
      outputMatrix.setValue(x, y, val);
    }
  }

  Tensor3<float> outputTensor = Tensor3<float>(1, outputMatrix.getNumRows(), 1);
  for (size_t i = 0; i < outputMatrix.getNumRows(); i++) {
    outputTensor.setValue(0, i, 0, outputMatrix.getValue(0, i));
  }
  return outputTensor;
}

Tensor3<float> DenseLayer::backwards(Tensor3<float> deltas) {}