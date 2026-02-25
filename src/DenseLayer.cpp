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
  this->activations = Matrix<float>(1, outputSize);
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
      this->activations.setValue(x, y, val);
      if (this->activation == RELU) {
        val = relu(val);
      } else if (this->activation == SIGMOID) {
        val = sigmoid(val);
      }
      outputMatrix.setValue(x, y, val);
    }
  }

  Tensor3<float> outputTensor = Tensor3<float>(outputMatrix.getNumRows(), 1, 1);
  for (size_t i = 0; i < outputMatrix.getNumRows(); i++) {
    outputTensor.setValue(i, 0, 0, outputMatrix.getValue(0, i));
  }
  this->lastInput = inputMat;
  return outputTensor;
}

Tensor3<float> DenseLayer::backwards(Tensor3<float> prevLayerDeltas) {
  Matrix<float> prevLayerDeltasMat = Matrix<float>(1, prevLayerDeltas.getWidth());
  for (size_t i = 0; i < prevLayerDeltas.getWidth(); i++) {
    prevLayerDeltasMat.setValue(0, i, prevLayerDeltas.getValue(i, 0, 0));
  }

  if (this->activation == NONE) {
    this->deltas = prevLayerDeltasMat;
  } else {
    this->deltas = hadamard(
        prevLayerDeltasMat,
        apply(this->activations, this->activation == RELU ? reluDerivative : sigmoidDerivative));
  }
  Matrix<float> prevDeltas = cross(transpose(this->weights), deltas);
  Tensor3<float> result = Tensor3<float>(this->inputSize, 1, 1);
  for (size_t i = 0; i < (size_t)this->inputSize; i++) {
    result.setValue(i, 0, 0, prevDeltas.getValue(0, i));
  }
  return result;
}

void DenseLayer::update(float learningRate) {
  Matrix<float> weightDeltas = cross(this->deltas, transpose(this->lastInput));
  this->weights = this->weights - (weightDeltas * learningRate);
  for (size_t i = 0; i < this->biases.getNumRows(); i++) {
    float biasDelta = this->deltas.getValue(0, i) * learningRate;
    this->biases.setValue(0, i, this->biases.getValue(0, i) - biasDelta);
  }
}