#pragma once
#include <Layer.hpp>
#include <Matrix.hpp>
class DenseLayer : public Layer {
private:
  int inputSize;
  int outputSize;
  Matrix<float> weights;
  Matrix<float> biases;
  ActivationFunction activation;

public:
  DenseLayer(int inputSize, int outputSize, ActivationFunction activation = RELU);
  Tensor3<float> forward(Tensor3<float> input) override;
  Tensor3<float> backwards(Tensor3<float> deltas) override;
};