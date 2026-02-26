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
  Matrix<float> lastInput;
  Matrix<float> activations;
  Matrix<float> deltas;

public:
  DenseLayer(int inputSize, int outputSize, ActivationFunction activation = RELU);
  Tensor3<float> forward(Tensor3<float> input) override;
  Tensor3<float> backwards(Tensor3<float> prevLayerDeltas) override;
  void update(float learningRate) override;
  void initWeights() override;
  void setWeights(Matrix<float> weights);
  void setBiases(Matrix<float> biases);
  Matrix<float> getWeights();
  Matrix<float> getBiases();
  int getInputSize();
  int getOutputSize();
  ActivationFunction getActivation();
};