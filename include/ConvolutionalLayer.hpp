#pragma once
#include <Activations.hpp>
#include <Layer.hpp>
#include <Matrix.hpp>
#include <vector>

class ConvolutionalLayer : public Layer {
private:
  int filterCount;
  int filterSize;
  int filterDepth;
  std::vector<Tensor3<float>> filters;
  Tensor3<float> biases;
  Matrix<float> flatLastInput;
  Matrix<float> flatActivations;
  Matrix<float> deltas;

public:
  ConvolutionalLayer(int filterSize, int filterDepth, int filterCount,
                     ActivationFunction activation = RELU);
  Tensor3<float> forward(Tensor3<float> input) override;
  Tensor3<float> backwards(Tensor3<float> prevLayerDeltas) override;
  void update(float learningRate) override;
};