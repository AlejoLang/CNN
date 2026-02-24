#pragma once
#include <Layer.hpp>
#include <vector>

class ConvolutionalLayer : public Layer {
private:
  int filterCount;
  int filterSize;
  int filterDepth;
  std::vector<Tensor3<float>> filters;
  Tensor3<float> biases;

public:
  ConvolutionalLayer(int filterSize, int filterDepth, int filterCount);
  Tensor3<float> forward(Tensor3<float> input) override;
};