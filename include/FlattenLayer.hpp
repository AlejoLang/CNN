#pragma once
#include <Layer.hpp>
class FlattenLayer : public Layer {
private:
  int inputWidth;
  int inputHeight;
  int inputDepth;

public:
  FlattenLayer(int inputWidth, int inputHeight, int inputDepth);
  Tensor3<float> forward(Tensor3<float> input) override;
};