#pragma once
#include <Layer.hpp>

class MaxPoolLayer : public Layer {
private:
  int poolSize;
  int poolDepth;

public:
  MaxPoolLayer(int size, int depth);
  Tensor3<float> forward(Tensor3<float> input) override;
};