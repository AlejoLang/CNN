#pragma once
#include <Tensor3.hpp>

class Layer {
public:
  Layer();
  virtual Tensor3<float> forward(Tensor3<float> input);
  virtual Tensor3<float> backwards(Tensor3<float> deltas);
};