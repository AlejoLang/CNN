#pragma once
#include <Layer.hpp>

class GAP : public Layer {
private:
  int inputWidth;
  int inputHeight;

public:
  GAP(int inputWidth, int inputHeight, ActivationFunction activation = ActivationFunction::NONE);
  Tensor3<float> forward(Tensor3<float> input) override;
  Tensor3<float> backwards(Tensor3<float> prevLayerDeltas) override;
  int getInputWidth() const {
    return inputWidth;
  }
  int getInputHeight() const {
    return inputHeight;
  }
};