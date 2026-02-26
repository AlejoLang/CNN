#pragma once
#include <Layer.hpp>
#include <vector>

class MaxPoolLayer : public Layer {
private:
  int poolSize;
  int poolDepth;
  int inputWidth;
  int inputHeight;
  std::vector<std::vector<int>> maxIndexes;

public:
  MaxPoolLayer(int size, int depth);
  Tensor3<float> forward(Tensor3<float> input) override;
  Tensor3<float> backwards(Tensor3<float> prevLayerDeltas) override;
};