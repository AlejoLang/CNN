#pragma once
#include <Layer.hpp>
#include <Tensor3.hpp>
#include <vector>

class Network {
private:
  std::vector<Layer*> layers;

public:
  Network() = default;
  void addLayer(Layer* layer);
  Tensor3<float> forward(Tensor3<float> input);
  void backwards(Tensor3<float> result, Tensor3<float> expected);
  void update(float learningRate);
  ~Network();
};