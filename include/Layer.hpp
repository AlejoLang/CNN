#pragma once
#include <Activations.hpp>
#include <Tensor3.hpp>

class Layer {
protected:
  ActivationFunction activation;

public:
  Layer(ActivationFunction activation = ActivationFunction::NONE) : activation(activation) {};
  virtual Tensor3<float> forward(Tensor3<float> input) = 0;
  virtual Tensor3<float> backwards(Tensor3<float> prevLayerDeltas) = 0;
  virtual void update(float learningRate) {};
  virtual void initWeights() {};
  virtual ~Layer() = default;
};