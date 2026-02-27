#include <GAP.hpp>

GAP::GAP(int inputWidth, int inputHeight, ActivationFunction activation) : Layer(activation) {
  this->inputWidth = inputWidth;
  this->inputHeight = inputHeight;
}

Tensor3<float> GAP::forward(Tensor3<float> input) {
  Tensor3<float> output = Tensor3<float>(1, 1, input.getChannels());
  for (size_t c = 0; c < input.getChannels(); c++) {
    float sum = 0;
    for (size_t y = 0; y < input.getHeight(); y++) {
      for (size_t x = 0; x < input.getWidth(); x++) {
        sum += input.getValue(x, y, c);
      }
    }
    output.setValue(0, 0, c, sum / (input.getWidth() * input.getHeight()));
  }
  return output;
}

Tensor3<float> GAP::backwards(Tensor3<float> prevLayerDeltas) {
  Tensor3<float> output =
      Tensor3<float>(this->inputWidth, this->inputHeight, prevLayerDeltas.getChannels());
  for (size_t c = 0; c < prevLayerDeltas.getChannels(); c++) {
    float delta = prevLayerDeltas.getValue(0, 0, c) / (output.getWidth() * output.getHeight());
    for (size_t y = 0; y < output.getHeight(); y++) {
      for (size_t x = 0; x < output.getWidth(); x++) {
        output.setValue(x, y, c, delta);
      }
    }
  }
  return output;
}