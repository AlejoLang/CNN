#include <FlattenLayer.hpp>

FlattenLayer::FlattenLayer(int inputWidth, int inputHeight, int inputDepth) {
  this->inputWidth = inputWidth;
  this->inputHeight = inputHeight;
  this->inputDepth = inputDepth;
}

Tensor3<float> FlattenLayer::forward(Tensor3<float> input) {
  Tensor3<float> output =
      Tensor3<float>(input.getWidth() * input.getHeight() * input.getChannels(), 1, 1);
  for (size_t c = 0; c < input.getChannels(); c++) {
    int channelBase = c * input.getWidth() * input.getHeight();
    for (size_t y = 0; y < input.getHeight(); y++) {
      int rowBase = channelBase + (y * input.getWidth());
      for (size_t x = 0; x < input.getWidth(); x++) {
        float val = input.getValue(x, y, c);
        output.setValue(rowBase + x, 0, 0, val);
      }
    }
  }
  return output;
}

Tensor3<float> FlattenLayer::backwards(Tensor3<float> prevLayerDeltas) {
  Tensor3<float> output = Tensor3<float>(this->inputWidth, this->inputHeight, this->inputDepth);
  for (size_t c = 0; c < output.getChannels(); c++) {
    int channelBase = c * output.getWidth() * output.getHeight();
    for (size_t y = 0; y < output.getHeight(); y++) {
      int rowBase = channelBase + (y * output.getWidth());
      for (size_t x = 0; x < output.getWidth(); x++) {
        float val = prevLayerDeltas.getValue(rowBase + x, 0, 0);
        output.setValue(x, y, c, val);
      }
    }
  }
  return output;
}