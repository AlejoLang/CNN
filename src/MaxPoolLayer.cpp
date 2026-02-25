#include <MaxPoolLayer.hpp>
#include <cmath>

MaxPoolLayer::MaxPoolLayer(int size, int depth) {
  this->poolSize = size;
  this->poolDepth = depth;
}

Tensor3<float> MaxPoolLayer::forward(Tensor3<float> input) {
  int slidesW = input.getWidth() / this->poolSize;
  int slidesH = input.getHeight() / this->poolSize;
  Tensor3<float> output = Tensor3<float>(slidesW, slidesH, this->poolDepth);
  for (size_t c = 0; c < this->poolDepth; c++) {
    for (size_t y = 0; y < slidesH; y++) {
      for (size_t x = 0; x < slidesW; x++) {
        float maxVal = -MAXFLOAT;
        for (size_t poolY = 0; poolY < this->poolSize; poolY++) {
          int inY = y * this->poolSize + poolY;
          for (size_t poolX = 0; poolX < this->poolSize; poolX++) {
            int inX = x * this->poolSize + poolX;
            float val = input.getValue(inX, inY, c);
            if (val > maxVal) {
              maxVal = val;
            }
          }
        }
        output.setValue(x, y, c, maxVal);
      }
    }
  }
  return output;
}

Tensor3<float> MaxPoolLayer::backwards(Tensor3<float> deltas) {}