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
  this->maxIndexes.clear();
  for (size_t c = 0; c < this->poolDepth; c++) {
    this->maxIndexes.push_back(std::vector<int>());
    for (size_t y = 0; y < slidesH; y++) {
      for (size_t x = 0; x < slidesW; x++) {
        float maxVal = -MAXFLOAT;
        int maxDisplacement = 0;
        for (size_t poolY = 0; poolY < this->poolSize; poolY++) {
          int inY = y * this->poolSize + poolY;
          for (size_t poolX = 0; poolX < this->poolSize; poolX++) {
            int inX = x * this->poolSize + poolX;
            float val = input.getValue(inX, inY, c);
            if (val > maxVal) {
              maxVal = val;
              maxDisplacement = poolY * this->poolSize + poolX;
            }
          }
        }
        output.setValue(x, y, c, maxVal);
        this->maxIndexes[c].push_back(maxDisplacement);
      }
    }
  }
  return output;
}

Tensor3<float> MaxPoolLayer::backwards(Tensor3<float> deltas) {
  Tensor3<float> output = Tensor3<float>(deltas.getWidth() * this->poolSize,
                                         deltas.getHeight() * this->poolSize, this->poolDepth);
  for (size_t c = 0; c < this->poolDepth; c++) {
    for (size_t y = 0; y < deltas.getHeight(); y++) {
      for (size_t x = 0; x < deltas.getWidth(); x++) {
        float delta = deltas.getValue(x, y, c);
        int maxIndex = this->maxIndexes[c][y * deltas.getWidth() + x];
        int poolY = maxIndex / this->poolSize;
        int poolX = maxIndex % this->poolSize;
        int outY = y * this->poolSize + poolY;
        int outX = x * this->poolSize + poolX;
        output.setValue(outX, outY, c, delta);
      }
    }
  }
  return output;
}