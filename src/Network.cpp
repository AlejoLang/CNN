#include <Activations.hpp>
#include <ConvolutionalLayer.hpp>
#include <DenseLayer.hpp>
#include <FlattenLayer.hpp>
#include <GAP.hpp>
#include <MaxPoolLayer.hpp>
#include <Network.hpp>
#include <fstream>
#include <iostream>

void Network::addLayer(Layer* layer) {
  layer->initWeights();
  this->layers.push_back(layer);
}

Tensor3<float> Network::forward(Tensor3<float> input) {
  Tensor3<float> output = input;
  for (size_t i = 0; i < this->layers.size(); i++) {
    output = this->layers[i]->forward(output);
  }
  // Numerically stable softmax: subtract max before exponentiating
  float maxVal = output.getValue(0, 0, 0);
  for (size_t i = 1; i < output.getWidth(); i++) {
    float val = output.getValue(i, 0, 0);
    if (val > maxVal)
      maxVal = val;
  }
  float sum = 0;
  for (size_t i = 0; i < output.getWidth(); i++) {
    float val = expf(output.getValue(i, 0, 0) - maxVal);
    output.setValue(i, 0, 0, val);
    sum += val;
  }
  for (size_t i = 0; i < output.getWidth(); i++) {
    output.setValue(i, 0, 0, output.getValue(i, 0, 0) / sum);
  }
  return output;
}

void Network::backwards(Tensor3<float> result, Tensor3<float> expected) {
  // Gradient of MSE loss w.r.t. softmax output: dL/ds_i = 2*(s_i - y_i)
  // Gradient through softmax jacobian: dL/dz_i = s_i * (dL/ds_i - sum_j(dL/ds_j * s_j))
  int n = result.getWidth();
  float dot = 0.0f;
  for (int i = 0; i < n; i++) {
    float si = result.getValue(i, 0, 0);
    float yi = const_cast<Tensor3<float>&>(expected).getValue(i, 0, 0);
    dot += 2.0f * (si - yi) * si;
  }
  Tensor3<float> output = Tensor3<float>(n, 1, 1);
  for (int i = 0; i < n; i++) {
    float si = result.getValue(i, 0, 0);
    float yi = const_cast<Tensor3<float>&>(expected).getValue(i, 0, 0);
    float grad = si * (2.0f * (si - yi) - dot);
    output.setValue(i, 0, 0, grad);
  }
  for (int i = (int)this->layers.size() - 1; i >= 0; i--) {
    output = this->layers[i]->backwards(output);
  }
}

void Network::update(float learningRate) {
  for (size_t i = 0; i < this->layers.size(); i++) {
    this->layers[i]->update(learningRate);
  }
}

void Network::saveWeights(std::string path) {
  auto file = std::ofstream(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for saving weights: " << path << std::endl;
    return;
  }
  for (auto* layer : this->layers) {
    if (ConvolutionalLayer* convLayer = dynamic_cast<ConvolutionalLayer*>(layer)) {
      Matrix<float> filters = convLayer->getFilters();
      Matrix<float> biases = convLayer->getBiases();
      LayerType type = CONVOLUTIONAL;
      file.write(reinterpret_cast<char*>(&type), sizeof(LayerType));
      int filterCount = convLayer->getFilterCount();
      int filterSize = convLayer->getFilterSize();
      int filterDepth = convLayer->getFilterDepth();
      file.write(reinterpret_cast<char*>(&filterCount), sizeof(int));
      file.write(reinterpret_cast<char*>(&filterSize), sizeof(int));
      file.write(reinterpret_cast<char*>(&filterDepth), sizeof(int));
      file.write(reinterpret_cast<char*>(filters.getValues()),
                 sizeof(float) * filters.getNumRows() * filters.getNumCols());
      file.write(reinterpret_cast<char*>(biases.getValues()),
                 sizeof(float) * biases.getNumRows() * biases.getNumCols());

    } else if (DenseLayer* denseLayer = dynamic_cast<DenseLayer*>(layer)) {
      Matrix<float> weights = denseLayer->getWeights();
      Matrix<float> biases = denseLayer->getBiases();
      LayerType type = DENSE;
      file.write(reinterpret_cast<char*>(&type), sizeof(LayerType));
      int inputSize = denseLayer->getInputSize();
      int outputSize = denseLayer->getOutputSize();
      file.write(reinterpret_cast<char*>(&inputSize), sizeof(int));
      file.write(reinterpret_cast<char*>(&outputSize), sizeof(int));
      file.write(reinterpret_cast<char*>(weights.getValues()),
                 sizeof(float) * weights.getNumRows() * weights.getNumCols());
      file.write(reinterpret_cast<char*>(biases.getValues()),
                 sizeof(float) * biases.getNumRows() * biases.getNumCols());
    } else if (MaxPoolLayer* poolLayer = dynamic_cast<MaxPoolLayer*>(layer)) {
      LayerType type = MAXPOOL;
      file.write(reinterpret_cast<char*>(&type), sizeof(LayerType));
      int poolSize = poolLayer->getPoolSize();
      int poolDepth = poolLayer->getPoolDepth();
      file.write(reinterpret_cast<char*>(&poolSize), sizeof(int));
      file.write(reinterpret_cast<char*>(&poolDepth), sizeof(int));
    } else if (FlattenLayer* flattenLayer = dynamic_cast<FlattenLayer*>(layer)) {
      LayerType type = FLATTEN;
      file.write(reinterpret_cast<char*>(&type), sizeof(LayerType));
      int inputWidth = flattenLayer->getInputWidth();
      int inputHeight = flattenLayer->getInputHeight();
      int inputDepth = flattenLayer->getInputDepth();
      file.write(reinterpret_cast<char*>(&inputWidth), sizeof(int));
      file.write(reinterpret_cast<char*>(&inputHeight), sizeof(int));
      file.write(reinterpret_cast<char*>(&inputDepth), sizeof(int));
    } else if (GAP* gapLayer = dynamic_cast<GAP*>(layer)) {
      LayerType type = GAP_LAYER;
      file.write(reinterpret_cast<char*>(&type), sizeof(LayerType));
      int inputWidth = gapLayer->getInputWidth();
      int inputHeight = gapLayer->getInputHeight();
      file.write(reinterpret_cast<char*>(&inputWidth), sizeof(int));
      file.write(reinterpret_cast<char*>(&inputHeight), sizeof(int));
    } else {
      std::cerr << "Unknown layer type during saveWeights, skipping layer" << std::endl;
    }
  }
  file.close();
}

void Network::loadWeights(std::string path) {
  auto file = std::ifstream(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for loading weights: " << path << std::endl;
    return;
  }
  this->layers.clear();
  while (file.peek() != EOF) {
    LayerType type;
    file.read(reinterpret_cast<char*>(&type), sizeof(LayerType));
    if (type == CONVOLUTIONAL) {
      int filterCount, filterSize, filterDepth;
      file.read(reinterpret_cast<char*>(&filterCount), sizeof(int));
      file.read(reinterpret_cast<char*>(&filterSize), sizeof(int));
      file.read(reinterpret_cast<char*>(&filterDepth), sizeof(int));
      Matrix<float> filters(filterCount, filterSize * filterSize * filterDepth);
      Matrix<float> biases(1, filterCount);
      file.read(reinterpret_cast<char*>(filters.getValues()),
                sizeof(float) * filters.getNumRows() * filters.getNumCols());
      file.read(reinterpret_cast<char*>(biases.getValues()),
                sizeof(float) * biases.getNumRows() * biases.getNumCols());
      ConvolutionalLayer* convLayer = new ConvolutionalLayer(filterSize, filterDepth, filterCount);
      convLayer->setFilters(filters);
      convLayer->setBiases(biases);
      this->layers.push_back(convLayer);
    } else if (type == DENSE) {
      int inputSize, outputSize;
      file.read(reinterpret_cast<char*>(&inputSize), sizeof(int));
      file.read(reinterpret_cast<char*>(&outputSize), sizeof(int));
      Matrix<float> weights(inputSize, outputSize);
      Matrix<float> biases(1, outputSize);
      file.read(reinterpret_cast<char*>(weights.getValues()),
                sizeof(float) * weights.getNumRows() * weights.getNumCols());
      file.read(reinterpret_cast<char*>(biases.getValues()),
                sizeof(float) * biases.getNumRows() * biases.getNumCols());
      DenseLayer* denseLayer = new DenseLayer(inputSize, outputSize);
      denseLayer->setWeights(weights);
      denseLayer->setBiases(biases);
      this->layers.push_back(denseLayer);
    } else if (type == MAXPOOL) {
      int poolSize, poolDepth;
      file.read(reinterpret_cast<char*>(&poolSize), sizeof(int));
      file.read(reinterpret_cast<char*>(&poolDepth), sizeof(int));
      MaxPoolLayer* poolLayer = new MaxPoolLayer(poolSize, poolDepth);
      this->layers.push_back(poolLayer);
    } else if (type == FLATTEN) {
      int inputWidth, inputHeight, inputDepth;
      file.read(reinterpret_cast<char*>(&inputWidth), sizeof(int));
      file.read(reinterpret_cast<char*>(&inputHeight), sizeof(int));
      file.read(reinterpret_cast<char*>(&inputDepth), sizeof(int));
      FlattenLayer* flattenLayer = new FlattenLayer(inputWidth, inputHeight, inputDepth);
      this->layers.push_back(flattenLayer);
    } else if (type == GAP_LAYER) {
      int inputWidth, inputHeight;
      file.read(reinterpret_cast<char*>(&inputWidth), sizeof(int));
      file.read(reinterpret_cast<char*>(&inputHeight), sizeof(int));
      this->layers.push_back(new GAP(inputWidth, inputHeight));
    } else {
      std::cerr << "Unknown layer type during loadWeights, skipping layer" << std::endl;
    }
  }
  file.close();
}

Network::~Network() {
  for (size_t i = 0; i < this->layers.size(); i++) {
    delete this->layers[i];
  }
}
