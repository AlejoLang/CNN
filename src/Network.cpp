#include <Network.hpp>
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
  float sum = 0;
  for (size_t i = 0; i < output.getWidth(); i++) {
    float val = output.getValue(i, 0, 0);
    val = exp(val);
    output.setValue(i, 0, 0, val);
    sum += val;
  }
  for (size_t i = 0; i < output.getWidth(); i++) {
    float val = output.getValue(i, 0, 0);
    val /= sum;
    output.setValue(i, 0, 0, val);
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

Network::~Network() {
  for (size_t i = 0; i < this->layers.size(); i++) {
    delete this->layers[i];
  }
}
