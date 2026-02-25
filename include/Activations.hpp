#pragma once
#include <cmath>

enum ActivationFunction { RELU, SIGMOID, NONE };

float relu(float x) {
  return x > 0 ? x : 0;
}
float reluDerivative(float x) {
  return x > 0 ? 1.0f : 0.0f;
}
float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}
float sigmoidDerivative(float x) {
  float s = sigmoid(x);
  return s * (1.0f - s);
}