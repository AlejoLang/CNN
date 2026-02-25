#include <Tensor3.hpp>
#include <stdexcept>

template <typename T>
Tensor3<T>::Tensor3() {
  this->w = 0;
  this->h = 0;
  this->c = 0;
  this->values = nullptr;
}

template <typename T>
Tensor3<T>::Tensor3(int width, int height, int channels) {
  this->w = width;
  this->h = height;
  this->c = channels;
  this->values = new T[width * height * channels]();
}

template <typename T>
Tensor3<T>::Tensor3(const Tensor3<T>& other) {
  this->w = other.w;
  this->h = other.h;
  this->c = other.c;
  int size = other.w * other.h * other.c;
  this->values = new T[size];
  for (int i = 0; i < size; i++)
    this->values[i] = other.values[i];
}

template <typename T>
Tensor3<T>& Tensor3<T>::operator=(const Tensor3<T>& other) {
  if (this == &other)
    return *this;
  delete[] this->values;
  this->w = other.w;
  this->h = other.h;
  this->c = other.c;
  int size = other.w * other.h * other.c;
  this->values = new T[size];
  for (int i = 0; i < size; i++)
    this->values[i] = other.values[i];
  return *this;
}

template <typename T>
int Tensor3<T>::getWidth() {
  return this->w;
}
template <typename T>
int Tensor3<T>::getHeight() {
  return this->h;
}
template <typename T>
int Tensor3<T>::getChannels() {
  return this->c;
}

template <typename T>
T Tensor3<T>::getValue(int x, int y, int z) {
  if (x >= this->w || y >= this->h || z >= this->c) {
    throw std::invalid_argument("Coordinates out of bounds");
  }
  return this->values[(this->w * this->h * z) + (this->w * y) + x];
}
template <typename T>
void Tensor3<T>::setValue(int x, int y, int z, T value) {
  if (x >= this->w || y >= this->h || z >= this->c) {
    throw std::invalid_argument("Coordinates out of bounds");
  }
  this->values[(this->w * this->h * z) + (this->w * y) + x] = value;
}

template <typename T>
Tensor3<T> Tensor3<T>::operator+(const Tensor3<T>& t) {
  if (this->w != t.w || this->h != t.h || this->c != t.c) {
    throw std::invalid_argument("Tensors dimensions don't match");
  }
  Tensor3<T> result = Tensor3<T>(this->w, this->h, this->c);
  for (size_t z = 0; z < this->c; z++) {
    for (size_t y = 0; y < this->h; y++) {
      for (size_t x = 0; x < this->w; x++) {
        T val = this->getValue(x, y, z) + t.values[(t.w * t.h * z) + (t.w * y) + x];
        result.setValue(x, y, z, val);
      }
    }
  }
  return result;
}

template <typename T>
Tensor3<T>::~Tensor3() {
  if (this->values != nullptr) {
    delete[] this->values;
  }
}

template class Tensor3<float>;