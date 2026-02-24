#pragma once

template <typename T>
class Tensor3 {
private:
  int w, h, c;
  T* values;

public:
  Tensor3(int width, int height, int channels);
  int getWidth();
  int getHeight();
  int getChannels();
  T getValue(int x, int y, int z);
  void setValue(int x, int y, int z, T value);
  Tensor3<T> operator+(const Tensor3<T>& t);
  ~Tensor3();
};