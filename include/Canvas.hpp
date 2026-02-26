#pragma once
#include <SDL3/SDL_render.h>
#include <SDL3/SDL_video.h>

class Canvas {
private:
  SDL_Texture* texture;
  uint32_t* pixels;
  int width;
  int height;

public:
  Canvas(int width, int height, SDL_Renderer* renderer);
  uint32_t* getBuffer();
  int getWidth();
  int getHeight();
  uint32_t getPixel(int x, int y);
  void setPixel(int x, int y, uint32_t color);
  void clear(uint32_t color);
  void render(SDL_Renderer* renderer, SDL_FRect* rect = NULL);
  ~Canvas();
};