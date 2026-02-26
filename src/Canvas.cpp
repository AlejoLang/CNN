#include <Canvas.hpp>

Canvas::Canvas(int width, int height, SDL_Renderer* renderer) {
  this->width = width;
  this->height = height;
  this->pixels = new uint32_t[width * height]();
  this->texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING,
                                    width, height);
  SDL_SetTextureScaleMode(this->texture, SDL_ScaleMode::SDL_SCALEMODE_PIXELART);
}

uint32_t* Canvas::getBuffer() {
  return this->pixels;
}
int Canvas::getWidth() {
  return this->width;
}
int Canvas::getHeight() {
  return this->height;
}
uint32_t Canvas::getPixel(int x, int y) {
  return this->pixels[y * this->width + x];
}

void Canvas::setPixel(int x, int y, uint32_t color) {
  if (x < 0 || x >= this->width || y < 0 || y >= this->height) {
    return;
  }
  this->pixels[y * this->width + x] = color;
}

void Canvas::clear(uint32_t color) {
  for (size_t i = 0; i < (size_t)(this->width * this->height); i++) {
    this->pixels[i] = color;
  }
}

void Canvas::render(SDL_Renderer* renderer, SDL_FRect* rect) {
  SDL_UpdateTexture(this->texture, NULL, this->pixels, this->width * sizeof(uint32_t));
  SDL_RenderTexture(renderer, this->texture, NULL, rect);
}

Canvas::~Canvas() {
  if (this->texture != nullptr) {
    SDL_DestroyTexture(this->texture);
  }
  delete[] this->pixels;
}