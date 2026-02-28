#include <Canvas.hpp>
#include <ConvolutionalLayer.hpp>
#include <DenseLayer.hpp>
#include <FlattenLayer.hpp>
#include <GAP.hpp>
#include <MaxPoolLayer.hpp>
#include <Network.hpp>
#include <SDL3/SDL.h>
#include <SDL3/SDL_rect.h>
#include <SDL3/SDL_render.h>
#include <SDL3/SDL_video.h>
#include <Tensor3.hpp>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <matio.h>
#include <random>
#include <vector>

void load_data(std::string path, std::vector<Tensor3<float>>& images,
               std::vector<Tensor3<float>>& labels) {
  mat_t* dataset = Mat_Open(path.c_str(), MAT_ACC_RDONLY);
  if (!dataset) {
    std::cerr << "Couldn't open the file" << std::endl;
    return;
  }
  matvar_t* dataVar = Mat_VarRead(dataset, "data");
  if (!dataVar) {
    std::cerr << "Cannot find variable 'data'\n";
    Mat_Close(dataset);
    return;
  }

  if (!dataVar->data) {
    std::cerr << "Data variable has no data\n";
    Mat_VarFree(dataVar);
    Mat_Close(dataset);
    return;
  }

  size_t rows = dataVar->dims[0]; // 784
  size_t cols = dataVar->dims[1]; // 70000

  std::cout << "Data dimensions: " << rows << " x " << cols << std::endl;
  std::cout << "Data class type: " << dataVar->class_type << std::endl;

  images = std::vector<Tensor3<float>>(cols, Tensor3<float>(28, 28, 1));

  // Handle different data types
  if (dataVar->class_type == MAT_C_DOUBLE) {
    double* data = static_cast<double*>(dataVar->data);

    // Some MNIST .mat files store pixels as double in [0,255].
    // Detect that case and normalize to [0,1] to avoid sigmoid saturation.
    double maxPixel = 0.0;
    for (size_t idx = 0; idx < rows * cols; ++idx) {
      if (data[idx] > maxPixel) {
        maxPixel = data[idx];
      }
    }
    const double scale = (maxPixel > 1.0) ? 255.0 : 1.0;

    for (size_t c = 0; c < cols; ++c)
      for (size_t r = 0; r < rows; ++r)
        images[c].setValue(r % 28, r / 28, 0, static_cast<float>(data[r + c * rows] / scale));
  } else if (dataVar->class_type == MAT_C_UINT8) {
    uint8_t* data = static_cast<uint8_t*>(dataVar->data);
    for (size_t c = 0; c < cols; ++c)
      for (size_t r = 0; r < rows; ++r)
        images[c].setValue(r % 28, r / 28, 0,
                           static_cast<float>(data[r + c * rows]) / 255.0f); // Normalize to [0, 1]
  } else {
    std::cerr << "Unsupported data type: " << dataVar->class_type << std::endl;
    Mat_VarFree(dataVar);
    Mat_Close(dataset);
    return;
  }

  Mat_VarFree(dataVar);

  /* -------- Read labels -------- */
  matvar_t* labelVar = Mat_VarRead(dataset, "label");
  if (!labelVar) {
    std::cerr << "Cannot find variable 'label'\n";
    Mat_Close(dataset);
    return;
  }

  if (!labelVar->data) {
    std::cerr << "Label variable has no data\n";
    Mat_VarFree(labelVar);
    Mat_Close(dataset);
    return;
  }

  labels = std::vector<Tensor3<float>>(cols, Tensor3<float>(10, 1, 1));

  if (labelVar->class_type == MAT_C_DOUBLE) {
    double* labels_raw = static_cast<double*>(labelVar->data);
    for (size_t i = 0; i < cols; ++i)
      labels[i].setValue(static_cast<int>(labels_raw[i]), 0, 0, 1);
  } else if (labelVar->class_type == MAT_C_UINT8) {
    uint8_t* labels_raw = static_cast<uint8_t*>(labelVar->data);
    for (size_t i = 0; i < cols; ++i)
      labels[i].setValue(static_cast<int>(labels_raw[i]), 0, 0, 1);
  } else if (labelVar->class_type == MAT_C_SINGLE) {
    float* labels_raw = static_cast<float*>(labelVar->data);
    for (size_t i = 0; i < cols; ++i)
      labels[i].setValue(static_cast<int>(labels_raw[i]), 0, 0, 1);
  } else {
    std::cerr << "Unsupported label type: " << labelVar->class_type << std::endl;
  }
  Mat_VarFree(labelVar);
  Mat_Close(dataset);
}

struct TrainItem {
  Tensor3<float> image;
  Tensor3<float> label;
};

void train_mode(Network& net, const std::vector<Tensor3<float>>& images,
                const std::vector<Tensor3<float>>& labels, std::string savePath);
void test_mode(Network& net);

Tensor3<float> augment(const Tensor3<float>& src, std::mt19937& rng) {
  std::uniform_int_distribution<int> shift_dist(-2, 2);
  std::uniform_real_distribution<float> zoom_dist(0.9f, 1.1f);

  int dx = shift_dist(rng);
  int dy = shift_dist(rng);
  float scale = zoom_dist(rng);

  Tensor3<float> out(28, 28, 1);
  float cx = 13.5f, cy = 13.5f;

  for (int y = 0; y < 28; y++) {
    for (int x = 0; x < 28; x++) {
      float sx = (x - cx) / scale + cx - dx;
      float sy = (y - cy) / scale + cy - dy;

      int ix = static_cast<int>(std::round(sx));
      int iy = static_cast<int>(std::round(sy));

      if (ix >= 0 && ix < 28 && iy >= 0 && iy < 28)
        out.setValue(x, y, 0, const_cast<Tensor3<float>&>(src).getValue(ix, iy, 0));
    }
  }
  return out;
}

std::vector<TrainItem> augment_dataset(const std::vector<TrainItem>& trainItem, std::mt19937& rng) {
  std::vector<TrainItem> augmented;
  for (const auto& item : trainItem) {
    augmented.push_back({augment(item.image, rng), item.label});
  }
  return augmented;
}

int main(int argc, char* argv[]) {

  Network net = Network();

  // --test (path to .bin file) or --train(path to .mat file)
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " --train <path_to_mat_file> OR " << argv[0]
              << " --test <path_to_bin_file>" << std::endl;
    return 1;
  }
  std::string mode = argv[1];
  if (mode == "--train") {
    if (argc < 3) {
      std::cerr << "No .mat specified" << std::endl;
      return 1;
    }
    std::vector<Tensor3<float>> images, labels;
    load_data(argv[2], images, labels);
    train_mode(net, images, labels, "mnist_cnn_weights.bin");

  } else if (mode == "--test") {
    if (argc < 3) {
      std::cerr << "No .bin file specified" << std::endl;
      return 1;
    }
    net.loadWeights(argv[2]);
    test_mode(net);
  } else {
    std::cerr << "Unknown mode: " << mode << ". Use --train or --test" << std::endl;
    return 1;
  }

  return 0;
}

void train_mode(Network& net, const std::vector<Tensor3<float>>& images,
                const std::vector<Tensor3<float>>& labels, std::string savePath) {

  net.addLayer(new ConvolutionalLayer(3, 1, 8));
  net.addLayer(new MaxPoolLayer(2, 8));
  net.addLayer(new ConvolutionalLayer(3, 8, 16));
  net.addLayer(new MaxPoolLayer(2, 16));
  net.addLayer(new FlattenLayer(5, 5, 16));
  net.addLayer(new DenseLayer(5 * 5 * 16, 120));
  net.addLayer(new DenseLayer(120, 84));
  net.addLayer(new DenseLayer(84, 10));

  std::vector<TrainItem> samples;
  for (size_t i = 0; i < images.size(); i++) {
    samples.push_back({images[i], labels[i]});
  }
  std::mt19937 rng(std::random_device{}());
  std::shuffle(samples.begin(), samples.end(), rng);
  std::vector<TrainItem> testData(samples.begin() + samples.size() * 0.8, samples.end());
  std::vector<TrainItem> trainData(samples.begin(), samples.begin() + samples.size() * 0.8);

  std::shuffle(trainData.begin(), trainData.end(), rng);
  trainData = augment_dataset(trainData, rng);

  for (size_t epoch = 0; epoch < 10; epoch++) {
    float totalLoss = 0.0f;
    // Measure time each 1000 samples
    auto startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < (int)trainData.size(); i++) {
      const TrainItem& item = trainData[i];
      Tensor3<float> output = net.forward(item.image);
      net.backwards(output, item.label);
      net.update(0.01f);

      // Calculate loss (MSE)
      float sampleLoss = 0.0f;
      for (size_t j = 0; j < output.getWidth(); j++) {
        float predicted = output.getValue(j, 0, 0);
        float actual = const_cast<Tensor3<float>&>(item.label).getValue(j, 0, 0);
        sampleLoss += (predicted - actual) * (predicted - actual);
      }
      totalLoss += sampleLoss / output.getChannels();
      if (i % 1000 == 0 && i > 0) {
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        std::cout << "Processed " << i << " samples in " << elapsed.count() << " seconds. "
                  << " Sample loss: " << sampleLoss / output.getChannels() << std::endl;
        startTime = std::chrono::high_resolution_clock::now();
      }
    }
    std::cout << "Epoch " << epoch + 1 << ", Loss: " << totalLoss / (trainData.size()) << std::endl;
    std::shuffle(trainData.begin(), trainData.end(), rng);
    trainData = augment_dataset(trainData, rng);
  }
  // Evaluate on test data
  int correct = 0;
  for (const TrainItem& item : testData) {
    Tensor3<float> output = net.forward(item.image);
    int predictedLabel = 0;
    float maxVal = output.getValue(0, 0, 0);
    for (size_t j = 1; j < (size_t)output.getWidth(); j++) {
      float val = output.getValue(j, 0, 0);
      if (val > maxVal) {
        maxVal = val;
        predictedLabel = j;
      }
    }
    int actualLabel = 0;
    for (size_t j = 0; j < (size_t)const_cast<Tensor3<float>&>(item.label).getWidth(); j++) {
      if (const_cast<Tensor3<float>&>(item.label).getValue(j, 0, 0) == 1) {
        actualLabel = j;
        break;
      }
    }
    if (predictedLabel == actualLabel) {
      correct++;
    }
  }
  std::cout << "Test Accuracy: " << (float)correct / testData.size() * 100 << "%" << std::endl;
  net.saveWeights(savePath);
}

void test_mode(Network& net) {
  SDL_Init(SDL_INIT_VIDEO);
  SDL_Window* window = SDL_CreateWindow("CNN Test", 640, 480, SDL_WINDOW_RESIZABLE);
  SDL_Renderer* renderer = SDL_CreateRenderer(window, NULL);
  Canvas canvas = Canvas(28, 28, renderer);
  int wh, ww;
  SDL_GetWindowSize(window, &wh, &ww);
  SDL_FRect rect = {0, 0, (float)wh, (float)ww};
  bool exit = false;
  bool mousePressed = false;
  while (!exit) {
    SDL_RenderClear(renderer);
    canvas.render(renderer, &rect);
    SDL_RenderPresent(renderer);

    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      switch (event.type) {
      case SDL_EVENT_MOUSE_BUTTON_DOWN: {
        if (event.button.button == SDL_BUTTON_LEFT) {

          mousePressed = true;
        }
        break;
      }
      case SDL_EVENT_MOUSE_BUTTON_UP: {
        if (event.button.button == SDL_BUTTON_LEFT) {
          mousePressed = false;
        }
        break;
      }
      case SDL_EVENT_MOUSE_MOTION: {
        if (mousePressed) {
          int x = event.motion.x * 28 / wh;
          int y = event.motion.y * 28 / ww;
          canvas.setPixel(x, y, 0xFFFFFFFF);
          canvas.setPixel(x + 1, y, 0xFFFFFFFF);
          canvas.setPixel(x - 1, y, 0xFFFFFFFF);
          canvas.setPixel(x, y + 1, 0xFFFFFFFF);
          canvas.setPixel(x, y - 1, 0xFFFFFFFF);
        }
        break;
      }
      case SDL_EVENT_KEY_DOWN: {
        if (event.key.key == SDLK_C) {
          canvas.clear(0x00000000);
          break;
        }
        if (event.key.key == SDLK_RETURN) {
          uint32_t* pixels = canvas.getBuffer();
          Tensor3<float> input = Tensor3<float>(28, 28, 1);
          Tensor3<float> output = net.forward(input);
          for (size_t i = 0; i < output.getWidth(); i++) {
            std::cout << i << ": " << std::fixed << std::setprecision(2)
                      << output.getValue(i, 0, 0) * 100 << "%" << std::endl;
          }
          break;
        }
        break;
      }
      case SDL_EVENT_QUIT: {
        exit = true;
        break;
      }
      default:
        break;
      }
    }
  }

  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
}