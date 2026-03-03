# CNN

Implementation of a CNN in C++.
To change the network configuration, `src/main.cpp` needs to be changed and the project has to be built again.

## Training
The code uses the MNIST digits dataset to train the network on a specific mat format, [like this](https://www.kaggle.com/datasets/avnishnish/mnist-original).
```bash
./CNN --train <path-to-mat-file> -O <path-to-output-file>
```

## Testing
The test of the network is performed with [SDL](https://www.libsdl.org/) wich creates a canvas in wich digits can be drawn and uppon pressing [Enter] the results will be shown on console, using [C] will clear the canvas
To test, a .bin file with the weights is needed and it's loaded using:

```bash
./CNN --test <path-to-bin-file>
```

