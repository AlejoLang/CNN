# CNN

Implementation of a CNN on C++.
To change the network configuration, `src/main.cpp` needs to be changed and the project has to be built again.
The code uses the MNIST digits dataset to train the network, [this one for example](https://www.kaggle.com/datasets/avnishnish/mnist-original).

```bash
./CNN --train <path-to-mat-file> -O <path-to-output-file>
```

The test of the network is done with [SDL](https://www.libsdl.org/) and can be used once generated the .bin file with the network weights using

```bash
./CNN --test <path-to-bin-file>
```

