# One Ring to rule them all
Functional tests for Multi-Layer Perceptron implementations, using OOP paradigm

This is the classical XOR test.

Why 2 XOR and 3 XOR?

-> To reduce hazard and improve learning stability.

## Classic MLP
http://patrice.dargenton.free.fr/ia/ialab/perceptron.html (french)

From C++ (at 22/08/2000): https://github.com/sylbarth/mlp

## Object-oriented programming MLP
https://github.com/PatriceDargenton/multilayer-perceptron-vb (VB .Net)

From : https://github.com/RutledgePaulV/multilayer-perceptron-vb (VB .Net)

## Matrix MLP: implementation using matrix products
https://github.com/PatriceDargenton/Matrix-MultiLayerPerceptron (VB .Net)

From C#: https://github.com/PatriceDargenton/perceptrons

From C#: https://github.com/nlabiris/perceptrons


## Vectorized Matrix MLP: implementation using matrix products, including samples vector
https://github.com/PatriceDargenton/Vectorized_MultilayerPerceptron (VB .Net)

From C#: https://github.com/PatriceDargenton/Vectorized-multilayer-neural-network

From C#: https://github.com/HectorPulido/Vectorized-multilayer-neural-network


# Versions

10/05/2020 V1.05
- Vectorized Matrix MLP: faster tests

10/05/2020 V1.04
- Homogenization of function names
- Vectorized Matrix MLP: standard tests
- clsMLPGeneric: PrintParameters: parameters added

02/05/2020 V1.03
- OOP MLP version
- MatrixMLP: PrintOutput fixed
- Vectorized Matrix MLP: faster tests
- PrintParameters: activation function name displayed

17/04/2020 V1.02
- clsMLPGeneric: MustOverride WeightInit(layer%, weights#(,))
- Activation function added: Double threshold
- Print output standardized
- Variable names simplification

12/04/2020 V1.01 Initial commit