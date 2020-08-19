One Ring to rule them all
---

Functional tests for Multi-Layer Perceptron implementations, using OOP paradigm

This is the classical XOR test.

<!-- TOC -->

- [Documentation](#documentation)
    - [Why 2 XOR and 3 XOR?](#why-2-xor-and-3-xor)
- [MLP implementations](#mlp-implementations)
    - [MLP implementations in VB .Net](#mlp-implementations-in-vb-net)
        - [Classic MLP](#classic-mlp)
        - [Object-oriented programming MLP](#object-oriented-programming-mlp)
        - [Matrix MLP: implementation using matrix products](#matrix-mlp-implementation-using-matrix-products)
        - [Vectorized Matrix MLP: implementation using matrix products, including samples vector](#vectorized-matrix-mlp-implementation-using-matrix-products-including-samples-vector)
        - [Tensor MLP: implementation using tensor](#tensor-mlp-implementation-using-tensor)
    - [MLP implementations using frameworks and libraries](#mlp-implementations-using-frameworks-and-libraries)
        - [Accord.NET MLP: implementation using Accord.NET Framework](#accordnet-mlp-implementation-using-accordnet-framework)
        - [Encog MLP: implementation using Encog Framework](#encog-mlp-implementation-using-encog-framework)
        - [TensorFlow MLP: implementation using TensorFlow.NET Framework](#tensorflow-mlp-implementation-using-tensorflownet-framework)
- [Versions](#versions)

<!-- /TOC -->

# Documentation

## Why 2 XOR and 3 XOR?

-> To reduce hazard and improve learning stability, see [MLPComparison.xls](MLPComparison.xls).

# MLP implementations

## MLP implementations in VB .Net

### Classic MLP
http://patrice.dargenton.free.fr/ia/ialab/perceptron.html (french)

From C++ (at 22/08/2000): https://github.com/sylbarth/mlp


### Object-oriented programming MLP
https://github.com/PatriceDargenton/multilayer-perceptron-vb (VB .Net)

From : https://github.com/RutledgePaulV/multilayer-perceptron-vb (VB .Net)


### Matrix MLP: implementation using matrix products
https://github.com/PatriceDargenton/Matrix-MultiLayerPerceptron (VB .Net)

From C#: https://github.com/PatriceDargenton/perceptrons

From C#: https://github.com/nlabiris/perceptrons


### Vectorized Matrix MLP: implementation using matrix products, including samples vector
https://github.com/PatriceDargenton/Vectorized_MultilayerPerceptron (VB .Net)

From C#: https://github.com/PatriceDargenton/Vectorized-multilayer-neural-network

From C#: https://github.com/HectorPulido/Vectorized-multilayer-neural-network


### Tensor MLP: implementation using tensor

From C#: https://github.com/HectorPulido/Machine-learning-Framework-Csharp


## MLP implementations using frameworks and libraries

Note: do the first compilation in debug mode! (there is actually a bug in VS 2019 if you do the first compilation in release mode, the packages will not be referenced in debug mode after!)

### Accord.NET MLP: implementation using Accord.NET Framework

From C#: http://accord-framework.net/docs/html/T_Accord_Neuro_Learning_BackPropagationLearning.htm

See: https://github.com/accord-net/framework

Packages added:
```
  <package id="Accord" version="3.8.2-alpha" targetFramework="net452" />
  <package id="Accord.Genetic" version="3.8.2-alpha" targetFramework="net452" />
  <package id="Accord.MachineLearning" version="3.8.2-alpha" targetFramework="net452" />
  <package id="Accord.Math" version="3.8.2-alpha" targetFramework="net452" />
  <package id="Accord.Neuro" version="3.8.2-alpha" targetFramework="net452" />
  <package id="Accord.Statistics" version="3.8.2-alpha" targetFramework="net452" />
```

### Encog MLP: implementation using Encog Framework

From C#: https://github.com/encog/encog-dotnet-core

See: https://www.heatonresearch.com/encog

Package added:
```
  <package id="encog-dotnet-core" version="3.4.0" targetFramework="net472" />
```

### TensorFlow MLP: implementation using TensorFlow.NET Framework

From C#: https://github.com/SciSharp/SciSharp-Stack-Examples/blob/master/src/TensorFlowNET.Examples/NeuralNetworks/NeuralNetXor.cs

See: https://github.com/SciSharp/TensorFlow.NET

and: https://tensorflownet.readthedocs.io/en/latest

Packages added:
```
  <package id="Google.Protobuf" version="3.11.4" targetFramework="net472" />
  <package id="Microsoft.ML.TensorFlow.Redist" version="0.14.0" targetFramework="net472" />
  <package id="NumSharp.Lite" version="0.1.7" targetFramework="net472" />
  <package id="Protobuf.Text" version="0.4.0" targetFramework="net472" />
  <package id="System.Buffers" version="4.4.0" targetFramework="net472" />
  <package id="System.Memory" version="4.5.3" targetFramework="net472" />
  <package id="System.Numerics.Vectors" version="4.4.0" targetFramework="net472" />
  <package id="System.Runtime.CompilerServices.Unsafe" version="4.5.2" targetFramework="net472" />
  <package id="TensorFlow.NET" version="0.15.1" targetFramework="net472" />
```

# Versions

19/08/2020 V1.13
- TensorFlow MLP added

11/08/2020 V1.12
- Refactored code in clsMLPGeneric: ComputeError(), ComputeAverageErrorFromLastError(), ComputeAverageError() and TestOneSample(input!(), ByRef ouput!())
- Standard test TestMLP3XORHTangent fixed
- Encog MLP added

03/08/2020 V1.11
- ActivationFunctionForMatrix ->
  ActivationFunctionOptimized

03/08/2020 V1.10
- Sigmoid and Hyperbolic Tangent (Bipolar Sigmoid) activations: optimized also with gain<>1
- Hyperbolic Tangent (Bipolar Sigmoid) activation: input/2
- Tests added for 4 and 5 layers
- Standard tests for 3 implementations
- Accord.NET MLP added (PlatformTarget: AnyCPU -> x64)

05/07/2020 V1.09
- Matrix class using Math.Net (I have not succeeded in using extension methods, only in creating a new class which uses MathNet.Numerics.LinearAlgebra.Matrix(Of Double))

25/06/2020 V1.08
- Matrix.ToVectorArraySingle() -> ToArraySingle()
- clsMLPGeneric: output Matrix instead of ouput array
- Single Matrix class: 2 times faster

06/06/2020 V1.07
- Source code cleaned
- Matrix MLP: finally weightAdjustment is not used in this implementation (only learningRate)
- Vectorized Matrix MLP: weight adjustment
- Classic MLP MatrixMLP.Matrix -> VectorizedMatrixMLP.Matrix (Perceptron.Util namespace)
- LinearAlgebra.Matrix: common for Tensor MLP and Vectorized matrix MLP (Perceptron.Util namespace)
- ComputeAverageError: in generic class
- Tests added for semi-stochastic and stochastic learning mode
- TrainSemiStochastic: fixed
- Tensor MLP added

16/05/2020 V1.06
- Vectorized Matrix MLP, OOP MLP: faster tests
- Tests: Assert rounded loss <= expected loss (instead of equality) to test other implementation without exactly the same loss

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