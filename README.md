One Ring to rule them all
---

Functional tests for Multi-Layer Perceptron implementations, using O.O.P. paradigm (Object-Oriented Programming)
==

# Introduction
What is a functional test? It is a test which allows to verify the entire functioning of a process, while a unit test allows to check a function, a class, an elementary process. How can we do a functional test for a neural network whereas weights initialization is random, how to guarantee learning process with this part of hazard? Precisely, it is necessary to eliminate any random part in order to be able to carry out a functional test. For this, it is necessary to capture the weights after the initialization of the network, check if the network works well with this random draw (if not, start again) and as soon as this draw makes it possible to reach the desired result, a new functional test is ready: it is sufficient to initialize the network identically, to reload these weights, to redo the training (with the same number of iterations), and to check that the training, the loss and the prediction are indeed always identical, whatever modifications you will now be able to make to the source code with confidence. If the datasets are very small, if the size of the network is minimal, then this list of weights is not very large, we can very well include it in the source code of a functional test. Sometimes a neural network comes with a save and reload system, that includes the whole structure of the network with its weights, but we only need the weights, not the structure, since this structure will be already redefined in the functional test. We therefore need a procedure to display the network weights, rounded for example to two decimal digits, as well as a procedure to reload these weights.

# Example
Let see a functionnal test for a small learning example:
```
    Public Sub TestMLP1XOR(mlp As clsMLPGeneric,
        Optional nbIterations% = 5000,
        Optional expectedLoss# = 0.04#,
        Optional learningRate! = 0.05!,
        Optional weightAdjustment! = 0.1!,
        Optional gain! = 2)

        mlp.inputArray = {
            {1, 0},
            {0, 0},
            {0, 1},
            {1, 1}}
        mlp.targetArray = {
            {1},
            {0},
            {1},
            {0}}
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.InitializeStruct({2, 2, 1}, addBiasColumn:=True)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)

        mlp.InitializeWeights(1, {
            {-0.75, 0.64, -0.09},
            {0.12, 0.75, -0.63}})
        mlp.InitializeWeights(2, {
            {-0.79, -0.13, 0.58}})

        mlp.Train()

        Dim expectedOutput = mlp.targetArray
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub
```

# Table of contents

<!-- TOC -->

- [Functional tests for Multi-Layer Perceptron implementations, using O.O.P. paradigm Object-Oriented Programming](#functional-tests-for-multi-layer-perceptron-implementations-using-oop-paradigm-object-oriented-programming)
- [Introduction](#introduction)
- [Example](#example)
- [Table of contents](#table-of-contents)
- [List of small datasets tested](#list-of-small-datasets-tested)
- [List of frameworks and libraries remaining to be tested](#list-of-frameworks-and-libraries-remaining-to-be-tested)
- [MLP implementations in VB .Net](#mlp-implementations-in-vb-net)
    - [Classic MLP](#classic-mlp)
    - [Object-oriented programming MLP](#object-oriented-programming-mlp)
    - [Matrix MLP: implementation using matrix products](#matrix-mlp-implementation-using-matrix-products)
    - [Vectorized Matrix MLP: implementation using matrix products, including samples vector](#vectorized-matrix-mlp-implementation-using-matrix-products-including-samples-vector)
    - [Tensor MLP: implementation using tensor](#tensor-mlp-implementation-using-tensor)
    - [RProp MLP: implementation using Resilient Back Propagation algorithm](#rprop-mlp-implementation-using-resilient-back-propagation-algorithm)
- [MLP implementations using frameworks and libraries](#mlp-implementations-using-frameworks-and-libraries)
    - [Accord.NET MLP: implementation using Accord.NET Framework](#accordnet-mlp-implementation-using-accordnet-framework)
    - [Encog MLP: implementation using Encog Framework](#encog-mlp-implementation-using-encog-framework)
    - [TensorFlow MLP: implementation using TensorFlow.NET Framework](#tensorflow-mlp-implementation-using-tensorflownet-framework)
    - [Keras MLP: implementation using Keras.NET Framework](#keras-mlp-implementation-using-kerasnet-framework)
    - [NeuralNet MLP: implementation using NeuralNet.NET Framework](#neuralnet-mlp-implementation-using-neuralnetnet-framework)
- [MLP comparison](#mlp-comparison)
- [Version history](#version-history)

<!-- /TOC -->

# List of small datasets tested
- The classical XOR test, and also 2 XOR and 3 XOR (to reduce hazard and improve learning stability, see [MLPComparison.xls](MLPComparison.xls));
- The [Iris flower](https://en.wikipedia.org/wiki/Iris_flower_data_set) test;
- The [Sunspots](https://courses.cs.washington.edu/courses/cse599/01wi/admin/Assignments/nn.html) test.

# List of frameworks and libraries remaining to be tested
- [ML.NET](https://github.com/dotnet/machinelearning) (Microsoft Machine Learning for .NET): ML.NET requires the definition of a class representing the object to learn and predict, if we want to be able to avoid this definition and do tests in a generic way, then it is specified in the FAQ that it is necessary to use the FeatureVector, but I did not find any example of implementation!
- [CNTK](https://github.com/microsoft/CNTK) (Microsoft Cognitive Toolkit)
- [Gym.NET](https://github.com/SciSharp/Gym.NET)
- [Bright Wire](https://github.com/jdermody/brightwire)
- (forgot some?)

# MLP implementations in VB .Net

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


## Tensor MLP: implementation using tensor

From C#: https://github.com/HectorPulido/Machine-learning-Framework-Csharp


## RProp MLP: implementation using Resilient Back Propagation algorithm

From C#: https://github.com/nokitakaze/ResilientBackProp

Note: Multithread mode doesn't work very well, it only works well when the ThreadCount is an exact multiple of the sample number in the dataset. A solution remains to be found therefore.

# MLP implementations using frameworks and libraries

Note: do the first compilation in debug mode! (there is actually a bug in VS 2019 if you do the first compilation in release mode, the packages will not be referenced in debug mode after!)

## Accord.NET MLP: implementation using Accord.NET Framework

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

## Encog MLP: implementation using Encog Framework

From C#: https://github.com/encog/encog-dotnet-core

See: https://www.heatonresearch.com/encog

Package added:
```
  <package id="encog-dotnet-core" version="3.4.0" targetFramework="net472" />
```

## TensorFlow MLP: implementation using TensorFlow.NET Framework

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

## Keras MLP: implementation using Keras.NET Framework

From C#: https://github.com/SciSharp/Keras.NET

See: https://scisharp.github.io/Keras.NET

and: https://keras.io

Packages added:
```
  <package id = "Keras.NET" version="3.7.4.2" targetFramework="net472" />
  <package id = "Microsoft.CSharp" version="4.5.0" targetFramework="net472" />
  <package id = "Numpy.Bare" version="3.7.1.11" targetFramework="net472" />
  <package id = "Python.Runtime.NETStandard" version="3.7.1" targetFramework="net472" />
  <package id = "System.Reflection.Emit" version="4.3.0" targetFramework="net472" />
```
```
 Python 3.7 is required at runtime: https://www.python.org/downloads
 For PowerShell installations, type:
 python -mpip install numpy      : fix "No module named 'numpy'"
 python -mpip install keras      : fix "No module named 'keras'"
 python -mpip install tensorflow : fix "Keras requires TensorFlow 2.2 or higher"
```

## NeuralNet MLP: implementation using NeuralNet.NET Framework

From C# : https://github.com/Sergio0694/NeuralNetwork.NET

https://www.nuget.org/packages/NeuralNetwork.NET Nuget install

https://scisharp.github.io/SciSharp Other .NET Machine Learning projects

GetWeights and SetWeights are required for functional tests, see there:
https://github.com/PatriceDargenton/NeuralNetwork.NET/tree/get_set_weights

The stable branch contains the get_set_weights and other branches:
https://github.com/PatriceDargenton/NeuralNetwork.NET/tree/stable

Packages updated:
```
System.Buffers.4.4.0 -> System.Buffers.4.5.1
System.Numerics.Vectors.4.4.0 -> System.Numerics.Vectors.4.5.0
```
Packages added:
```
    Alea.3.0.4
    FSharp.Core.4.2.3
    JetBrains.Annotations.2018.2.1
    Newtonsoft.Json.11.0.2
    SharpZipLib.NETStandard.1.0.7
    SixLabors.ImageSharp.1.0.0
    NeuralNetwork.NET.2.1.3
  <package id = "Keras.NET" version="3.7.4.2" targetFramework="net472" />
  <package id = "Microsoft.CSharp" version="4.5.0" targetFramework="net472" />
  <package id = "Numpy.Bare" version="3.7.1.11" targetFramework="net472" />
  <package id = "Python.Runtime.NETStandard" version="3.7.1" targetFramework="net472" />
  <package id = "System.Reflection.Emit" version="4.3.0" targetFramework="net472" />
```
```
  To fix SixLabors.ImageSharp FileLoadException (0x80131040) bug:
  Install-Package SixLabors.ImageSharp -Version 1.0.0-beta0007
  (bug starting from 1.0.0-rc0001 version)
```

Note: To update the packages, you will need to restore this deleted line in the project file (in order to compile with VS2013):
```
<Import Project="packages\Microsoft.Net.Compilers.Toolset.3.9.0\build\Microsoft.Net.Compilers.Toolset.props" Condition="Exists('packages\Microsoft.Net.Compilers.Toolset.3.9.0\build\Microsoft.Net.Compilers.Toolset.props')" />
```

# MLP comparison

[MLPComparison.xls](MLPComparison.xls)

# Version history

20/03/2021 V1.35
- Sunspots tests and matrix size checks fixed

18/03/2021 V1.34
- Console demo: MLP menu added

18/03/2021 V1.33
- Packages updated:
```
Packages updated:
----------------
Protobuf.Text.0.4.0 -> Protobuf.Text.0.5.0
NumSharp.Lite.0.1.7 -> NumSharp.Lite.0.1.12
Google.Protobuf.3.11.4 -> Google.Protobuf.3.15.6
System.Memory.4.5.3 -> System.Memory.4.5.4
System.Runtime.CompilerServices.Unsafe.4.5.2 -> System.Runtime.CompilerServices.Unsafe.5.0.0
Keras.NET.3.7.4.2 -> Keras.NET.3.8.5
Numpy.Bare.3.7.1.11 -> Numpy.Bare.3.8.1.25
System.Reflection.Emit.4.3.0 -> System.Reflection.Emit.4.7.0
Newtonsoft.Json.11.0.2 -> Newtonsoft.Json.12.0.3
Microsoft.Net.Compilers.Toolset.3.1.0 -> Microsoft.Net.Compilers.Toolset.3.9.0
Microsoft.CSharp.4.5.0 -> Microsoft.CSharp.4.7.0
JetBrains.Annotations.2018.2.1 -> JetBrains.Annotations.2020.3.0
FSharp.Core.4.2.3 -> FSharp.Core.5.0.1

Package added:
-------------
pythonnet_netstandard_py38_win.2.5.1 -> pythonnet_netstandard_py38_win.2.5.1.1

Packages not updated:
--------------------
SixLabors.ImageSharp.1.0.0-beta0007 -> SixLabors.ImageSharp.1.0.3: FileLoadException (0x80131040)
TensorFlow.NET.0.15.1 -> TensorFlow.NET.0.40.0: tf.placeholder() is not compatible with eager execution
```
- Python 3.7 -> Python 3.8

14/03/2021 V1.32
- clsMLPNeuralNetLib added: NeuralNet.Net Framework

30/01/2021 V1.31
- Sunspots dataset added (time series dataset)
- clsMLPGeneric: series array (for example time series)

09/01/2021 V1.30
- clsMLPGeneric.nbHiddenNeurons -> moved to specific classes

09/01/2021 V1.29
- HTangent -> Tanh

08/01/2021 V1.28
- Matrix.ToArraySingle -> ToArrayOfSingle

08/01/2021 V1.27
- Dataset directory

03/01/2021 V1.26
- clsMLPTensorFlow.TestOneSample fixed
- clsMLPClassic.TestOneSample fixed
- modMLPTest.TestMLP2XORSigmoid fixed
- modActivation: Sigmoid and Tanh limits fixed
- clsRndExtension: NextDoubleGreaterThanZero added (RProp MLP)
- clsMLPGenericVec.TrainVectorBatch: nbIterations computed from nbIterationsBatch
- clsMLPGeneric.ShowWeights added, to compare configurations
- clsMLPGeneric.classificationObjective added (RProp MLP)
- clsMLPGeneric.useNguyenWidrowWeightsInitialization added (RProp MLP)
- clsMLPGeneric.minRandomValue added (RProp MLP)
- clsMLPRProp added

12/12/2020 V1.25
- clsMLPGeneric.averageError: Single -> Double

06/12/2020 V1.24
- clsMLPGeneric.InitializeStruct function
- clsMLPAccord: Resilient Backpropagation Learning
- clsMLPEncog: Classic Backpropagation Learning
- clsMLPKeras: Randomize(): weights rounding fixed
- clsMLPTensor: InitializeSequential() fixed

21/11/2020 V1.23
- Console menu added
- Iris flower prediction analog test added
- clsVecMatrix.ComputeErrorOneSample() : clsMLPGeneric's version used
- clsMLPGeneric.GetActivationFunctionType() added with enumActivationFunctionType
- clsMLPGeneric.RoundWeights() added
- clsMLPGeneric.ComputeErrorOneSample(targetArray!(,)) added
- clsMLPGeneric.ComputeAverageErrorOneSample!(targetArray!(,)) added
- clsMLPGeneric.CloseSession() -> CloseTrainingSession()
- clsMLPHelper.Fill2DArrayOfDouble function fixed
- Gaussian activation function derivate fixed
- Sinus activation function derivate fixed
- clsMLPAccord.TestOneSample() : output matrix setted
- clsMLPAccord averageError fixed
- clsMLPAccord PrintWeights() fixed
- clsMLPEncog averageError fixed
- clsMLPTensor : InitializeSequential() added and InitializeGradient() renamed
- clsMLPTensor averageError fixed

11/10/2020 V1.22
- Randomize: Default range: [0, 1] -> [-0.5, +0.5]
- clsMLPOOP: Activation function: BipolarSigmoid renamed as HyperbolicTangent (and previous HyperbolicTangent deleted)
- clsTensorMLP, clsMatrixMLP: Iris flower prediction test added
- Compilation: Strict mode enabled

04/10/2020 V1.21
- Iris flower prediction test added
- Hyperbolic Tangent (Tanh) derivative fixed
- clsMLPGeneric.TestAllSamples: simplified
- clsMLPGeneric.PrintParameters: minimalSuccessTreshold displayed
- clsMLPGeneric.ShowThisIteration: also for last iteration
- clsTensorMLP: nbHiddenNeurons must be identical to nbInputNeurons

26/09/2020 V1.20
- clsMLPTensor optimized

19/09/2020 V1.19
- Hyperbolic Tangent (Tanh) gain inversion: gain:=-2 -> gain:=2

19/09/2020 V1.18
- clsMLPGeneric.Initialize: weightAdjustment optional

19/09/2020 V1.17
- MatrixMLP: works fine, 3XOR tests added with three activation functions
- clsVectorizedMLPGeneric.neuronCount -> clsMLPGeneric, and displayed in PrintParameters()
- Compute success and fails after Train()
- Iris flower test added: https://en.wikipedia.org/wiki/Iris_flower_data_set

29/08/2020 V1.16
- Activation function: gain and center optional

29/08/2020 V1.15
- TensorMLP: SetOuput1D -> SetOuput1DOneSample (not all samples)
- Learning mode added: VectorialBatch (learn all samples in order as a vector for a batch of iterations)
- PrintWeights added for one XOR tests
- PrintOutput: option force display added
- Refactored code in clsMLPGenericVec: TrainVectorOneIteration(), SetOuput1D()
- Keras MLP added

21/08/2020 V1.14
- ComputeSuccess added
- 2 XOR and 3 XOR added for all tests in console mode
- Refactored code in clsMLPGeneric: PrintOutput(iteration%)

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