One Ring to rule them all
---

Functional tests for Multi-Layer Perceptron implementations, using O.O.P. paradigm (Object-Oriented Programming)
==

<p align="center">
<img src="http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-tanh-221-big2.gif" alt="xor-tanh-221-big2.gif"
  title = "Machine Learning and Deep Learning simply minimize the distance between the crosses and the curve (this is the XOR example here: Exclusive or)" />
    <br>
    <em>XOR: activation: tanh, structure: {2, 2, 1}</em>
    <br>
    <em>Machine Learning and Deep Learning simply minimize the distance between the crosses and the curve (this is the XOR example here: Exclusive or)</em>
    <br>
</p>

# Table of contents

<!-- TOC -->

- [Functional tests for Multi-Layer Perceptron implementations, using O.O.P. paradigm Object-Oriented Programming](#functional-tests-for-multi-layer-perceptron-implementations-using-oop-paradigm-object-oriented-programming)
- [Table of contents](#table-of-contents)
- [More animated gifs](#more-animated-gifs)
- [Introduction](#introduction)
- [Example](#example)
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
    - [Bright Wire MLP: implementation using Bright Wire Framework](#bright-wire-mlp-implementation-using-bright-wire-framework)
- [MLP comparison](#mlp-comparison)
- [Version history](#version-history)

<!-- /TOC -->

# More animated gifs

Here are animations, according to different structures and activation functions, with the logic gate XOR (exclusive OR) which, as it has only 2 inputs, can be represented in 3D, even during learning. The output is drawn on the verticale axis, according to the two horizontal and perpendicular inputs. It is a logical gate, but nothing prevents it from being drawn as if it were an analog function, and even, why not, from adding points! This is the simplest example that can illustrate the learning algorithm of the multilayer perceptron, and historically the one which made it possible to definitively validate the principle of neural networks in 1986, which perceptron is still used today in Deep Learning: this is the "FullyConnected" method used in most of these software libraries, and this software brick is even the heart of neural networks; there are also for example the convolutional networks, but the idea then is just to carry out a preprocessing directly on the pixels of the image, otherwise the principle remains quite similar. If we save the weights of the neural network and later restore them, then the learning algorithm is perfectly deterministic, which has the advantage of being able to program functional tests, to ensure that the software works always correctly, when adding features.

<p align="center">
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-tanh-221.gif"
        title = "XOR: activation: tanh, structure: {2, 2, 1}" />
    <br><em>XOR: activation: tanh, structure: {2, 2, 1}</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-tanh-2221.gif"
        title = "XOR: activation: tanh, structure: {2, 2, 2, 1}" />
    <br><em>XOR: activation: tanh, structure: {2, 2, 2, 1}</em>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-sigmoid-221.gif"
        title = "XOR: activation: sigmoid, structure: {2, 2, 1}" />
    <br><em>XOR: activation: sigmoid, structure: {2, 2, 1}</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-elu-221.gif"
        title = "XOR: activation: ELU (Exponential Linear Units), structure: {2, 2, 1}" />
    <br><em>XOR: activation: ELU (Exponential Linear Units), structure: {2, 2, 1}</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-double-threshold-221.gif"
        title = "XOR: activation: double threshold, structure: {2, 2, 1}" />
    <br><em>XOR: activation: double threshold, structure: {2, 2, 1}</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-relu-221.gif"
        title = "XOR: activation: ReLU (Rectified Linear Units), structure: {2, 2, 1}" />
    <br><em>XOR: activation: ReLU (Rectified Linear Units), structure: {2, 2, 1}</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-gaussian-221.gif"
        title = "XOR: activation: gaussian, structure: {2, 2, 1}" />
    <br><em>XOR: activation: gaussian, structure: {2, 2, 1}</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-arctan-221.gif"
        title = "XOR: activation: arctan, structure: {2, 2, 1}" />
    <br><em>XOR: activation: arctan, structure: {2, 2, 1}</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-sinus-221.gif"
        title = "XOR: activation: sinus, structure: {2, 2, 1}" />
    <br><em>XOR: activation: sinus, structure: {2, 2, 1}</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-sinus-221-zoomed-out.gif"
        title = "XOR: activation: sinus, structure: {2, 2, 1} (from afar)" />
    <br><em>XOR: activation: sinus, structure: {2, 2, 1} (from afar)</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-mish-221.gif"
        title = "XOR: activation: mish, structure: {2, 2, 1}" />
    <br><em>XOR: activation: mish, structure: {2, 2, 1}</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-analog-elu-221.gif"
        title = "XOR-analog: activation: ELU (Exponential Linear Units), structure: {2, 2, 1}" />
    <br><em>XOR-analog: activation: ELU (Exponential Linear Units), structure: {2, 2, 1}</em>
    <br>
        <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-analog-elu-231.gif"
        title = "XOR-analog: activation: ELU (Exponential Linear Units), structure: {2, 3, 1}" />
    <br><em>XOR-analog: activation: ELU (Exponential Linear Units), structure: {2, 3, 1}</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-analog-tanh-231.gif"
        title = "XOR-analog: activation: tanh, structure: {2, 3, 1}" />
    <br><em>XOR-analog: activation: tanh, structure: {2, 3, 1}</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-analog-tanh-2331.gif"
        title = "XOR-analog: activation: tanh, structure: {2, 3, 3, 1}" />
    <br><em>XOR-analog: activation: tanh, structure: {2, 3, 3, 1}</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-analog-double-threshold-241b.gif"
        title = "XOR-analog: activation: double threshold, structure: {2, 4, 1}" />
    <br><em>XOR-analog: activation: double threshold, structure: {2, 4, 1}</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-analog-relu-241.gif"
        title = "XOR-analog: activation: ReLU (Rectified Linear Units), structure: {2, 4, 1}" />
    <br><em>XOR-analog: activation: ReLU (Rectified Linear Units), structure: {2, 4, 1}</em>
    <br>
        <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-analog-arctan-231.gif"
        title = "XOR-analog: activation: arctan, structure: {2, 3, 1}" />
    <br><em>XOR-analog: activation: arctan, structure: {2, 3, 1}</em>
    <br>
        <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-analog-gaussian-231.gif"
        title = "XOR-analog: activation: gaussian, structure: {2, 3, 1}" />
    <br><em>XOR-analog: activation: gaussian, structure: {2, 3, 1}</em>
    <br>
        <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-analog-mish-231.gif"
        title = "XOR-analog: activation: mish, structure: {2, 3, 1}" />
    <br><em>XOR-analog: activation: mish, structure: {2, 3, 1}</em>
    <br>
        <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-analog-sigmoid-231.gif"
        title = "XOR-analog: activation: sigmoid, structure: {2, 3, 1}" />
    <br><em>XOR-analog: activation: sigmoid, structure: {2, 3, 1}</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-tanh-231-sgd.gif"
        title = "XOR: activation: tanh, structure: {2, 3, 1}, optimizer: SGD" />
    <br><em>XOR: activation: tanh, structure: {2, 3, 1}, optimizer: SGD</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-tanh-231-momentum.gif"
        title = "XOR: activation: tanh, structure: {2, 3, 1}, optimizer: Momentum" />
    <br><em>XOR: activation: tanh, structure: {2, 3, 1}, optimizer: Momentum</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-tanh-231-adagrad.gif"
        title = "XOR: activation: tanh, structure: {2, 3, 1}, optimizer: AdaGrad" />
    <br><em>XOR: activation: tanh, structure: {2, 3, 1}, optimizer: AdaGrad</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-tanh-231-adadelta.gif"
        title = "XOR: activation: tanh, structure: {2, 3, 1}, optimizer: AdaDelta" />
    <br><em>XOR: activation: tanh, structure: {2, 3, 1}, optimizer: AdaDelta</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-tanh-231-rmsprop.gif"
        title = "XOR: activation: tanh, structure: {2, 3, 1}, optimizer: RMSProp" />
    <br><em>XOR: activation: tanh, structure: {2, 3, 1}, optimizer: RMSProp</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-tanh-231-adam.gif"
        title = "XOR: activation: tanh, structure: {2, 3, 1}, optimizer: Adam" />
    <br><em>XOR: activation: tanh, structure: {2, 3, 1}, optimizer: Adam</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-tanh-231-adamax.gif"
        title = "XOR: activation: tanh, structure: {2, 3, 1}, optimizer: AdaMax" />
    <br><em>XOR: activation: tanh, structure: {2, 3, 1}, optimizer: AdaMax</em>
    <br>
    <br><img src = "http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-tanh-231-rprop.gif"
        title = "XOR: activation: tanh, structure: {2, 3, 1}, optimizer: RProp" />
    <br><em>XOR: activation: tanh, structure: {2, 3, 1}, optimizer: RProp</em>
</p>

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



# List of small datasets tested
- The classical XOR test, and also 2 XOR and 3 XOR (to reduce hazard and improve learning stability, see [MLPComparison.xls](MLPComparison.xls));
- What if the XOR logical operation was analogical instead of logical? See this truth table (xor-analog):
```
    | Input  |Output|
    | A   B  |      |
    |--------|------|
    |  1   0 |    1 |
    |  0   0 |    0 |
    |  0   1 |    1 |
    |  1   1 |    0 |
    |0.9 0.1 |  0.9 |
    |0.1 0.1 |  0.1 |
    |0.1 0.9 |  0.9 |
    |0.9 0.9 |  0.1 |
    |0.5 0.5 |  0.5 |
```
- The [Iris flower](https://en.wikipedia.org/wiki/Iris_flower_data_set) test;
- The [Sunspots](https://courses.cs.washington.edu/courses/cse599/01wi/admin/Assignments/nn.html) test.

# List of frameworks and libraries remaining to be tested
- [ML.NET](https://github.com/dotnet/machinelearning) (Microsoft Machine Learning for .NET): See there for an example with XOR and AutoML in C#:
    - https://github.com/PatriceDargenton/machinelearning-samples/tree/main/samples/csharp/getting-started/XOR
	- and in VB:
https://github.com/PatriceDargenton/machinelearning-samples/tree/VB/samples/visualbasic/getting-started/XOR
	- AutoML is very interesting for testing many algorithms using the same data! But in my example, only the FastTree algorithm is working for the moment, I still need to find and tune at least one algorithm with a neural network, and show the weights in a console demo, like for the other libraries.
	- But there are major drawbacks with ML.NET:
		* ML.NET requires the definition of a class representing the object, and must therefore be compiled before running the app;
		* Even using FeatureVector, it is still required to compile before running the app (to set the size of the vector);
		* There is no multi-ouput regression with ML.NET: if you have multiple outputs to learn (e.g. to test 2XOR or 3XOR), then you must learn an array of independent ML.NET networks (instead of an integrated multi-output network, and potentially dependant outputs);
		* A small dataset (e.g. XOR one) needs to be duplicated in order to have enough rows to learn (it would be better to define a number of iterations for learning, like in the other libraries, it may be an AutoML drawback).
- [CNTK](https://github.com/microsoft/CNTK) (Microsoft Cognitive Toolkit)
- [SynapseML](https://github.com/microsoft/SynapseML) from Microsoft (use python)
- [Synapses](https://github.com/mrdimosthenis/Synapses)
- [Vulpes](https://github.com/fsprojects/Vulpes)
- [OpenCVDotNet](https://code.google.com/archive/p/opencvdotnet)
- [Emgu CV](https://emgu.com/wiki/index.php/Main_Page)
- [MxNet.Sharp](https://github.com/deepakkumar1984/MxNet.Sharp)
- [Gym.NET](https://github.com/SciSharp/Gym.NET)
- [Torch.NET](https://github.com/SciSharp/Torch.NET)
- [SciSharp-Learn](https://github.com/SciSharp/scikit-learn.net)
- [SiaNet](https://github.com/SciSharp/SiaNet)
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

What is the recommanded minimum structure with a tensor? See [NeuralNet.NET Framework](#neuralnet-mlp-implementation-using-neuralnetnet-framework)

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

## Bright Wire MLP: implementation using Bright Wire Framework

From https://github.com/jdermody/brightwire .Net core (.Net 5 and .Net 6)

From https://github.com/jdermody/brightwire-v2 .Net 4.6


Packages added:
```
  .Net 4
  <package id="BrightWire.Net4" version="2.1.1" targetFramework="net472" />
  <package id="protobuf-net" version="2.4.0" targetFramework="net472" />
  <package id="System.ValueTuple" version="4.5.0" targetFramework="net472" />
```
```
  .Net 5
  <PackageReference Include="BrightData.Numerics" Version="3.0.2" />
  <PackageReference Include="BrightWire" Version="3.0.2" />
```
```
  .Net 6
  <PackageReference Include="BrightData.Numerics" Version="3.0.3" />
  <PackageReference Include="BrightWire" Version="3.0.3" />
```

Note: The optimization is not complete, as the lack of stability during the training process makes functional tests difficult to perform. Despite the setted weights, there is still a random factor that could not be identified. So there will be no performance comparison data in the [Excel workbook](MLPComparison.xls) at this time.

# MLP comparison

[MLPComparison.xls](MLPComparison.xls)

# Version history

17/12/2021 V2.04
- GetWeight updated, SetWeight added

11/12/2021 V2.03
- [Bright Wire](https://github.com/jdermody/brightwire) MLP added

10/12/2021 V2.02
- Matrix.ToStringWithFormat() : removeNegativeSignFromZero function added

16/10/2021 V2.01
- A single source code, 3 solutions, 3 .NET frameworks (.Net472, .Net5, .Net6)

26/09/2021 V1.45 -> 'Version1' branch
- Test modules renamed

26/09/2021 V1.44
- Packages updated:
```
Packages updated:
----------------
FSharp.Core.5.0.1 -> FSharp.Core.6.0.0
Google.Protobuf.3.15.6 -> Google.Protobuf.3.18.0
JetBrains.Annotations.2020.3.0 -> JetBrains.Annotations.2021.2.0
Newtonsoft.Json.12.0.3 -> Newtonsoft.Json.13.0.1
```
19/09/2021 V1.43
- Tests moved in UnitTestOne-Ring-to-rule-them-all project and Microsoft.VisualStudio.QualityTools.UnitTestFramework.dll link removed

16/08/2021 V1.42
- Tests added with animated gifs: xor-analog-arctan-231, xor-analog-gaussian-231, xor-analog-mish-231 and xor-analog-sigmoid-231
- Information added in animated gifs: targets reached drawn in green

14/07/2021 V1.41
- Analog XOR dataset and tests added: see animated gifs
- [Mish](https://github.com/Sergio0694/NeuralNetwork.NET/issues/93) activation function added (added also for [NeuralNet library](https://github.com/PatriceDargenton/NeuralNetwork.NET/tree/stable))
- Average error computing: for one sample and for all samples
- Signed error computing added
- Animated gifs: first iterations offsets fixed using signed error adjustement around 0 (we can see the differences from previous gifs using the sign _ at the end of the gif name, for example "xor-tanh-221_.gif" there: http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-tanh-221_.gif)
- Animated gifs: information added during iterations: structure of the neural network, weight of neurons, activation function and derivative, output signal and target, history of the error curve
- clsMLPOOP: Randomize range fixed
- clsMLPTensor: ComputeAverageError: moved into clsMLPGeneric function
- clsMLPAccord: averageError computing: moved into clsMLPGeneric function
- clsMLPClassic test: MLP1XORTanh281Gif: neural network structure fixed: {2, 3, 1} -> {2, 8, 1}
- clsMLPClassic test: MLP1XORSinus221Gif: activation function fixed: Gaussian -> Sinus
- clsMLPGeneric: GetMLPType MustOverride function added
- NetworkOOP.MultilayerPerceptron -> clsMLPOOP
- MatrixMLP.MultiLayerPerceptron -> clsMLPMatrix

29/04/2021 V1.40
- Tests added with animated gifs of the learning process
- Double threshold activation function: derivate fixed

15/04/2021 V1.39
- ShowWeights moved into clsMLPGeneric

10/04/2021 V1.38
- clsMLPMatrix: standard weight initialization

01/04/2021 V1.37
- [MLPComparison.xls](MLPComparison.xls) updated
- New Sunspots test added

20/03/2021 V1.36
- trainingAlgorithm moved in clsMLPGeneric

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