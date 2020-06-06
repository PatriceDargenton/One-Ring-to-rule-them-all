
Imports Perceptron.Util ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Imports Microsoft.VisualStudio.TestTools.UnitTesting

Module modMLPTensorTest

    Sub Main()
        Console.WriteLine("Tensor MultiLayerPerceptron with the classical XOR test.")
        TensorMLPTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()
    End Sub

    Public Sub TensorMLPTest()

        Dim mlp As New clsMLPTensor

        mlp.ShowMessage("Tensor MLP test")
        mlp.ShowMessage("---------------")

        mlp.Init(learningRate:=0.1!, weightAdjustment:=0.05!)

        Dim nbIterations%

        mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=0)
        'mlp.SetActivationFunction(TActivationFunction.HyperbolicTangent, gain:=1, center:=0)
        'mlp.SetActivationFunction(TActivationFunction.ELU, gain:=1, center:=0.4)

        nbIterations = 5000

        mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)

        mlp.Randomize()

        mlp.PrintWeights()

        Console.WriteLine()
        Console.WriteLine("Press a key to start.")
        Console.ReadKey()
        Console.WriteLine()

        mlp.targetArray = m_targetArrayXOR

        mlp.printOutput_ = True
        mlp.nbIterations = nbIterations
        mlp.inputArray = m_inputArrayXOR
        mlp.TrainVector() ' Works fine
        'mlp.Train()
        'mlp.Train(enumLearningMode.Systematic) ' Works fine
        'mlp.Train(enumLearningMode.SemiStochastic) ' Works
        'mlp.Train(enumLearningMode.Stochastic) ' Works

        mlp.ShowMessage("Tensor MLP test: Done.")

    End Sub

End Module

Namespace MLPTensorTests

    <TestClass()>
    Public Class clsMLPTensorTest

        Private m_mlp As New clsMLPTensor

        ' Some tests works, and not others?
        'Private m_mlp As New VectorizedMatrixMLP.clsVectorizedMatrixMLP

        'Private m_mlp As New clsMLPClassic ' Not same weight array size
        'Private m_mlp As New MatrixMLP.MultiLayerPerceptron ' InitializeWeights not implemented
        'Private m_mlp As New NetworkOOP.MultilayerPerceptron ' Not same weight array size

        <TestInitialize()>
        Public Sub Init()
        End Sub

        Private Sub InitXOR()
            m_mlp.Init(learningRate:=0.1!, weightAdjustment:=0)
            m_mlp.inputArray = m_inputArrayXOR
            m_mlp.targetArray = m_targetArrayXOR
        End Sub

        Private Sub Init2XOR()
            m_mlp.Init(learningRate:=0.1!, weightAdjustment:=0)
            m_mlp.inputArray = m_inputArray2XOR
            m_mlp.targetArray = m_targetArray2XOR
        End Sub

        Private Sub Init3XOR()
            m_mlp.Init(learningRate:=0.1!, weightAdjustment:=0)
            m_mlp.inputArray = m_inputArray3XOR
            m_mlp.targetArray = m_targetArray3XOR
        End Sub

        <TestMethod()>
        Public Sub TensorMLP1XORSystematic()

            InitXOR()
            m_mlp.learningRate = 0.3
            m_mlp.weightAdjustment = 0.25
            m_mlp.nbIterations = 2000
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=0)

            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.14, 0.13, 0.71},
                {0.16, 0.74, 0.3}})
            m_mlp.InitializeWeights(2, {
                {0.25},
                {0.09},
                {0.56}})
            m_mlp.WeightInitLayerLinear(1, {
                {0.24, 0.74, -0.04, 0.68, -0.72},
                {-0.86, 0.1, -0.66, -0.44, 0.58}})
            m_mlp.WeightInitLayerLinear(2, {
                {-0.12, 0.06, 0.72, -0.48, -0.66},
                {0.88, -0.1, 0.92, -0.96, 0.06},
                {0.1, -0.16, -0.36, 0.62, 0.36},
                {-0.56, 0.42, 0.78, -0.96, 0.22},
                {-0.78, -0.68, -0.86, -0.82, -0.66}})
            m_mlp.WeightInitLayerLinear(3, {
                {0.18},
                {-1.0},
                {0.66},
                {-0.7},
                {0.56}})
            m_mlp.InitializeSequential()

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR

            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim expectedLoss# = 0.03
            'Dim loss! = m_mlp.ComputeAverageError() ' Vector learning mode only
            Dim loss! = m_mlp.ComputeAverageErrorFromAllSamples!()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(lossRounded <= expectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub TensorMLP1XORSemiStochastic()

            InitXOR()
            m_mlp.learningRate = 0.3
            m_mlp.weightAdjustment = 0.25
            m_mlp.nbIterations = 2000
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=0)

            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.14, 0.13, 0.71},
                {0.16, 0.74, 0.3}})
            m_mlp.InitializeWeights(2, {
                {0.25},
                {0.09},
                {0.56}})
            m_mlp.WeightInitLayerLinear(1, {
                {0.24, 0.74, -0.04, 0.68, -0.72},
                {-0.86, 0.1, -0.66, -0.44, 0.58}})
            m_mlp.WeightInitLayerLinear(2, {
                {-0.12, 0.06, 0.72, -0.48, -0.66},
                {0.88, -0.1, 0.92, -0.96, 0.06},
                {0.1, -0.16, -0.36, 0.62, 0.36},
                {-0.56, 0.42, 0.78, -0.96, 0.22},
                {-0.78, -0.68, -0.86, -0.82, -0.66}})
            m_mlp.WeightInitLayerLinear(3, {
                {0.18},
                {-1.0},
                {0.66},
                {-0.7},
                {0.56}})
            m_mlp.InitializeSequential()

            m_mlp.Train(enumLearningMode.SemiStochastic)

            Dim expectedOutput = m_targetArrayXOR

            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim expectedLoss# = 0.03
            'Dim loss! = m_mlp.ComputeAverageError() ' Vector learning mode only
            Dim loss! = m_mlp.ComputeAverageErrorFromAllSamples!()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(lossRounded <= expectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub TensorMLP1XORStochastic()

            InitXOR()
            m_mlp.learningRate = 0.3
            m_mlp.weightAdjustment = 0.25
            m_mlp.nbIterations = 20000
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=0)

            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.14, 0.13, 0.71},
                {0.16, 0.74, 0.3}})
            m_mlp.InitializeWeights(2, {
                {0.25},
                {0.09},
                {0.56}})
            m_mlp.WeightInitLayerLinear(1, {
                {0.24, 0.74, -0.04, 0.68, -0.72},
                {-0.86, 0.1, -0.66, -0.44, 0.58}})
            m_mlp.WeightInitLayerLinear(2, {
                {-0.12, 0.06, 0.72, -0.48, -0.66},
                {0.88, -0.1, 0.92, -0.96, 0.06},
                {0.1, -0.16, -0.36, 0.62, 0.36},
                {-0.56, 0.42, 0.78, -0.96, 0.22},
                {-0.78, -0.68, -0.86, -0.82, -0.66}})
            m_mlp.WeightInitLayerLinear(3, {
                {0.18},
                {-1.0},
                {0.66},
                {-0.7},
                {0.56}})
            m_mlp.InitializeSequential()

            m_mlp.Train(enumLearningMode.Stochastic)

            Dim expectedOutput = m_targetArrayXOR

            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim expectedLoss# = 0.02
            'Dim loss! = m_mlp.ComputeAverageError() ' Vector learning mode only
            Dim loss! = m_mlp.ComputeAverageErrorFromAllSamples!()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(lossRounded <= expectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub TensorMLP1XORSigmoid()

            InitXOR()
            m_mlp.learningRate = 0.1
            m_mlp.weightAdjustment = 0.2
            m_mlp.nbIterations = 5000 ' Sigmoid: works
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=0)

            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.88, 0.8, 0.36},
                {0.99, 0.53, 0.46}})
            m_mlp.InitializeWeights(2, {
                {0.41},
                {0.8},
                {0.12}})
            m_mlp.WeightInitLayerLinear(1, {
                {-0.12, -0.78, -0.84, -0.46, -0.02},
                {-0.46, 0.4, 0.16, 0.84, 0.74}})
            m_mlp.WeightInitLayerLinear(2, {
                {0.86, -0.14, -0.48, 0.58, -0.16},
                {0.2, -1.0, 0.5, -0.32, 0.86},
                {-0.84, 0.28, -0.22, -0.64, 0.72},
                {0.76, -0.5, -0.44, 0.3, 0.36},
                {0.86, -0.82, -0.68, -0.34, -0.52}})
            m_mlp.WeightInitLayerLinear(3, {
                {-0.86},
                {-0.02},
                {-0.42},
                {0.42},
                {-0.84}})
            m_mlp.InitializeSequential()

            m_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            'm_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR

            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim expectedLoss# = 0.01
            Dim loss! = m_mlp.ComputeAverageError()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(lossRounded <= expectedLoss, True)

        End Sub

        ' addBiasColumn:=False is not implemented
        '<TestMethod()>
        Public Sub TensorMLP1XORSigmoidWithoutBias()

            m_mlp.Init(learningRate:=0.1!, weightAdjustment:=0)
            InitXOR()
            m_mlp.nbIterations = 5000
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=0)

            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=False)

            m_mlp.InitializeWeights(1, {
                {0.97, 0.31},
                {0.03, 0.73}})
            m_mlp.InitializeWeights(2, {
                {0.0},
                {0.99}})
            m_mlp.WeightInitLayerLinear(1, {
                {0.44, -0.04, -0.82, 1.0},
                {0.94, 0.22, 0.34, 0.66}}, addBias:=False)
            m_mlp.WeightInitLayerLinear(2, {
                {-1.0, 0.98, -0.58, 0.24},
                {-0.68, 0.54, -0.36, -0.36},
                {0.9, 1.0, -0.14, -0.8},
                {0.08, -0.82, -0.18, -0.16}}, addBias:=False)
            m_mlp.WeightInitLayerLinear(3, {
                {0.18},
                {-0.48},
                {-0.64},
                {0.14}}, addBias:=False)
            m_mlp.InitializeSequential()

            m_mlp.TrainVector()
            'm_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR

            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim outputMaxtrix As Matrix = m_mlp.outputArraySingle
            Dim sOutput = outputMaxtrix.ToStringWithFormat(dec:="0.0")
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim expectedLoss# = 0
            Dim loss! = m_mlp.ComputeAverageError()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(lossRounded <= expectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub TensorMLP2XORSigmoid()

            Init2XOR()
            m_mlp.learningRate = 0.1
            m_mlp.weightAdjustment = 0.2
            m_mlp.nbIterations = 900 ' Sigmoid: works
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=0)

            m_mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.2, 0.77, 0.31, 0.42, 0.24},
                {0.8, 0.45, 0.86, 0.78, 0.45},
                {0.5, 0.21, 0.02, 0.13, 0.76},
                {0.54, 0.08, 0.2, 0.58, 0.94}})
            m_mlp.InitializeWeights(2, {
                {0.07, 0.15},
                {0.76, 0.86},
                {0.79, 0.9},
                {0.29, 0.22},
                {0.88, 0.54}})

            m_mlp.WeightInitLayerLinear(1, {
                {-0.72, 0.14, -0.24, 0.62, 0.28, -0.26, 0.82, 0.8, -0.68},
                {0.4, -0.12, -0.44, -0.84, 0.02, 0.46, -0.14, -0.16, -0.68},
                {0.1, -0.32, 0.02, 0.14, 0.0, -0.06, 0.36, -0.88, -0.96},
                {-0.12, 0.26, -0.32, 0.02, -0.54, 0.96, -0.5, -0.38, 0.86}})
            m_mlp.WeightInitLayerLinear(2, {
                {0.64, -0.6, -0.04, -0.2, 0.26, -0.64, -0.92, 0.76, 0.0},
                {0.58, -0.86, -0.5, -0.74, -0.28, -0.04, -0.74, 0.34, 0.08},
                {-0.94, 0.12, -0.86, 0.82, -0.74, 0.18, -0.3, -0.06, -0.46},
                {0.64, -0.62, -0.58, -0.4, 0.66, -0.6, 0.62, 0.2, -0.56},
                {0.34, -0.7, 0.42, -0.36, 0.06, 0.24, 0.94, 0.78, 0.98},
                {0.54, -0.38, -0.46, 0.72, -0.24, 0.1, -0.12, -0.56, 0.5},
                {0.7, 0.82, -0.86, 0.8, -0.92, -0.68, 0.82, -0.56, 0.38},
                {-0.44, -0.02, -0.54, -0.9, -0.34, 0.52, -0.48, -0.08, 0.04},
                {0.66, 0.4, -0.94, -0.1, 0.88, -0.52, 0.18, 0.16, -0.68}})
            m_mlp.WeightInitLayerLinear(3, {
                {-1.0, 0.92},
                {0.64, 0.32},
                {0.7, 0.24},
                {-0.12, 0.9},
                {0.38, -0.72},
                {0.54, -0.78},
                {0.1, -0.18},
                {-0.38, 0.88},
                {0.38, -0.2}})
            m_mlp.InitializeSequential()

            m_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            'm_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR

            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim expectedLoss# = 0.02
            Dim loss! = m_mlp.ComputeAverageError()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(lossRounded <= expectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub TensorMLP2XORHTangent()

            Init2XOR()
            m_mlp.learningRate = 0.05
            m_mlp.weightAdjustment = 0.07
            m_mlp.nbIterations = 3000 ' 400 : Train, 3000 : TrainVector
            m_mlp.SetActivationFunction(TActivationFunction.HyperbolicTangent, gain:=1, center:=-0.05)

            m_mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.67, 0.1, 0.27, 0.37, 0.78},
                {0.47, 0.84, 0.99, 0.49, 0.41},
                {0.73, 0.98, 0.67, 0.63, 0.41},
                {0.63, 0.09, 0.98, 0.7, 0.01}})
            m_mlp.InitializeWeights(2, {
                {0.57, 0.97},
                {0.87, 0.13},
                {0.52, 0.44},
                {0.47, 0.72},
                {0.33, 0.01}})

            m_mlp.WeightInitLayerLinear(1, {
                {-0.88, 0.9, 0.02, 0.68, -0.96, -0.04, -0.94, 0.38, 0.36},
                {-0.72, 0.32, -0.54, 0.48, -0.88, -0.28, -0.42, -0.02, 0.92},
                {0.86, 0.76, 0.96, 0.1, -0.48, 0.62, 0.86, 0.4, -0.54},
                {-0.74, 0.7, -0.32, 0.98, -0.76, 0.32, -0.04, -0.32, 0.56}})
            m_mlp.WeightInitLayerLinear(2, {
                {-0.04, 0.64, 0.22, 0.84, 0.18, -0.22, -0.4, 0.12, -0.3},
                {-0.32, -0.54, 0.62, 0.54, -0.52, 0.88, 0.04, 0.56, -0.1},
                {-0.96, 0.04, 0.38, 0.4, 0.84, -0.36, -0.5, 0.8, 0.7},
                {-0.34, -0.72, 0.08, 0.14, -0.5, 0.44, 0.16, 0.62, 0.34},
                {-0.32, -0.98, -0.44, 0.18, -0.52, 0.4, -0.1, 0.18, -0.06},
                {-0.16, -0.28, 0.22, -0.2, -0.04, -0.32, -0.6, -0.08, 0.66},
                {-0.82, 0.56, 0.82, -0.4, 0.34, 0.4, 0.1, 0.94, -0.16},
                {0.62, 0.54, 0.96, -0.82, -0.6, -0.14, -0.46, -0.66, 0.54},
                {-0.66, -0.14, -0.46, 1.0, -0.52, -0.34, 0.7, 0.66, 0.08}})
            m_mlp.WeightInitLayerLinear(3, {
                {-0.52, 0.86},
                {0.34, -0.6},
                {-0.26, 0.58},
                {0.78, -0.04},
                {-0.94, 0.54},
                {-0.9, -0.32},
                {0.18, -0.92},
                {-0.46, -0.44},
                {0.28, 0.64}})
            m_mlp.InitializeSequential()

            m_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            'm_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR

            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim expectedLoss# = 0.01
            Dim loss! = m_mlp.ComputeAverageError()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(lossRounded <= expectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub TensorMLP2XORELU()

            Init2XOR()
            m_mlp.learningRate = 0.1
            m_mlp.weightAdjustment = 0.05
            m_mlp.nbIterations = 200
            m_mlp.SetActivationFunction(TActivationFunction.ELU, gain:=1, center:=0.4)

            m_mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.24, 0.98, 0.23, 0.77, 0.16},
                {0.65, 0.11, 0.53, 0.42, 0.84},
                {0.22, 0.2, 0.51, 0.48, 0.8},
                {0.04, 0.27, 0.5, 0.42, 0.73}})
            m_mlp.InitializeWeights(2, {
                {0.19, 0.65},
                {1.0, 0.35},
                {0.88, 0.73},
                {0.89, 0.16},
                {0.15, 0.52}})

            m_mlp.WeightInitLayerLinear(1, {
                {-0.06, -0.78, 0.04, -0.36, 0.76, -0.7, 0.4, 0.06, 0.5},
                {1.0, -0.66, 0.2, 0.06, -0.52, -0.08, -0.56, 0.28, 0.86},
                {-0.1, -0.28, -0.66, -0.72, 0.24, -0.44, 0.34, 0.2, 0.96},
                {0.76, 0.8, -0.14, 0.52, 0.9, -0.26, 0.8, -0.26, -0.78}})
            m_mlp.WeightInitLayerLinear(2, {
                {0.34, -0.62, 0.22, 0.3, -0.32, 0.5, 0.5, -0.16, 0.12},
                {0.16, -0.76, 0.52, 0.78, 0.32, -0.82, 0.92, 0.44, 0.58},
                {-0.3, -0.34, -0.02, -0.52, 0.3, -0.44, -0.68, 0.62, 0.26},
                {-0.34, -0.54, -0.54, -0.54, 0.26, 0.74, -0.3, 0.1, -0.1},
                {-0.36, 0.58, -0.96, -0.14, -0.22, -0.6, 0.46, -0.82, -0.04},
                {-0.56, 0.98, -0.52, -0.32, 0.6, -0.54, 0.16, 0.12, -0.92},
                {0.24, -0.14, 0.08, -0.34, -0.02, 0.06, -0.78, -0.16, -0.62},
                {-0.34, -0.3, -0.02, 0.78, 0.08, -0.78, 0.3, 0.28, 0.84},
                {0.56, 0.84, 0.88, -0.42, 0.02, 0.12, 0.6, 0.88, 0.64}})
            m_mlp.WeightInitLayerLinear(3, {
                {-0.22, 0.98},
                {-0.14, 1.0},
                {0.3, -0.86},
                {0.66, 0.46},
                {-0.78, 0.8},
                {0.98, -0.38},
                {-0.02, -0.36},
                {0.94, -0.98},
                {-0.22, 0.48}})
            m_mlp.InitializeSequential()

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR

            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim expectedLoss# = 0
            Dim loss! = m_mlp.ComputeAverageError()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(lossRounded <= expectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub TensorMLP3XORSigmoid()

            Init3XOR()
            m_mlp.learningRate = 0.1
            m_mlp.weightAdjustment = 0.2
            m_mlp.nbIterations = 400 ' Sigmoid: works
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=0)

            m_mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.68, 0.24, 0.14, 0.76, 0.92, 0.79, 0.99},
                {0.82, 0.35, 0.84, 0.04, 0.03, 0.82, 0.04},
                {0.31, 0.53, 0.82, 0.99, 0.07, 0.27, 0.28},
                {0.69, 0.99, 0.74, 0.21, 0.6, 0.12, 0.48},
                {0.28, 0.09, 0.43, 0.88, 0.47, 0.94, 0.64},
                {0.22, 0.87, 0.11, 0.44, 0.81, 0.67, 0.57}})
            m_mlp.InitializeWeights(2, {
                {0.85, 0.91, 0.37},
                {0.19, 0.43, 0.32},
                {0.02, 0.03, 0.81},
                {0.09, 0.47, 0.21},
                {0.55, 0.99, 0.25},
                {0.4, 0.55, 0.32},
                {0.67, 0.51, 0.54}})

            m_mlp.WeightInitLayerLinear(1, {
                {-0.48, -0.18, -0.68, 0.14, 0.76, -0.2, -0.82, 0.32, 0.44, 0.1, -0.48, 0.22, 0.42},
                {0.68, -0.86, -0.26, -0.96, -0.64, 0.6, -0.08, -0.5, -0.44, -0.32, -0.16, -0.5, -0.24},
                {0.3, 0.92, -0.06, 0.12, -0.76, -0.72, -0.68, -0.94, 0.18, 0.0, -0.56, 0.24, 0.08},
                {0.84, 0.86, -0.26, -0.82, 0.08, -0.58, 0.22, 0.68, -0.7, -0.64, -0.94, 0.06, -0.72},
                {0.74, -0.9, -0.4, 0.96, -0.86, 0.48, -0.38, 0.0, 0.48, -0.74, -0.62, -0.68, -0.14},
                {-0.76, -0.1, 0.36, -0.5, 0.14, -0.7, -0.2, 0.26, 0.74, 0.06, 0.76, -0.62, 0.62}})
            m_mlp.WeightInitLayerLinear(2, {
                {-0.58, 0.3, 0.08, 0.0, 0.56, -0.12, -0.96, 0.96, -0.46, -0.78, 0.46, 0.22, -0.12},
                {-0.06, -0.38, -0.9, -0.64, 0.6, -0.64, 0.86, -0.78, -0.82, -0.68, -0.68, 0.8, 0.22},
                {0.76, -0.74, 0.02, 1.0, 0.04, -0.16, 0.58, -0.48, 0.06, 0.34, 0.92, -0.52, -0.3},
                {0.5, -0.72, -0.1, 0.7, -0.32, 0.88, 0.28, -0.72, 0.36, -0.82, 0.18, 0.38, 0.46},
                {0.4, -0.48, 0.4, -0.76, -0.02, -0.24, 0.2, -0.66, 0.14, 0.78, -0.06, -0.46, 0.16},
                {-0.38, 0.64, -0.64, 0.88, 0.3, -0.82, 0.88, -0.1, -0.14, 0.6, 0.32, -0.54, 0.64},
                {-0.56, -0.48, -0.06, -0.6, -0.92, 0.84, -0.38, 0.58, 0.44, 0.06, 0.12, -0.18, -0.64},
                {0.16, 0.28, -0.64, -0.64, -0.52, 0.94, 0.18, 0.52, 0.26, 0.64, 0.92, 0.48, -0.12},
                {0.0, 0.5, -0.44, -0.46, -0.08, -0.92, 0.78, 0.34, -0.68, -0.32, 0.4, -0.26, 0.7},
                {0.12, 0.92, 0.58, 0.18, -0.42, 0.24, 0.06, -0.06, 0.02, -0.4, -0.44, -0.52, 0.1},
                {0.36, 0.3, -0.88, 0.18, -0.14, 0.02, -0.08, 0.2, -0.16, 0.12, 0.04, -0.08, -0.88},
                {0.02, 0.04, 0.02, -0.16, -0.4, -0.06, 0.64, -0.22, -0.16, 0.24, 0.94, -0.92, 0.06},
                {-0.32, -0.6, 0.96, 0.98, -0.1, 1.0, -0.96, 0.82, -0.28, -0.52, 0.22, 0.14, 0.82}})
            m_mlp.WeightInitLayerLinear(3, {
                {-0.62, 0.82, -0.48},
                {-0.72, -0.18, -0.46},
                {-0.74, -0.54, -0.78},
                {-0.98, 0.92, -0.84},
                {-1.0, 0.6, -0.14},
                {-0.66, -0.46, -0.96},
                {-0.82, 0.1, 0.8},
                {-0.64, -0.5, 0.24},
                {-0.14, -0.8, 0.04},
                {-0.14, -0.7, -0.68},
                {-0.44, 0.8, 0.72},
                {-0.22, -0.42, 0.82},
                {-0.74, -0.44, -0.56}})
            m_mlp.InitializeSequential()

            m_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            'm_mlp.Train()

            Dim expectedOutput = m_targetArray3XOR

            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim expectedLoss# = 0.03
            Dim loss! = m_mlp.ComputeAverageError()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(lossRounded <= expectedLoss, True)

        End Sub

        ' Slow: 15 sec.
        '<TestMethod()>
        Public Sub TensorMLP3XORELU()

            Init3XOR()
            m_mlp.learningRate = 0.1
            m_mlp.weightAdjustment = 0.05
            m_mlp.nbIterations = 700
            m_mlp.SetActivationFunction(TActivationFunction.ELU, gain:=1, center:=2)

            m_mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.17, 0.01, 0.86, 0.99, 0.24, 0.42, 0.57},
                {0.29, 0.35, 0.79, 0.48, 0.23, 0.37, 0.2},
                {0.73, 0.44, 0.75, 0.43, 0.66, 0.71, 0.89},
                {0.64, 0.48, 0.91, 0.27, 0.31, 0.44, 0.53},
                {0.07, 0.85, 0.26, 0.88, 0.02, 0.72, 0.48},
                {0.32, 0.65, 0.9, 0.96, 0.87, 0.5, 0.84}})
            m_mlp.InitializeWeights(2, {
                {0.45, 0.56, 0.62},
                {0.13, 0.4, 0.05},
                {0.37, 0.11, 0.67},
                {0.79, 0.6, 0.82},
                {0.61, 0.53, 0.53},
                {0.95, 0.72, 0.92},
                {0.98, 0.04, 0.22}})

            m_mlp.WeightInitLayerLinear(1, {
                {0.00, 0.06, 0.2, -0.58, 0.3, 0.44, -0.18, 0.58, 0.7, -0.06, 0.56, -0.58, -0.9},
                {-0.62, 0.84, -0.42, -0.72, 0.84, -0.22, -0.66, 0.92, -0.66, -0.06, -0.44, -0.6, -0.78},
                {0.9, 0.58, 0.4, -0.64, -0.92, 0.78, -0.08, 0.22, 0.9, -0.92, -0.96, 0.82, 0.5},
                {-0.34, 0.92, -0.36, 0.64, -0.34, 0.64, 0.2, 0.12, -0.32, 0.24, 0.32, 0.16, 0.18},
                {0.74, -0.48, -0.34, -0.88, -0.36, -0.98, 0.08, 0.54, 0.24, -0.84, 0.34, -0.14, 0.78},
                {0.5, -0.14, -0.52, 0.76, -0.46, -0.54, -0.66, -0.86, -0.58, 0.28, -0.3, -0.7, -0.08}})
            m_mlp.WeightInitLayerLinear(2, {
                {0.2, 0.08, 0.22, -0.66, -0.9, 0.22, -0.94, -0.4, 0.18, -0.3, 0.22, 0.96, 0.4},
                {0.8, -0.6, 0.1, -0.32, -0.52, -0.68, 0.78, 0.84, 0.7, -0.74, -0.8, 0.48, -0.22},
                {-0.3, 0.68, 0.04, 0.32, 0.24, 0.96, 0.82, 0.74, -0.18, -0.66, -0.1, -0.94, -0.88},
                {-0.18, 0.18, -0.68, -0.84, 0.26, 0.8, 0.14, -0.34, 0.94, 0.02, 0.74, -0.2, -0.62},
                {-0.5, 0.06, 0.5, -0.18, 0.02, -0.14, 0.32, -0.48, -0.62, 0.56, 0.86, 0.46, 0.26},
                {-0.86, 0.66, -0.02, -0.6, -0.22, 0.78, -0.58, -0.8, -0.02, -0.34, 0.38, -0.9, -0.06},
                {0.68, 0.64, -0.96, 0.74, 0.58, -0.54, 0.6, 0.32, -0.32, 0.34, 0.18, 0.32, -0.96},
                {-0.24, 0.6, -0.56, 0.64, -0.54, -0.28, 1.0, 0.66, 0.48, 0.68, 0.54, -0.76, 0.96},
                {-0.62, -0.82, 0.54, -0.6, -0.9, -0.26, -0.94, 0.18, 0.68, -0.52, -0.36, 0.54, -0.72},
                {0.00, 0.66, -0.18, 0.00, 0.64, 0.24, 0.46, 0.74, 0.66, -0.4, -0.46, 0.02, -0.06},
                {0.38, -0.04, -0.84, -0.66, -0.8, 0.76, 0.62, -0.94, 0.42, 0.78, -0.74, 0.22, -0.56},
                {0.26, -0.16, 0.08, 0.12, 0.78, -0.9, -0.82, 0.72, -0.68, -0.16, -0.52, -0.96, -0.7},
                {-0.22, -0.78, -0.28, 0.58, 0.00, 0.38, 0.16, 0.34, 0.12, 0.18, 0.52, -0.86, -0.56}})
            m_mlp.WeightInitLayerLinear(3, {
                {0.78, -0.36, -0.06},
                {-0.74, 0.04, -0.28},
                {-0.58, -0.22, -0.8},
                {-0.1, -0.42, 0.48},
                {-0.24, -0.06, -0.24},
                {0.68, -0.38, 0.18},
                {0.68, -0.7, 0.9},
                {-0.02, 0.42, 0.34},
                {0.82, -0.22, -0.1},
                {-0.94, 0.1, -0.86},
                {0.66, 0.56, -0.02},
                {0.34, -0.68, 0.52},
                {-0.22, -0.94, 0.8}})
            m_mlp.InitializeSequential()

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = m_targetArray3XOR

            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim expectedLoss# = 0
            Dim loss! = m_mlp.ComputeAverageError()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(lossRounded <= expectedLoss, True)

        End Sub

    End Class

End Namespace