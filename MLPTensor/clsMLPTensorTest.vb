
Imports Perceptron.DLFramework ' Tensor
Imports Perceptron.DLFramework.Layers ' Linear, Sequential
Imports Perceptron.DLFramework.Layers.Loss ' MeanSquaredError
Imports Perceptron.DLFramework.Optimizers ' StochasticGradientDescent

Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Imports Microsoft.VisualStudio.TestTools.UnitTesting

Module modMLPTensorTest

    Sub Main()
        'SimpleTest()
        Console.WriteLine("Tensor MultiLayerPerceptron with the classical XOR test.")
        TensorMLPXorTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()
    End Sub

    Public Sub SimpleTest()

        Dim r As New Random

        ' Works with an easy test:
        'Dim sourceDble As Double(,) = {{0, 0}, {0, 1}, {1, 0}, {1, 1}}
        'Dim targetDble As Double(,) = {{0}, {1}, {0}, {1}}
        ' Does not work with the classical and difficult XOR test:
        Dim sourceDble As Double(,) = {{1, 0}, {0, 0}, {0, 1}, {1, 1}}
        Dim targetDble As Double(,) = {{1}, {0}, {1}, {0}}

        Dim sourceMatrix As Matrix = sourceDble
        Dim data As New Tensor(sourceMatrix, autoGrad:=True)

        Dim targetMatrix As Matrix = targetDble
        Dim target As New Tensor(targetMatrix, autoGrad:=True)

        Dim seq As New Sequential()
        seq.Layers.Add(New Linear(2, 3, r, addBias:=True))
        seq.Layers.Add(New Linear(3, 1, r, addBias:=True))

        Dim sgd As New StochasticGradientDescent(seq.Parameters, 0.1!)

        Dim mse As New MeanSquaredError()

        For i = 0 To 20
            Dim pred = seq.Forward(data)
            Dim loss = mse.Forward(pred, target)
            loss.Backward(New Tensor(Matrix.Ones(loss.Data.c, loss.Data.r)))
            sgd.Step_(zero:=True)
            Debug.WriteLine("Epoch: {" & i & "} Loss: { " & loss.ToString() & "}")
        Next

    End Sub

    Public Sub TensorMLPXorTest(Optional nbXor% = 1)

        Dim mlp As New clsMLPTensor

        mlp.ShowMessage("Tensor MLP Xor test")
        mlp.ShowMessage("-------------------")

        mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.05!)

        Dim nbIterations%

        mlp.SetActivationFunctionOptimized(
            enumActivationFunctionOptimized.Sigmoid)
        'mlp.SetActivationFunctionOptimized(
        '    enumActivationFunctionOptimized.HyperbolicTangent, gain:=2)

        ' Works only for 1 XOR, and 2 XOR in not vectorized learning mode:
        'mlp.SetActivationFunctionOptimized(
        '    enumActivationFunctionOptimized.ELU) ', center:=0.4)

        nbIterations = 2000

        mlp.printOutput_ = True
        mlp.printOutputMatrix = False
        mlp.nbIterations = nbIterations

        If nbXor = 1 Then
            mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR4Layers2331, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR5Layers23331, addBiasColumn:=True)
            mlp.printOutputMatrix = True
            mlp.inputArray = m_inputArrayXOR
            mlp.targetArray = m_targetArrayXOR
        ElseIf nbXor = 2 Then
            mlp.inputArray = m_inputArray2XOR
            mlp.targetArray = m_targetArray2XOR
            'mlp.InitializeStruct(m_neuronCount2XOR462, addBiasColumn:=True)
            mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)
        ElseIf nbXor = 3 Then
            mlp.inputArray = m_inputArray3XOR
            mlp.targetArray = m_targetArray3XOR
            mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
        End If

        mlp.Randomize()

        mlp.PrintWeights()

        WaitForKeyToStart()

        mlp.TrainVector() ' Works fine
        'mlp.Train()
        'mlp.Train(enumLearningMode.Systematic) ' Works fine
        'mlp.Train(enumLearningMode.SemiStochastic) ' Works
        'mlp.Train(enumLearningMode.Stochastic) ' Works

        mlp.ShowMessage("Tensor MLP Xor test: Done.")

        If nbXor > 1 Then Exit Sub

        WaitForKeyToContinue("Press a key to print MLP weights")
        mlp.PrintWeights()

    End Sub

End Module

Namespace TensorMLP

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
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.inputArray = m_inputArrayXOR
            m_mlp.targetArray = m_targetArrayXOR
        End Sub

        Private Sub Init2XOR()
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.inputArray = m_inputArray2XOR
            m_mlp.targetArray = m_targetArray2XOR
        End Sub

        Private Sub Init3XOR()
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.inputArray = m_inputArray3XOR
            m_mlp.targetArray = m_targetArray3XOR
        End Sub

        <TestMethod()>
        Public Sub TensorMLP1XORSystematic()

            InitXOR()
            m_mlp.learningRate = 0.3
            m_mlp.weightAdjustment = 0.25
            m_mlp.nbIterations = 2000
            m_mlp.SetActivationFunctionOptimized(
                enumActivationFunctionOptimized.Sigmoid)

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
            m_mlp.InitializeGradient()

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.03
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub TensorMLP1XORSemiStochastic()

            InitXOR()
            m_mlp.learningRate = 0.3
            m_mlp.weightAdjustment = 0.25
            m_mlp.nbIterations = 2000
            m_mlp.SetActivationFunctionOptimized(
                enumActivationFunctionOptimized.Sigmoid)

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
            m_mlp.InitializeGradient()

            m_mlp.Train(enumLearningMode.SemiStochastic)

            Dim expectedOutput = m_targetArrayXOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.03
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub TensorMLP1XORStochastic()

            InitXOR()
            m_mlp.learningRate = 0.3
            m_mlp.weightAdjustment = 0.25
            m_mlp.nbIterations = 20000
            m_mlp.SetActivationFunctionOptimized(
                enumActivationFunctionOptimized.Sigmoid)

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
            m_mlp.InitializeGradient()

            m_mlp.Train(enumLearningMode.Stochastic)

            Dim expectedOutput = m_targetArrayXOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub TensorMLP1XORSigmoid()

            InitXOR()
            m_mlp.learningRate = 0.1
            m_mlp.weightAdjustment = 0.2
            m_mlp.nbIterations = 7000 '5000 ' Sigmoid: works
            m_mlp.SetActivationFunctionOptimized(
                enumActivationFunctionOptimized.Sigmoid)

            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)

            ' 2 x 3
            m_mlp.InitializeWeights(1, {
                {0.88, 0.8, 0.36},
                {0.99, 0.53, 0.46}})
            ' 3 x 1
            m_mlp.InitializeWeights(2, {
                {0.41},
                {0.8},
                {0.12}})
            ' 2 x 5
            m_mlp.WeightInitLayerLinear(1, {
                {-0.12, -0.78, -0.84, -0.46, -0.02},
                {-0.46, 0.4, 0.16, 0.84, 0.74}})
            ' 5 x 5
            m_mlp.WeightInitLayerLinear(2, {
                {0.86, -0.14, -0.48, 0.58, -0.16},
                {0.2, -1.0, 0.5, -0.32, 0.86},
                {-0.84, 0.28, -0.22, -0.64, 0.72},
                {0.76, -0.5, -0.44, 0.3, 0.36},
                {0.86, -0.82, -0.68, -0.34, -0.52}})
            ' 5 x 1
            m_mlp.WeightInitLayerLinear(3, {
                {-0.86},
                {-0.02},
                {-0.42},
                {0.42},
                {-0.84}})
            m_mlp.InitializeGradient()

            'm_mlp.PrintWeights()
            'm_mlp.printOutput_ = True
            'm_mlp.printOutputMatrix = True

            m_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            'm_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.03
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub TensorMLP1XORSigmoid4Layers()

            InitXOR()
            m_mlp.learningRate = 0.1
            m_mlp.weightAdjustment = 0.2
            m_mlp.nbIterations = 4000 ' Sigmoid: works
            m_mlp.SetActivationFunctionOptimized(
                enumActivationFunctionOptimized.Sigmoid)

            m_mlp.InitializeStruct({2, 2, 2, 1}, addBiasColumn:=True)

            ' 2 x 3
            m_mlp.InitializeWeights(1, {
                {0.2, 0.94, 0.59},
                {0.66, 0.9, 0.98}})
            ' 3 x 3
            m_mlp.InitializeWeights(2, {
                {0.06, 0.54, 0.97},
                {0.48, 0.48, 0.86},
                {0.82, 0.45, 0.08}})
            ' 3 x 1
            m_mlp.InitializeWeights(3, {
                {0.12},
                {0.45},
                {0.29}})
            ' 2 x 5
            m_mlp.WeightInitLayerLinear(1, {
                {0.7, -0.5, 0.26, -0.78, -0.2},
                {-0.14, 0.9, 0.04, -0.8, 0.22}})
            ' 5 x 5
            m_mlp.WeightInitLayerLinear(2, {
                {-0.12, -0.2, -0.66, 0.06, -0.58},
                {0.2, 0.58, -0.92, 0.26, 0.6},
                {-0.08, 0.76, -0.32, 0.34, -0.06},
                {-0.92, 0.0, 0.92, -0.92, -0.28},
                {0.48, -0.64, -0.86, 0.06, -0.2}})
            ' 5 x 5
            m_mlp.WeightInitLayerLinear(3, {
                {-0.54, 0.62, -0.84, 0.08, -0.68},
                {0.42, -0.24, 0.76, -0.08, -0.8},
                {0.14, -0.38, 0.9, 0.3, -0.56},
                {0.32, -0.92, -0.02, 0.3, 0.66},
                {0.92, 0.84, 0.92, 0.28, -0.28}})
            ' 5 x 1
            m_mlp.WeightInitLayerLinear(4, {
                {0.86},
                {0.98},
                {-0.04},
                {0.48},
                {0.74}})
            m_mlp.InitializeGradient()

            'm_mlp.PrintWeights()
            'm_mlp.printOutput_ = True

            m_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            'm_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.04
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        ' addBiasColumn:=False is not implemented
        '<TestMethod()>
        Public Sub TensorMLP1XORSigmoidWithoutBias()

            InitXOR()
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.nbIterations = 5000
            m_mlp.SetActivationFunctionOptimized(
                enumActivationFunctionOptimized.Sigmoid)

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
            m_mlp.InitializeGradient()

            m_mlp.TrainVector()
            'm_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR

            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub TensorMLP2XORSigmoid()

            Init2XOR()
            m_mlp.learningRate = 0.1
            m_mlp.weightAdjustment = 0.2
            m_mlp.nbIterations = 900 ' Sigmoid: works
            m_mlp.SetActivationFunctionOptimized(
                enumActivationFunctionOptimized.Sigmoid)

            m_mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)

            ' 4 x 5
            m_mlp.InitializeWeights(1, {
                {0.2, 0.77, 0.31, 0.42, 0.24},
                {0.8, 0.45, 0.86, 0.78, 0.45},
                {0.5, 0.21, 0.02, 0.13, 0.76},
                {0.54, 0.08, 0.2, 0.58, 0.94}})
            ' 5 x 2
            m_mlp.InitializeWeights(2, {
                {0.07, 0.15},
                {0.76, 0.86},
                {0.79, 0.9},
                {0.29, 0.22},
                {0.88, 0.54}})

            ' 4 x 9 (9 = 4 + 5 = nbInputs + nbHiddens, nbHiddens = nbInputs + 1 bias)
            m_mlp.WeightInitLayerLinear(1, {
                {-0.72, 0.14, -0.24, 0.62, 0.28, -0.26, 0.82, 0.8, -0.68},
                {0.4, -0.12, -0.44, -0.84, 0.02, 0.46, -0.14, -0.16, -0.68},
                {0.1, -0.32, 0.02, 0.14, 0.0, -0.06, 0.36, -0.88, -0.96},
                {-0.12, 0.26, -0.32, 0.02, -0.54, 0.96, -0.5, -0.38, 0.86}})
            ' 9 x 9
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
            ' 9 x 2
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
            m_mlp.InitializeGradient()

            'm_mlp.PrintWeights()
            'm_mlp.printOutput_ = True

            m_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            'm_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.03
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub TensorMLP2XORHTangent()

            Init2XOR()
            m_mlp.learningRate = 0.05
            m_mlp.weightAdjustment = 0.07
            m_mlp.nbIterations = 3000 ' 400 : Train, 3000 : TrainVector
            m_mlp.SetActivationFunctionOptimized(
                enumActivationFunctionOptimized.HyperbolicTangent, center:=-0.05)

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
            m_mlp.InitializeGradient()

            m_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            'm_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub TensorMLP2XORELU()

            Init2XOR()
            m_mlp.learningRate = 0.1
            m_mlp.weightAdjustment = 0.05
            m_mlp.nbIterations = 200
            m_mlp.SetActivationFunctionOptimized(
                enumActivationFunctionOptimized.ELU, center:=0.4)

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
            m_mlp.InitializeGradient()

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub TensorMLP3XORSigmoid()

            Init3XOR()
            m_mlp.learningRate = 0.1
            m_mlp.weightAdjustment = 0.2
            m_mlp.nbIterations = 700 ' Sigmoid: works
            m_mlp.SetActivationFunctionOptimized(
                enumActivationFunctionOptimized.Sigmoid)

            m_mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)

            ' 6 x 7
            m_mlp.InitializeWeights(1, {
                {0.68, 0.24, 0.14, 0.76, 0.92, 0.79, 0.99},
                {0.82, 0.35, 0.84, 0.04, 0.03, 0.82, 0.04},
                {0.31, 0.53, 0.82, 0.99, 0.07, 0.27, 0.28},
                {0.69, 0.99, 0.74, 0.21, 0.6, 0.12, 0.48},
                {0.28, 0.09, 0.43, 0.88, 0.47, 0.94, 0.64},
                {0.22, 0.87, 0.11, 0.44, 0.81, 0.67, 0.57}})
            ' 7 x 3
            m_mlp.InitializeWeights(2, {
                {0.85, 0.91, 0.37},
                {0.19, 0.43, 0.32},
                {0.02, 0.03, 0.81},
                {0.09, 0.47, 0.21},
                {0.55, 0.99, 0.25},
                {0.4, 0.55, 0.32},
                {0.67, 0.51, 0.54}})

            ' 6 x 13 (13 = 6 + 7)
            m_mlp.WeightInitLayerLinear(1, {
                {-0.48, -0.18, -0.68, 0.14, 0.76, -0.2, -0.82, 0.32, 0.44, 0.1, -0.48, 0.22, 0.42},
                {0.68, -0.86, -0.26, -0.96, -0.64, 0.6, -0.08, -0.5, -0.44, -0.32, -0.16, -0.5, -0.24},
                {0.3, 0.92, -0.06, 0.12, -0.76, -0.72, -0.68, -0.94, 0.18, 0.0, -0.56, 0.24, 0.08},
                {0.84, 0.86, -0.26, -0.82, 0.08, -0.58, 0.22, 0.68, -0.7, -0.64, -0.94, 0.06, -0.72},
                {0.74, -0.9, -0.4, 0.96, -0.86, 0.48, -0.38, 0.0, 0.48, -0.74, -0.62, -0.68, -0.14},
                {-0.76, -0.1, 0.36, -0.5, 0.14, -0.7, -0.2, 0.26, 0.74, 0.06, 0.76, -0.62, 0.62}})
            ' 13 x 13
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
            ' 13 x 3
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
            m_mlp.InitializeGradient()

            m_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            'm_mlp.Train()

            Dim expectedOutput = m_targetArray3XOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        ' Slow: 15 sec.
        '<TestMethod()>
        Public Sub TensorMLP3XORELU()

            Init3XOR()
            m_mlp.learningRate = 0.1
            m_mlp.weightAdjustment = 0.05
            m_mlp.nbIterations = 700
            m_mlp.SetActivationFunctionOptimized(
                enumActivationFunctionOptimized.ELU, center:=2)

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
                {0.0, 0.06, 0.2, -0.58, 0.3, 0.44, -0.18, 0.58, 0.7, -0.06, 0.56, -0.58, -0.9},
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
                {0.0, 0.66, -0.18, 0.0, 0.64, 0.24, 0.46, 0.74, 0.66, -0.4, -0.46, 0.02, -0.06},
                {0.38, -0.04, -0.84, -0.66, -0.8, 0.76, 0.62, -0.94, 0.42, 0.78, -0.74, 0.22, -0.56},
                {0.26, -0.16, 0.08, 0.12, 0.78, -0.9, -0.82, 0.72, -0.68, -0.16, -0.52, -0.96, -0.7},
                {-0.22, -0.78, -0.28, 0.58, 0.0, 0.38, 0.16, 0.34, 0.12, 0.18, 0.52, -0.86, -0.56}})
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
            m_mlp.InitializeGradient()

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = m_targetArray3XOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub TensorMLPIrisFlowerLogicalPrediction()

            ' 97.8% prediction, 98.3% learning with 200 iterations in 1.5 sec.

            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)
            'InitIrisFlowerLogicalPrediction(m_mlp)
            m_mlp.inputArray = m_inputArrayIrisFlowerTrain
            m_mlp.targetArray = m_targetArrayIrisFlowerLogicalTrain

            ' Set activation function before InitializeStruct
            m_mlp.SetActivationFunctionOptimized(
                enumActivationFunctionOptimized.Sigmoid)
            m_mlp.InitializeStruct({4, 4, 4, 3}, addBiasColumn:=True)

            'm_mlp.minimalSuccessTreshold = 0.05
            'm_mlp.nbIterations = 900

            m_mlp.nbIterations = 200
            m_mlp.minimalSuccessTreshold = 0.3

            ' 4 x 5: First layer: 4 inputs x (4 hidden neurons + bias):
            m_mlp.InitializeWeights(1, {
                {-0.08, 0.47, -0.1, -0.07, 0.43},
                {0.15, -0.27, 0.23, -0.42, -0.17},
                {-0.19, 0.15, 0.35, -0.21, -0.15},
                {0.09, -0.48, 0.31, -0.11, 0.04}})
            ' 5 x 5: Second layer: (4 hidden neurons + bias) x (4 hidden neurons + bias):
            m_mlp.InitializeWeights(2, {
                {-0.41, 0.48, -0.32, 0.13, -0.02},
                {0.21, 0.29, -0.13, -0.34, -0.22},
                {-0.01, 0.34, -0.07, 0.18, 0.45},
                {-0.32, 0.13, 0.17, -0.45, -0.27},
                {-0.01, 0.22, 0.45, 0.26, 0.1}})
            ' 5 x 3: Third layer: (4 hidden neurons + bias) x 3 ouputs:
            m_mlp.InitializeWeights(3, {
                {0.25, 0.13, -0.44},
                {-0.12, -0.15, -0.38},
                {0.23, -0.38, 0.08},
                {0.31, -0.05, 0.29},
                {0.27, 0.45, -0.28}})

            ' 4 x 9: First linear layer: 4 inputs x (4 inputs + 4 hidden neurons + bias):
            m_mlp.WeightInitLayerLinear(1, {
                {-0.28, -0.3, -0.88, -0.4, -0.32, -1.08, -1.58, -1.66, -1.32},
                {-1.64, -0.06, -1.3, -0.48, -1.68, -1.9, -1.26, -1.94, -1.16},
                {-1.94, -0.54, -1.84, -0.54, -0.02, -0.38, -1.66, -0.48, -0.56},
                {-0.3, -0.26, -0.98, -1.22, -0.28, -0.56, -0.32, -1.26, -0.72}})
            ' 9 x 9: Second linear layer: (4 inputs + 4 hidden neurons + bias) x (4 inputs + 4 hidden neurons + bias):
            m_mlp.WeightInitLayerLinear(2, {
                {-1.68, -1.7, -0.16, -1.72, -0.92, -1.08, -0.56, -1.58, -1.24},
                {-0.48, -0.06, -0.06, -0.94, -1.12, -1.16, -1.28, -0.52, -1.58},
                {-1.72, -1.74, -0.28, -0.5, -0.74, -1.84, -0.52, -1.28, -1.4},
                {-0.34, -0.44, -1.78, -0.74, -0.14, -0.42, -1.18, -1.58, -0.26},
                {-1.0, -0.2, -1.62, -0.76, -1.96, -0.44, -1.14, -1.18, -0.42},
                {-0.5, -1.36, -1.14, -1.82, -1.94, -1.76, -0.98, -0.6, -1.52},
                {-0.44, -1.2, -0.96, -0.32, -1.22, -1.64, -1.68, -0.22, -1.16},
                {-1.46, -1.74, -1.92, -1.64, -1.76, -1.54, -0.9, -0.28, -0.32},
                {-1.96, -0.96, -1.78, -1.84, -1.36, -1.54, -1.42, -0.02, -1.92}})
            ' 9 x 9: Third linear layer: (4 inputs + 4 hidden neurons + bias) x (4 inputs + 4 hidden neurons + bias):
            m_mlp.WeightInitLayerLinear(3, {
                {-0.26, -0.52, -0.48, -0.02, -1.76, -1.54, -0.9, -0.72, -0.38},
                {-1.28, -0.66, -0.98, -1.96, -1.08, -1.74, -1.28, -1.68, -1.44},
                {-0.5, -0.84, -1.6, -1.62, -0.92, -1.66, -1.44, -1.02, -1.64},
                {-1.74, -0.6, -1.84, -1.42, -0.92, -1.2, -1.74, -1.42, -1.72},
                {-0.68, -1.42, -1.98, -0.38, -0.76, -1.04, -1.14, -1.62, -1.62},
                {-1.32, -2.0, -1.88, -0.04, -0.56, -1.66, -0.1, -0.92, -1.18},
                {-0.32, -0.64, -1.6, -0.82, -0.58, -0.72, -1.92, -1.16, -0.12},
                {-0.56, -1.86, -1.76, -1.8, -0.24, -1.66, -0.02, -0.6, -0.26},
                {-1.46, -0.12, -0.1, -0.56, -0.46, -1.3, -0.06, -0.12, -1.04}})
            ' 9 x 3: Fourth linear layer: (4 inputs + 4 hidden neurons + bias) x 3 outputs:
            m_mlp.WeightInitLayerLinear(4, {
                {-1.74, -1.7, -0.04},
                {-0.16, -1.32, 0.0},
                {-0.02, -1.42, -0.78},
                {-0.12, -1.86, -0.84},
                {-1.26, -0.46, -1.6},
                {-0.92, -0.58, -1.76},
                {-1.86, -1.54, -1.76},
                {-0.22, -0.02, -1.96},
                {-1.4, -0.64, -0.78}})
            m_mlp.InitializeGradient()

            'm_mlp.PrintWeights()
            'm_mlp.printOutput_ = True

            m_mlp.Train()

            Dim expectedSuccess# = 0.983
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.06
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            If m_mlp.successPC = 1 AndAlso m_mlp.minimalSuccessTreshold <= 0.05! Then
                Dim expectedOutput = m_targetArrayIrisFlowerLogicalTrain
                Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
                Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
                Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
                Assert.AreEqual(sExpectedOutput, sOutput)
            End If

            m_mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
                m_targetArrayIrisFlowerLogicalTest, nbOutputs:=3)
            Dim expectedSuccessPrediction# = 0.978
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

    End Class

End Namespace