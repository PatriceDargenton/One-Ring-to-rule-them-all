
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPClassic ' enumLearningMode

Imports Microsoft.VisualStudio.TestTools.UnitTesting

Module modMLPClassicTest

    Sub Main()
        Console.WriteLine("MultiLayerPerceptron with the classical XOR test.")
        ClassicMLPXorTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()
    End Sub

    Public Sub ClassicMLPXorTest(Optional nbXor% = 1)

        Dim mlp As New clsMLPClassic

        mlp.ShowMessage("Classic MLP Xor test")
        mlp.ShowMessage("--------------------")

        mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

        Dim nbIterations%

        'mlp.SetActivationFunction(enumActivationFunction.Sigmoid)
        'nbIterations = 10000 ' Sigmoid: works

        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent)
        nbIterations = 2000 ' Hyperbolic tangent: works fine

        'mlp.SetActivationFunction(enumActivationFunction.Gaussian)
        'nbIterations = 1000 ' Gaussian: works fine

        'mlp.SetActivationFunction(enumActivationFunction.Sinus)
        'nbIterations = 1000 ' Sinus: works fine

        'mlp.SetActivationFunction(enumActivationFunction.ArcTangent)
        'nbIterations = 1000 ' ArcTangent: works fine

        'mlp.SetActivationFunction(enumActivationFunction.ELU)
        'nbIterations = 2000 ' ELU: works

        'mlp.SetActivationFunction(enumActivationFunction.ReLu, gain:=0.9)
        'nbIterations = 1000 ' ReLU: works fine

        'mlp.SetActivationFunction(enumActivationFunction.ReLuSigmoid)
        'nbIterations = 10000 ' ReLUSigmoid: works?

        'mlp.SetActivationFunction(enumActivationFunction.DoubleThreshold)
        'nbIterations = 10000 ' DoubleThreshold: works fine

        mlp.printOutput_ = True
        mlp.printOutputMatrix = False
        mlp.nbIterations = nbIterations

        If nbXor = 1 Then
            mlp.inputArray = m_inputArrayXOR
            mlp.targetArray = m_targetArrayXOR
            mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR4Layers2331, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR5Layers23331, addBiasColumn:=True)
            mlp.printOutputMatrix = True
            mlp.nbIterations = 4000
        ElseIf nbXor = 2 Then
            mlp.inputArray = m_inputArray2XOR
            mlp.targetArray = m_targetArray2XOR
            mlp.InitializeStruct(m_neuronCount2XOR462, addBiasColumn:=True)
        ElseIf nbXor = 3 Then
            mlp.inputArray = m_inputArray3XOR
            mlp.targetArray = m_targetArray3XOR
            mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
        End If

        mlp.Randomize()
        mlp.PrintWeights()

        WaitForKeyToStart()

        'mlp.InitWeights(1, {
        '     {0.28, 0.28, 0.76},
        '     {0.25, 0.88, 0.62}})
        'mlp.InitWeights(2, {
        '     {0.56, 0.92, 0.19}})

        mlp.Train()
        'mlp.Train(enumLearningMode.SemiStochastic)
        'mlp.Train(enumLearningMode.Stochastic)

        mlp.ShowMessage("Classic MLP Xor test: Done.")

        If nbXor > 1 Then Exit Sub

        WaitForKeyToContinue("Press a key to print MLP weights")
        mlp.PrintWeights()

    End Sub

End Module

Namespace ClassicMLP

    <TestClass()>
    Public Class MultiLayerPerceptronTest

        Private m_mlp As New clsMLPClassic

        ' Weights are quite the same as MLP Classic, but not exactly:
        'Private m_mlp As New NetworkOOP.MultilayerPerceptron ' 24 success, 17 fails
        'Private m_mlp As New clsMLPAccord ' 17 success, 24 fails
        'Private m_mlp As New clsMLPEncog  ' 9 success, 32 fails
        'Private m_mlp As New MatrixMLP.MultiLayerPerceptron ' 9 success, 32 fails
        'Private m_mlp As New clsMLPRProp  ' 7 success, 34 fails
        'Private m_mlp As New clsMLPNeuralNet ' 3 success, 38 fails
        'Private m_mlp As New clsMLPTensorFlow ' 2 success, 39 fails

        ' Weights are not stored in the same way:
        'Private m_mlp As New VectorizedMatrixMLP.clsVectorizedMatrixMLP ' 24/24 fails
        'Private m_mlp As New clsMLPTensor ' 24/24 fails

        <TestInitialize()>
        Public Sub Init()
        End Sub

        Private Sub InitXOR()
            m_mlp.inputArray = m_inputArrayXOR
            m_mlp.targetArray = m_targetArrayXOR
            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
        End Sub

        Private Sub Init2XOR()
            m_mlp.inputArray = m_inputArray2XOR
            m_mlp.targetArray = m_targetArray2XOR
            m_mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)
        End Sub

        Private Sub Init3XOR()
            m_mlp.inputArray = m_inputArray3XOR
            m_mlp.targetArray = m_targetArray3XOR
            m_mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
        End Sub

        <TestMethod()>
        Public Sub MLP1XORSigmoidStdr()

            TestMLP1XOR(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLP1XOR4Layers()

            TestMLP1XOR4Layers(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLP1XOR5Layers()

            TestMLP1XOR5Layers(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLP1XORSemiStochastic()

            TestMLP1XORSemiStochastic(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLP1XORStochastic()

            TestMLP1XORStochastic(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLP1XORSigmoid()

            TestMLP1XORSigmoid(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLP1XORSigmoidWithoutBias()

            TestMLP1XORWithoutBias(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLP1XORSigmoidWithoutBias231()

            TestMLP1XORWithoutBias231(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLP1XORTanh()

            TestMLP1XORTanh(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLP1XORTanh261()

            TestMLP1XORTanh261(m_mlp, nbIterations:=600)

        End Sub

        <TestMethod()>
        Public Sub MLP1XORELU()

            TestMLP1XORELU(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLP1XORReLU()

            InitXOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 400
            m_mlp.SetActivationFunction(enumActivationFunction.ReLu, gain:=0.9)

            m_mlp.InitializeWeights(1, {
                {0.07, 0.79, 0.94},
                {0.33, 0.33, 0.93}})
            m_mlp.InitializeWeights(2, {
                {0.63, 0.79, 0.69}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.04
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MLP1XORReLUSigmoid()

            InitXOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 9000
            m_mlp.SetActivationFunction(enumActivationFunction.ReLuSigmoid)

            m_mlp.InitializeWeights(1, {
                {0.58, 0.23, 0.25},
                {0.88, 0.78, 0.18}})
            m_mlp.InitializeWeights(2, {
                {0.29, 0.34, 0.81}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MLP1XORDbleThreshold()

            InitXOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 5000
            m_mlp.SetActivationFunction(enumActivationFunction.DoubleThreshold)

            m_mlp.InitializeWeights(1, {
                {0.86, 0.37, 0.8},
                {1.0, 0.62, 0.17}})
            m_mlp.InitializeWeights(2, {
                {0.54, 0.83, 0.41}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MLP2XORSigmoid()

            TestMLP2XORSigmoid(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLP2XORTanh()

            TestMLP2XORTanh(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLP2XORTanh2()

            TestMLP2XORTanh2(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLP2XORTanh462()

            TestMLP2XORTanh462(m_mlp, nbIterations:=5000)

        End Sub

        <TestMethod()>
        Public Sub MLP2XORELU()

            Init2XOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.05!)

            m_mlp.nbIterations = 200
            m_mlp.SetActivationFunction(enumActivationFunction.ELU, center:=0.4!)

            m_mlp.InitializeWeights(1, {
                {0.15, 0.46, 0.49, 0.28, 0.68},
                {0.65, 0.02, 0.47, 0.47, 0.23},
                {0.53, 0.99, 0.03, 0.9, 0.66},
                {0.06, 0.42, 0.97, 0.91, 0.84}})
            m_mlp.InitializeWeights(2, {
                {0.29, 0.41, 0.17, 0.86, 0.85},
                {0.35, 0.38, 0.79, 0.57, 0.85}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MLP2XORDbleThreshold()

            Init2XOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 3000
            m_mlp.SetActivationFunction(enumActivationFunction.DoubleThreshold)

            m_mlp.InitializeWeights(1, {
                {0.43, 0.21, 0.16, 0.85, 0.2},
                {0.25, 0.5, 0.87, 0.5, 0.36},
                {0.21, 0.78, 0.68, 0.47, 0.35},
                {0.42, 0.09, 0.25, 0.87, 0.17}})
            m_mlp.InitializeWeights(2, {
                {0.88, 0.37, 0.12, 0.17, 0.79},
                {0.71, 0.88, 0.7, 0.83, 0.02}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MLP2XORReLU()

            Init2XOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 200
            m_mlp.SetActivationFunction(enumActivationFunction.ReLu, gain:=0.6, center:=0.4)

            m_mlp.InitializeWeights(1, {
                {0.97, 0.65, 0.44, 0.61, 0.73},
                {0.08, 0.65, 0.37, 0.13, 0.18},
                {0.23, 0.2, 0.64, 0.44, 0.27},
                {0.33, 0.46, 0.99, 0.49, 0.09}})
            m_mlp.InitializeWeights(2, {
                {0.02, 0.36, 0.11, 0.94, 0.82},
                {0.34, 0.92, 0.18, 1.0, 0.73}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MLP3XORSigmoid()

            TestMLP3XORSigmoid(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLP3XORSigmoid2()

            TestMLP3XORSigmoid2(m_mlp, nbIterations:=3000)

        End Sub

        <TestMethod()>
        Public Sub MLP3XORTanh()

            TestMLP3XORTanh(m_mlp, nbIterations:=1100)

        End Sub

        <TestMethod()>
        Public Sub MLP3XORTanh2()

            TestMLP3XORTanh2(m_mlp, nbIterations:=10000)

        End Sub

        <TestMethod()>
        Public Sub MLP3XORELU()

            Init3XOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.05!)

            m_mlp.nbIterations = 200
            m_mlp.SetActivationFunction(enumActivationFunction.ELU, center:=0.4!)

            m_mlp.InitializeWeights(1, {
                {0.26, 0.39, 0.79, 0.77, 0.48, 0.65, 0.88},
                {0.17, 0.49, 0.15, 0.72, 0.78, 0.83, 0.21},
                {0.3, 0.55, 0.73, 0.1, 0.29, 0.7, 0.92},
                {0.9, 0.36, 0.54, 0.85, 0.62, 0.77, 0.66},
                {0.95, 0.34, 0.61, 0.03, 0.91, 0.72, 0.85},
                {0.57, 0.7, 0.9, 0.73, 0.31, 0.77, 0.23}})
            m_mlp.InitializeWeights(2, {
                {0.02, 0.03, 0.31, 0.66, 0.32, 0.53, 0.3},
                {0.38, 0.24, 0.74, 0.17, 0.37, 0.33, 0.48},
                {0.42, 0.23, 0.63, 0.03, 0.11, 0.51, 0.54}})
            m_mlp.Train()

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MLP3XORELUStdr() ' 53 msec

            TestMLP3XORELU(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLP3XORGaussianStdr()

            TestMLP3XORGaussian(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLP3XORSinusStdr()

            TestMLP3XORSinus(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLP3XORDbleThreshold()

            Init3XOR()
            m_mlp.Initialize(learningRate:=2.0!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 100
            m_mlp.SetActivationFunction(enumActivationFunction.DoubleThreshold, center:=2)

            m_mlp.InitializeWeights(1, {
                {0.12, 0.3, 0.12, 0.28, 0.12, 0.2, 0.32},
                {0.05, 0.6, 0.24, 0.41, 0.75, 0.57, 0.76},
                {0.51, 0.62, 0.39, 0.7, 0.29, 0.53, 0.28},
                {0.51, 0.43, 0.45, 0.4, 0.36, 0.56, 0.74},
                {0.77, 0.82, 0.95, 0.06, 0.84, 0.71, 0.27},
                {0.66, 0.29, 0.85, 0.32, 0.92, 0.48, 0.29}})
            m_mlp.InitializeWeights(2, {
                {0.71, 0.99, 0.73, 0.06, 0.95, 0.55, 0.57},
                {0.38, 0.5, 0.37, 0.85, 0.78, 1.0, 0.61},
                {0.87, 0.67, 0.87, 0.76, 0.64, 0.59, 0.27}})
            m_mlp.Train()

            Dim sOutput = m_mlp.output.ToString() 'WithFormat(dec:="0.0")

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString() 'WithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MLP3XORReLU()

            Init3XOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.2!)

            m_mlp.nbIterations = 350
            m_mlp.SetActivationFunction(enumActivationFunction.ReLu, gain:=0.5!, center:=0.1!)

            m_mlp.InitializeWeights(1, {
                {0.13, 0.26, 0.18, 0.05, 0.07, 0.46, 0.96},
                {0.28, 0.31, 0.28, 1.0, 0.47, 0.32, 0.11},
                {0.73, 0.87, 0.17, 0.22, 0.82, 0.97, 0.7},
                {0.27, 0.57, 0.66, 0.53, 0.56, 0.1, 0.64},
                {0.9, 0.92, 0.94, 0.29, 0.86, 0.83, 0.35},
                {0.62, 0.15, 0.53, 0.86, 0.89, 0.16, 0.57}})
            m_mlp.InitializeWeights(2, {
                {0.47, 0.85, 0.17, 0.55, 0.45, 0.81, 0.49},
                {0.29, 0.03, 0.95, 0.51, 0.46, 0.85, 0.7},
                {0.52, 0.52, 0.51, 0.36, 0.96, 0.65, 0.41}})
            m_mlp.Train()

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MLPIrisFlowerAnalogTanh()

            ' 96.7% prediction, 96.7% learning with 200 iterations in 180 msec.

            TestMLPIrisFlowerAnalogTanh(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPIrisFlowerAnalogSigmoid()

            ' 96.7% prediction, 95.8% learning with 150 iterations in 150 msec.

            TestMLPIrisFlowerAnalogSigmoid(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPIrisFlowerAnalogGaussian()

            ' 93.3% prediction, 96.7% learning with 100 iterations in 77 msec.

            TestMLPIrisFlowerAnalogGaussian(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPIrisFlowerLogicalGaussian()

            ' 96.7% prediction, 96.4% learning with 100 iterations in 90 msec.

            TestMLPIrisFlowerLogicalGaussian(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPIrisFlowerLogicalSinus()

            ' 97.8% prediction, 93.3% learning with 200 iterations in 190 msec.

            TestMLPIrisFlowerLogicalSinus(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPIrisFlowerLogicalTanh()

            ' 97.8% prediction, 99.4% learning with 400 iterations in 915 msec.

            TestMLPIrisFlowerLogicalTanh(m_mlp, nbIterations:=400)

        End Sub

        <TestMethod()>
        Public Sub MLPIrisFlowerLogicalSigmoid()

            ' 97.8% prediction, 98.9% learning with 900 iterations in 1.2 sec.

            TestMLPIrisFlowerLogicalSigmoid(m_mlp) ', nbIterations:=900)

        End Sub

        <TestMethod()>
        Public Sub MLPSunspot1Sigmoid()

            ' 90.0% prediction, 73.5% learning with 400 iterations in 120 msec.

            TestMLPSunspot1Sigmoid(m_mlp, nbIterations:=400, expectedSuccess:=0.735,
                expectedSuccessPrediction:=0.9, expectedLoss:=0.07)

        End Sub

        <TestMethod()>
        Public Sub MLPSunspot1Tanh()

            ' 90.0% prediction, 75% learning with 200 iterations in 60 msec.

            TestMLPSunspot1Tanh(m_mlp, expectedSuccess:=0.75, expectedSuccessPrediction:=0.9)

        End Sub

        <TestMethod()>
        Public Sub MLPSunspot2Tanh()

            ' 93.4% prediction, 93.1% learning with 100 iterations in 57 msec.

            TestMLPSunspotTanh2(m_mlp)

        End Sub

    End Class

End Namespace