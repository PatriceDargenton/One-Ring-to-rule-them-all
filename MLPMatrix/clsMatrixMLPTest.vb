﻿
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPClassic ' enumLearningMode

Imports Microsoft.VisualStudio.TestTools.UnitTesting

Namespace MatrixMLP

    <TestClass()>
    Public Class MultiLayerPerceptronTest

        Private m_mlp As New MultiLayerPerceptron

        ' Some functions are not implemented
        'Private m_mlp As New NetworkOOP.MultilayerPerceptron
        'Private m_mlp As New clsMLPClassic 
        'Private m_mlp As New VectorizedMatrixMLP.clsVectorizedMatrixMLP 

        <TestInitialize()>
        Public Sub Init()
        End Sub

        Private Sub InitXOR()
            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
            m_mlp.inputArray = m_inputArrayXOR
            m_mlp.targetArray = m_targetArrayXOR
        End Sub

        Private Sub Init2XOR()
            m_mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)
            m_mlp.inputArray = m_inputArray2XOR
            m_mlp.targetArray = m_targetArray2XOR
        End Sub

        Private Sub Init3XOR()
            m_mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
            m_mlp.inputArray = m_inputArray3XOR
            m_mlp.targetArray = m_targetArray3XOR
        End Sub

        <TestMethod()>
        Public Sub MatrixMLP1XORSigmoid()

            Dim nbIterations% = 10000
            m_mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.Sigmoid, center:=2.2)

            InitXOR()
            m_mlp.Initialize(learningRate:=0.9)

            m_mlp.weights_ih = {
                {0.95, 0.82},
                {0.78, 0.27}}
            m_mlp.weights_ho = {
                {0.77, 0.27}}
            m_mlp.bias_h = {
                {0.91},
                {0.9}}
            m_mlp.bias_o = {
                {0.01}}

            m_mlp.nbIterations = nbIterations
            m_mlp.Train()

            Dim expectedOutput = New Single(,) {
                {0.99},
                {0.01},
                {0.99},
                {0.01}}

            Dim sOutput = m_mlp.output.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MatrixMLP1XORSigmoidWithoutBias()

            Dim nbIterations% = 100000
            m_mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.Sigmoid)

            InitXOR()
            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=False)
            m_mlp.Initialize(learningRate:=0.1)

            m_mlp.weights_ih = {
                {0.76, 0.81},
                {0.09, 0.16}}
            m_mlp.weights_ho = {
                {0.17, 0.17}}

            m_mlp.nbIterations = nbIterations
            m_mlp.Train()

            Dim expectedOutput = New Single(,) {
                {0.92},
                {0.01},
                {0.92},
                {0.06}}

            Dim sOutput = m_mlp.output.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.06
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MatrixMLP1XORHyperbolicTangent()

            Dim nbIterations% = 5000
            'm_mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.HyperbolicTangent)
            m_mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.HyperbolicTangent,
                gain:=2)

            InitXOR()
            m_mlp.Initialize(learningRate:=0.05)

            m_mlp.weights_ih = {
                {0.79, 0.81},
                {0.41, 0.84}}
            m_mlp.weights_ho = {
                {0.87, 0.11}}
            m_mlp.bias_h = {
                {0.31},
                {0.81}}
            m_mlp.bias_o = {
                {0.6}}

            m_mlp.nbIterations = nbIterations
            m_mlp.Train()

            Dim expectedOutput = New Single(,) {
                {0.97},
                {0},
                {0.97},
                {0}}

            Dim sOutput = m_mlp.output.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MatrixMLP1XORELU()

            Dim nbIterations% = 300
            m_mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.ELU, center:=-1.8)

            InitXOR()
            m_mlp.Initialize(learningRate:=0.1)

            m_mlp.weights_ih = {
                {0.57, 0.6},
                {0.34, 0.14}}
            m_mlp.weights_ho = {
                {0.8, 0.46}}
            m_mlp.bias_h = {
                {0.24},
                {0.29}}
            m_mlp.bias_o = {
                {0.27}}

            m_mlp.nbIterations = nbIterations
            m_mlp.Train()

            Dim expectedOutput = New Single(,) {
                {1},
                {0},
                {1},
                {0}}

            Dim sOutput = m_mlp.output.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MatrixMLP2XORSigmoid()

            m_mlp.nbIterations = 3000
            m_mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.Sigmoid, gain:=2)

            Init2XOR()
            m_mlp.Initialize(learningRate:=0.05)

            m_mlp.weights_ih = {
                {-0.11, 0.07, -0.05, 0.03},
                {-0.41, 0.15, 0.38, -0.49},
                {-0.36, 0.38, -0.33, -0.23},
                {-0.36, 0.22, -0.22, 0.28}}
            m_mlp.weights_ho = {
                {0.42, -0.25, 0.06, 0.11},
                {0.34, -0.08, 0.01, 0.4}}
            m_mlp.bias_h = {
                {0.11},
                {0.08},
                {-0.13},
                {0.13}}
            m_mlp.bias_o = {
                {-0.1},
                {-0.23}}

            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.03
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MatrixMLP2XORHyperbolicTangent()

            Dim nbIterations% = 5000
            'm_mlp.SetActivationFunctionOptimized(
            '   enumActivationFunctionOptimized.HyperbolicTangent, center:=0.5)
            m_mlp.SetActivationFunctionOptimized(
                enumActivationFunctionOptimized.HyperbolicTangent, gain:=2, center:=0.5)

            Init2XOR()
            m_mlp.Initialize(learningRate:=0.1)

            m_mlp.weights_ih = {
                {0.39, 0.5, 0.3, 0.25},
                {0.31, 0.07, 0.53, 0.15},
                {0.84, 0.47, 0.91, 0.86},
                {0.88, 0.13, 0.34, 0.81}}
            m_mlp.weights_ho = {
                {0.58, 0.76, 0.12, 0.45},
                {0.8, 0.01, 0.67, 0.75}}
            m_mlp.bias_h = {
                {0.74},
                {0.8},
                {0.05},
                {0.37}}
            m_mlp.bias_o = {
                {0.66},
                {0.42}}

            m_mlp.nbIterations = nbIterations
            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MatrixMLP3XORSigmoid()

            m_mlp.nbIterations = 900
            m_mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.Sigmoid, gain:=2)

            Init3XOR()
            m_mlp.InitializeStruct(m_neuronCount3XOR673, addBiasColumn:=True)
            m_mlp.Initialize(learningRate:=0.05)

            m_mlp.weights_ih = {
                {0.49, -0.26, 0.08, -0.47, -0.34, -0.01},
                {-0.21, -0.34, 0.41, 0.21, 0.47, 0.1},
                {-0.44, -0.38, -0.06, -0.5, 0.14, 0.08},
                {-0.22, -0.19, 0.33, -0.37, 0.19, 0.1},
                {0.43, -0.37, 0.32, -0.02, 0.27, -0.31},
                {-0.45, 0.43, 0.25, 0.41, -0.07, -0.04},
                {0.08, 0.36, -0.27, -0.06, -0.19, 0.27}}
            m_mlp.weights_ho = {
                {-0.32, 0.12, -0.07, 0.46, 0.09, -0.24, 0.04},
                {-0.04, -0.2, -0.3, 0.42, 0.18, -0.09, -0.05},
                {-0.15, 0.06, 0.08, -0.14, -0.03, 0.18, -0.37}}
            m_mlp.bias_h = {
                {0.02},
                {0.42},
                {0.17},
                {0.26},
                {-0.36},
                {0.19},
                {0.48}}
            m_mlp.bias_o = {
                {-0.08},
                {0.29},
                {-0.15}}

            m_mlp.Train()

            Dim expectedOutput = m_targetArray3XOR

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MatrixMLP3XORHyperbolicTangent()

            m_mlp.nbIterations = 800
            m_mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.HyperbolicTangent, gain:=2)

            Init3XOR()
            m_mlp.InitializeStruct(m_neuronCount3XOR673, addBiasColumn:=True)
            m_mlp.Initialize(learningRate:=0.02)

            m_mlp.weights_ih = {
                {-0.25, 0.05, -0.27, -0.09, -0.17, -0.31},
                {0.3, 0.08, -0.19, 0.02, 0.42, 0.1},
                {0.13, -0.32, -0.05, 0.3, 0.46, -0.24},
                {0.08, 0.46, -0.25, 0.25, 0.19, -0.21},
                {0.07, -0.37, 0.23, -0.03, -0.23, -0.39},
                {-0.38, -0.28, -0.34, -0.35, 0.36, 0.07},
                {-0.37, 0.06, 0.02, -0.15, 0.03, -0.35}}
            m_mlp.weights_ho = {
                {-0.24, -0.17, -0.44, 0.33, 0.14, 0.19, -0.18},
                {0.04, 0.37, 0.15, 0.42, 0.48, -0.45, 0.04},
                {-0.38, 0.26, -0.44, -0.43, -0.39, -0.45, -0.09}}
            m_mlp.bias_h = {
                {0.29},
                {-0.12},
                {-0.03},
                {-0.48},
                {0.09},
                {-0.27},
                {0.33}}
            m_mlp.bias_o = {
                {0.2},
                {-0.44},
                {0.11}}

            m_mlp.Train()

            Dim expectedOutput = m_targetArray3XOR

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MatrixMLP3XORELU()

            m_mlp.nbIterations = 1100
            m_mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.ELU)

            Init3XOR()
            m_mlp.InitializeStruct(m_neuronCount3XOR673, addBiasColumn:=True)
            m_mlp.Initialize(learningRate:=0.01)

            m_mlp.weights_ih = {
                {-0.47, 0.11, 0.47, 0.25, -0.09, 0.27},
                {0.24, 0.46, 0.21, 0.02, 0.1, 0.44},
                {-0.05, -0.26, -0.14, 0.42, 0.31, -0.45},
                {-0.07, -0.14, -0.42, -0.33, -0.02, -0.23},
                {0.19, -0.24, -0.04, -0.19, -0.38, 0.41},
                {-0.07, -0.31, 0.45, 0.46, 0.37, -0.45},
                {0.35, 0.46, 0.18, 0.34, 0.2, -0.32}}
            m_mlp.weights_ho = {
                {-0.46, 0.07, 0.49, 0.46, 0.13, -0.27, 0.47},
                {-0.12, 0.04, -0.45, 0.09, 0.45, -0.06, -0.01},
                {-0.02, 0.22, 0.01, -0.22, 0.15, 0.09, -0.49}}
            m_mlp.bias_h = {
                {0.19},
                {-0.23},
                {-0.21},
                {-0.1},
                {0.14},
                {0.37},
                {0.46}}
            m_mlp.bias_o = {
                {-0.4},
                {0.25},
                {0.02}}

            m_mlp.Train()

            Dim expectedOutput = m_targetArray3XOR

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MatrixMLPIrisAnalog()

            m_mlp.nbIterations = 1500
            m_mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.Sigmoid)

            InitIrisAnalog(m_mlp)
            m_mlp.InitializeStruct(m_neuronCountIrisAnalog451, addBiasColumn:=True)
            m_mlp.Initialize(learningRate:=0.1)

            m_mlp.weights_ih = {
                {-0.29, 0.17, 0.2, -0.23},
                {-0.08, -0.24, 0.46, 0.42},
                {-0.24, -0.35, -0.24, -0.06},
                {0.3, 0.23, -0.23, 0.04},
                {0.2, -0.13, 0.48, -0.06}}
            m_mlp.weights_ho = {
                {-0.33, -0.45, 0.34, 0.43, 0.34}}
            m_mlp.bias_h = {
                {-0.25},
                {-0.33},
                {0.24},
                {0.04},
                {-0.47}}
            m_mlp.bias_o = {
                {-0.35}}

            m_mlp.Train(learningMode:=enumLearningMode.Vectorial)

            Const expectedSuccess# = 0.953#
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.032#
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            If m_mlp.successPC = 1 Then
                Dim expectedOutput = m_targetArrayIrisAnalog
                Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
                Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
                Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
                Assert.AreEqual(sExpectedOutput, sOutput)
            End If

        End Sub

        <TestMethod()>
        Public Sub MatrixMLPIrisLogical()

            m_mlp.nbIterations = 400
            m_mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.Sigmoid)

            InitIrisLogical(m_mlp)
            m_mlp.Initialize(learningRate:=0.1)

            m_mlp.weights_ih = {
                {-0.33, -0.42, -0.14, -0.47},
                {-0.06, 0.45, -0.08, 0.33},
                {-0.09, -0.49, -0.18, 0.39},
                {0.2, -0.08, -0.01, -0.29}}
            m_mlp.weights_ho = {
                {0.35, 0.44, -0.07, 0.32},
                {-0.4, -0.21, -0.38, 0.39},
                {0.42, 0.11, 0.41, -0.44}}
            m_mlp.bias_h = {
                {0.18},
                {-0.37},
                {-0.19},
                {-0.16}}
            m_mlp.bias_o = {
                {0.44},
                {-0.08},
                {0.43}}

            m_mlp.Train(learningMode:=enumLearningMode.Vectorial)

            Const expectedSuccess# = 0.977#
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.038#
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            If m_mlp.successPC = 1 Then
                Dim expectedOutput = m_targetArrayIrisLogical
                Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
                Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
                Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
                Assert.AreEqual(sExpectedOutput, sOutput)
            End If

        End Sub

    End Class

End Namespace