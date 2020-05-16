
Option Infer On

'Imports Perceptron
'Imports Perceptron.MLP
'Imports Perceptron.MLP.ActivationFunction

Imports Microsoft.VisualStudio.TestTools.UnitTesting


Namespace VectorizedMatrixMLP

    <TestClass()> _
    Public Class clsVecMatrixMLPTest

        Private m_mlp As New clsVectorizedMatrixMLP

        'Private m_mlp As New clsMLPClassic ' Not same weight array size
        'Private m_mlp As New MatrixMLP.MultiLayerPerceptron ' WeightInit not implemented
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

        <TestMethod()> _
        Public Sub VectorizedMatrixMLPXORSigmoid()

            InitXOR()
            m_mlp.learningRate = 1.5
            m_mlp.nbIterations = 1000 ' Sigmoid: works
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=1)

            m_mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.WeightInit(1, {
                {0.04, 0.05, 0.09},
                {0.34, 0.02, 0.53},
                {0.21, 0.06, 0.93}})
            m_mlp.WeightInit(2, {
                {0.18},
                {0.97},
                {0.18},
                {0.24}})

            'm_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)
            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR

            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.04
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub VectorizedMatrixMLPXORSigmoidWithoutBias()

            m_mlp.Init(learningRate:=0.1!, weightAdjustment:=0.02!)
            InitXOR()
            m_mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=False)

            m_mlp.nbIterations = 30000
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=0.2!)

            Dim arVal1#(,) = {
                {0.7, 0.28},
                {0.25, 0.69}}
            Dim arVal2#(,) = {
                {0.2},
                {0.98}}
            m_mlp.WeightInit(1, arVal1)
            m_mlp.WeightInit(2, arVal2)

            m_mlp.Train()

            'Dim expectedOutput = m_targetArrayXOR
            Dim expectedOutput = New Double(,) {
                {0.9},
                {0.1},
                {0.9},
                {0.2}}

            Dim expectedMatrix As MatrixMLP.Matrix = expectedOutput ' Double(,) -> Matrix

            Dim outputMaxtrix As MatrixMLP.Matrix = m_mlp.outputArraySingle
            Dim sOutput = outputMaxtrix.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.12
            Dim rLoss# = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub VectorizedMatrixMLPXORSigmoidWithoutBias231()

            m_mlp.Init(learningRate:=0.1!, weightAdjustment:=0.02!)
            InitXOR()
            m_mlp.InitStruct(m_neuronCountXOR231, addBiasColumn:=False)

            m_mlp.nbIterations = 10000
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=0.2!)

            Dim arVal1#(,) = {
                {0.88, 0.94, 0.23},
                {0.02, 0.49, 0.25}}
            Dim arVal2#(,) = {
                {0.89},
                {0.65},
                {0.18}}
            m_mlp.WeightInit(1, arVal1)
            m_mlp.WeightInit(2, arVal2)

            m_mlp.Train()

            'Dim expectedOutput = m_targetArrayXOR
            Dim expectedOutput = New Double(,) {
                {0.9},
                {0.1},
                {0.9},
                {0.1}}

            Dim expectedMatrix As MatrixMLP.Matrix = expectedOutput ' Double(,) -> Matrix

            Dim outputMaxtrix As MatrixMLP.Matrix = m_mlp.outputArraySingle
            Dim sOutput = outputMaxtrix.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.1
            Dim rLoss# = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub VectorizedMatrixMLPXORSigmoidWithoutBias2()

            InitXOR()
            m_mlp.learningRate = 1.1
            m_mlp.nbIterations = 8000 ' Sigmoid: works
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1.1!, center:=0.5!)

            m_mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=False)

            m_mlp.WeightInit(1, {
                {0.65, 0.88},
                {0.79, 0.63}})
            m_mlp.WeightInit(2, {
                {0.1},
                {0.88}})

            'm_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)
            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR

            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.04
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLPXORHTangent()

            InitXOR()
            m_mlp.nbIterations = 600 ' Hyperbolic tangent: works
            m_mlp.SetActivationFunction(TActivationFunction.HyperbolicTangent, gain:=1, center:=1)

            m_mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.WeightInit(1, {
                 {-0.5, -0.78, -0.07},
                 {0.54, 0.32, -0.13},
                 {-0.29, 0.89, -0.8}})
            m_mlp.WeightInit(2, {
                 {0.28},
                 {-0.94},
                 {-0.5},
                 {-0.36}})

            'm_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)
            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR

            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.02
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLPXORGaussian()

            InitXOR()
            m_mlp.nbIterations = 400 ' Gaussian: works
            m_mlp.SetActivationFunction(TActivationFunction.Gaussian, gain:=1, center:=0)

            m_mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.WeightInit(1, {
                {-0.5, -0.78, -0.07},
                {0.54, 0.32, -0.13},
                {-0.29, 0.89, -0.8}})
            m_mlp.WeightInit(2, {
                {0.28},
                {-0.94},
                {-0.5},
                {-0.36}})

            'm_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)
            m_mlp.Train()

            Dim expectedOutput = New Double(,) {
                {0.99},
                {0.03},
                {0.99},
                {0.03}}

            'Dim sOutput = m_mlp.outputMatrix.ToString()
            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.02 '0.08
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLPXORSinus()

            InitXOR()
            m_mlp.nbIterations = 200 ' Sinus: works
            m_mlp.SetActivationFunction(TActivationFunction.Sinus, gain:=1, center:=0)

            m_mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=False)

            m_mlp.WeightInit(1, {
                {0.7, 0.8},
                {0.04, 0.59}})
            m_mlp.WeightInit(2, {
                {0.03},
                {0.66}})

            'm_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)
            m_mlp.Train()

            Dim expectedOutput = New Double(,) {
                {0.99},
                {0.0},
                {0.99},
                {0.0}}

            'Dim sOutput = m_mlp.outputMatrix.ToString()
            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.01 '0.02
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLPXORArcTangent()

            InitXOR()
            m_mlp.nbIterations = 500 ' Arc Tangent: works
            m_mlp.SetActivationFunction(TActivationFunction.ArcTangent, gain:=1, center:=0)

            m_mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.WeightInit(1, {
                {-0.5, -0.78, -0.07},
                {0.54, 0.32, -0.13},
                {-0.29, 0.89, -0.8}})
            m_mlp.WeightInit(2, {
                {0.28},
                {-0.94},
                {-0.5},
                {-0.36}})

            'm_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)
            m_mlp.Train()

            Dim expectedOutput = New Double(,) {
                {0.99},
                {0},
                {0.99},
                {0}}

            'Dim sOutput = m_mlp.outputMatrix.ToString()
            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.01 '0.03
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLPXORELU()

            InitXOR()
            m_mlp.nbIterations = 500 '400 ' ELU: works
            m_mlp.SetActivationFunction(TActivationFunction.ELU, gain:=0.1, center:=0.4)
            m_mlp.Init(learningRate:=0.07!, weightAdjustment:=1)

            m_mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.WeightInit(1, {
                {0.37, 0.08, 0.74},
                {0.59, 0.54, 0.32},
                {0.2, 0.78, 0.01}})
            m_mlp.WeightInit(2, {
                {0.5},
                {0.27},
                {0.4},
                {0.11}})

            'm_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)
            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR

            'Dim sOutput = m_mlp.outputMatrix.ToString()
            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0 '0.04
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLPXORReLUSigmoid()

            InitXOR()
            m_mlp.nbIterations = 5000 ' ReLUSigmoid: works
            m_mlp.SetActivationFunction(TActivationFunction.ReLuSigmoid, gain:=1, center:=0)

            m_mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.WeightInit(1, {
                {-0.5, -0.78, -0.07},
                {0.54, 0.32, -0.13},
                {-0.29, 0.89, -0.8}})
            m_mlp.WeightInit(2, {
                {0.28},
                {-0.94},
                {-0.5},
                {-0.36}})

            'm_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)
            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR

            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLPXORDbleThreshold()

            InitXOR()
            m_mlp.nbIterations = 4000
            m_mlp.SetActivationFunction(TActivationFunction.DoubleThreshold, gain:=1, center:=0)

            m_mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.WeightInit(1, {
                {0.21, 0.63, 0.54},
                {0.01, 0.43, 0.85},
                {0.77, 0.77, 0.27}})
            m_mlp.WeightInit(2, {
                {0.01},
                {0.57},
                {0.19},
                {0.66}})

            'm_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)
            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR

            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.04
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLP2XORSinus()

            Init2XOR()
            m_mlp.nbIterations = 400 ' Sinus: works
            m_mlp.SetActivationFunction(TActivationFunction.Sinus, gain:=1, center:=0)

            m_mlp.InitStruct(m_neuronCount2XOR, addBiasColumn:=True)

            m_mlp.WeightInit(1, {
                {0.15, 0.15, 0.7, 0.18, 0.84},
                {0.96, 0.32, 0.07, 0.01, 0.84},
                {0.81, 0.74, 0.84, 0.69, 0.98},
                {0.56, 0.27, 0.34, 0.08, 0.05},
                {0.71, 0.46, 0.4, 0.76, 0.22}})
            m_mlp.WeightInit(2, {
                {0.34, 0.83},
                {0.23, 0.4},
                {0.62, 0.15},
                {0.16, 0.84},
                {0.88, 0.57},
                {0.4, 0.93}})

            'm_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)
            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR

            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLP2XORGaussian()

            Init2XOR()
            m_mlp.nbIterations = 300 ' Gaussian: works
            m_mlp.SetActivationFunction(TActivationFunction.Gaussian, gain:=1, center:=1)

            m_mlp.InitStruct(m_neuronCount2XOR, addBiasColumn:=True)

            m_mlp.WeightInit(1, {
                {0.26, 0.34, 0.51, 0.79, 0.37},
                {0.38, 0.84, 0.97, 0.65, 0.93},
                {0.2, 0.22, 0.69, 0.25, 0.68},
                {0.45, 0.58, 0.77, 0.59, 0.46},
                {0.58, 0.46, 0.93, 0.47, 0.28}})
            m_mlp.WeightInit(2, {
                {0.23, 0.29},
                {0.75, 0.21},
                {0.52, 0.38},
                {0.63, 0.68},
                {0.96, 0.53},
                {0.31, 0.91}})

            'm_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)
            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR

            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.01
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLP2XORHTangent()

            Init2XOR()
            m_mlp.nbIterations = 300 ' HTan: works
            m_mlp.SetActivationFunction(TActivationFunction.HyperbolicTangent, gain:=1, center:=0.5)

            m_mlp.InitStruct(m_neuronCount2XOR, addBiasColumn:=True)

            m_mlp.WeightInit(1, {
                {0.86, 0.85, 0.65, 0.68, 0.72},
                {0.56, 0.75, 0.38, 0.56, 0.33},
                {0.62, 0.48, 0.73, 0.38, 0.19},
                {0.65, 0.43, 0.81, 0.22, 0.37},
                {0.9, 0.56, 0.25, 0.31, 0.78}})
            m_mlp.WeightInit(2, {
                {0.95, 0.22},
                {0.25, 0.58},
                {0.86, 0.86},
                {0.35, 0.68},
                {0.46, 0.76},
                {0.06, 0.21}})

            'm_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)
            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR

            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.02
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLP2XORSigmoid()

            Init2XOR()
            m_mlp.learningRate = 3.45
            m_mlp.nbIterations = 400 ' Sigmoid: works
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=0.5!)

            m_mlp.InitStruct(m_neuronCount2XOR, addBiasColumn:=True)

            m_mlp.WeightInit(1, {
                {0.02, 0.88, 0.28, 0.86, 0.3},
                {0.1, 0.03, 0.47, 0.91, 0.78},
                {0.48, 0.26, 0.17, 0.67, 0.67},
                {0.66, 0.36, 0.24, 0.21, 0.65},
                {0.04, 0.64, 0.65, 0.11, 0.28}})
            m_mlp.WeightInit(2, {
                {0.38, 0.31},
                {0.67, 0.2},
                {0.35, 0.97},
                {0.99, 0.79},
                {0.52, 0.31},
                {0.35, 0.44}})

            'm_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)
            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR

            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.02
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLP2XORArcTan()

            Init2XOR()
            m_mlp.learningRate = 0.05
            m_mlp.nbIterations = 500 '400 ' Arc tangent: works fine
            m_mlp.SetActivationFunction(TActivationFunction.ArcTangent, gain:=1, center:=0.9)

            m_mlp.InitStruct(m_neuronCount2XOR, addBiasColumn:=True)

            m_mlp.WeightInit(1, {
                {0.92, 0.83, 0.73, 0.13, 0.63},
                {0.07, 0.44, 0.83, 0.63, 0.67},
                {0.59, 0.88, 0.17, 0.97, 0.15},
                {0.85, 0.47, 0.98, 0.53, 0.33},
                {0.08, 0.39, 0.35, 0.52, 0.38}})
            m_mlp.WeightInit(2, {
                {0.77, 0.39},
                {0.72, 0.38},
                {0.12, 0.06},
                {0.16, 0.55},
                {0.85, 0.84},
                {0.41, 0.47}})

            'm_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)
            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR

            'Dim sOutput = m_mlp.outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0 '0.17
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLP2XORDbleThreshold()

            Init2XOR()
            m_mlp.learningRate = 0.15
            m_mlp.nbIterations = 900
            m_mlp.SetActivationFunction(TActivationFunction.DoubleThreshold, gain:=1, center:=0.2!)

            m_mlp.InitStruct(m_neuronCount2XOR, addBiasColumn:=True)

            m_mlp.WeightInit(1, {
                {0.85, 0.95, 0.99, 0.63, 0.09},
                {0.82, 0.05, 0.68, 0.26, 0.81},
                {0.79, 0.58, 0.7, 0.95, 0.85},
                {0.78, 0.07, 0.14, 0.46, 0.83},
                {0.4, 0.81, 0.06, 0.21, 0.95}})
            m_mlp.WeightInit(2, {
                {0.42, 0.49},
                {0.4, 0.98},
                {0.42, 0.71},
                {0.47, 0.75},
                {0.2, 0.44},
                {0.5, 0.21}})

            'm_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)
            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR

            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.01
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLP3XORGaussian()

            Init3XOR()
            m_mlp.learningRate = 0.1
            m_mlp.nbIterations = 100 ' Gaussian: works
            m_mlp.SetActivationFunction(TActivationFunction.Gaussian, gain:=1, center:=1)

            m_mlp.InitStruct(m_neuronCount3XOR, addBiasColumn:=True)

            m_mlp.WeightInit(1, {
                {0.65, 0.13, 0.57, 0.28, 0.82, 0.16, 0.44},
                {0.06, 0.45, 0.31, 0.44, 0.81, 0.49, 0.88},
                {0.27, 0.93, 0.1, 0.7, 0.84, 0.82, 0.2},
                {0.54, 0.67, 0.66, 0.92, 0.44, 0.74, 0.29},
                {0.24, 0.36, 0.14, 0.66, 0.34, 0.47, 0.32},
                {0.54, 0.2, 0.71, 0.89, 0.95, 0.85, 0.96},
                {0.42, 0.41, 0.58, 0.45, 0.14, 0.37, 1.0}})
            m_mlp.WeightInit(2, {
                {0.14, 0.58, 0.25},
                {0.31, 0.44, 0.66},
                {0.11, 0.45, 0.91},
                {0.36, 0.38, 0.42},
                {0.15, 0.82, 0.09},
                {0.17, 0.78, 0.47},
                {0.03, 0.57, 0.73},
                {0.73, 0.4, 0.81}})

            'm_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)
            m_mlp.Train()

            Dim expectedOutput = m_targetArray3XOR

            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.01
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLP3XORDbleThreshold()

            Init3XOR()
            m_mlp.learningRate = 0.8
            m_mlp.nbIterations = 400
            m_mlp.SetActivationFunction(TActivationFunction.DoubleThreshold, gain:=1, center:=2)

            m_mlp.InitStruct(m_neuronCount3XOR, addBiasColumn:=True)

            m_mlp.WeightInit(1, {
                {0.65, 0.98, 0.41, 0.71, 0.73, 0.69, 0.78},
                {0.82, 1.0, 0.34, 0.07, 0.08, 0.4, 0.29},
                {0.03, 0.86, 0.52, 0.98, 0.74, 0.91, 0.45},
                {0.86, 0.74, 0.63, 0.82, 0.66, 0.28, 0.76},
                {0.41, 0.47, 0.32, 0.34, 0.3, 0.16, 0.98},
                {0.8, 0.15, 0.03, 0.88, 0.58, 0.11, 0.1},
                {0.58, 0.02, 0.35, 0.75, 0.52, 0.47, 0.03}})
            m_mlp.WeightInit(2, {
                {0.28, 0.37, 0.8},
                {1.0, 0.13, 0.2},
                {0.11, 0.67, 0.38},
                {0.89, 0.07, 0.41},
                {0.03, 0.41, 0.53},
                {0.03, 0.73, 0.79},
                {0.23, 0.31, 0.23},
                {0.72, 0.49, 0.1}})

            'm_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)
            m_mlp.Train()

            Dim expectedOutput = m_targetArray3XOR

            Dim outputMatrix As Matrix = m_mlp.outputArraySingle ' Single(,) -> Matrix
            Dim sOutput$ = outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.01
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

    End Class

End Namespace