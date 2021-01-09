
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Imports Microsoft.VisualStudio.TestTools.UnitTesting

Namespace VectorizedMatrixMLP

    <TestClass()> _
    Public Class clsVecMatrixMLPTest

        Private m_mlp As New clsVectorizedMatrixMLP

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
        Public Sub VecMatrixMLP1XORSemiStochastic()

            InitXOR()
            m_mlp.learningRate = 0.3
            m_mlp.nbIterations = 6000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, center:=1)

            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.07, 0.82, 0.22},
                {0.62, 0.14, 0.79},
                {0.19, 0.77, 0.54}})
            m_mlp.InitializeWeights(2, {
                {0.31},
                {0.64},
                {0.17},
                {0.67}})

            m_mlp.Train(enumLearningMode.SemiStochastic)

            Dim expectedOutput = m_targetArrayXOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.04
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLP1XORStochastic()

            InitXOR()
            m_mlp.learningRate = 0.3
            m_mlp.nbIterations = 25000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, center:=1)

            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.07, 0.82, 0.22},
                {0.62, 0.14, 0.79},
                {0.19, 0.77, 0.54}})
            m_mlp.InitializeWeights(2, {
                {0.31},
                {0.64},
                {0.17},
                {0.67}})

            m_mlp.Train(enumLearningMode.Stochastic)

            Dim expectedOutput = m_targetArrayXOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.04
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLP1XORTrainVector()

            InitXOR()
            m_mlp.learningRate = 0.3
            m_mlp.nbIterations = 5000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, center:=1)

            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.82, 0.4, 0.9},
                {0.02, 0.88, 0.15},
                {0.49, 0.99, 0.6}})
            m_mlp.InitializeWeights(2, {
                {0.38},
                {0.86},
                {0.72},
                {0.82}})

            m_mlp.TrainVector()

            Dim expectedOutput = m_targetArrayXOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.15
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLP1XORSigmoid()

            InitXOR()
            m_mlp.learningRate = 1.5
            m_mlp.nbIterations = 1000 ' Sigmoid: works
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, center:=1)

            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.04, 0.05, 0.09},
                {0.34, 0.02, 0.53},
                {0.21, 0.06, 0.93}})
            m_mlp.InitializeWeights(2, {
                {0.18},
                {0.97},
                {0.18},
                {0.24}})

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.04
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLP1XORSigmoidWithoutBias()

            InitXOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.02!)
            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=False)

            m_mlp.nbIterations = 30000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, center:=0.2!)

            m_mlp.InitializeWeights(1, {
                {0.7, 0.28},
                {0.25, 0.69}})
            m_mlp.InitializeWeights(2, {
                {0.2},
                {0.98}})

            m_mlp.Train()

            'Dim expectedOutput = m_targetArrayXOR
            Dim expectedOutput = New Double(,) {
                {0.9},
                {0.1},
                {0.9},
                {0.1}}

            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.12
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLP1XORSigmoidWithoutBias231()

            InitXOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.02!)
            m_mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=False)

            m_mlp.nbIterations = 9000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, center:=0.2!)

            m_mlp.InitializeWeights(1, {
                {0.88, 0.94, 0.23},
                {0.02, 0.49, 0.25}})
            m_mlp.InitializeWeights(2, {
                {0.89},
                {0.65},
                {0.18}})

            m_mlp.Train()

            'Dim expectedOutput = m_targetArrayXOR
            Dim expectedOutput = New Double(,) {
                {0.9},
                {0.1},
                {0.9},
                {0.1}}

            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.1
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLP1XORSigmoidWithoutBias2()

            InitXOR()
            m_mlp.learningRate = 1.1
            m_mlp.nbIterations = 8000 ' Sigmoid: works
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=1.1!, center:=0.5!)

            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=False)

            m_mlp.InitializeWeights(1, {
                {0.65, 0.88},
                {0.79, 0.63}})
            m_mlp.InitializeWeights(2, {
                {0.1},
                {0.88}})

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.04
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLP1XORTanh()

            InitXOR()
            m_mlp.nbIterations = 600 ' Hyperbolic tangent: works
            'm_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, center:=1)
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2, center:=1)

            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                 {-0.5, -0.78, -0.07},
                 {0.54, 0.32, -0.13},
                 {-0.29, 0.89, -0.8}})
            m_mlp.InitializeWeights(2, {
                 {0.28},
                 {-0.94},
                 {-0.5},
                 {-0.36}})

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLP1XORGaussian()

            InitXOR()
            m_mlp.nbIterations = 400 ' Gaussian: works
            m_mlp.SetActivationFunction(enumActivationFunction.Gaussian)

            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {-0.5, -0.78, -0.07},
                {0.54, 0.32, -0.13},
                {-0.29, 0.89, -0.8}})
            m_mlp.InitializeWeights(2, {
                {0.28},
                {-0.94},
                {-0.5},
                {-0.36}})

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = New Double(,) {
                {0.99},
                {0.03},
                {0.99},
                {0.03}}

            Dim sOutput$ = m_mlp.output.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02 '0.08
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLP1XORSinus()

            InitXOR()
            m_mlp.nbIterations = 200 ' Sinus: works
            m_mlp.SetActivationFunction(enumActivationFunction.Sinus)

            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=False)

            m_mlp.InitializeWeights(1, {
                {0.7, 0.8},
                {0.04, 0.59}})
            m_mlp.InitializeWeights(2, {
                {0.03},
                {0.66}})

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = New Double(,) {
                {0.99},
                {0.0},
                {0.99},
                {0.0}}

            Dim sOutput$ = m_mlp.output.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01 '0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLP1XORArcTangent()

            InitXOR()
            m_mlp.nbIterations = 500 ' Arc Tangent: works
            m_mlp.SetActivationFunction(enumActivationFunction.ArcTangent)

            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {-0.5, -0.78, -0.07},
                {0.54, 0.32, -0.13},
                {-0.29, 0.89, -0.8}})
            m_mlp.InitializeWeights(2, {
                {0.28},
                {-0.94},
                {-0.5},
                {-0.36}})

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = New Double(,) {
                {0.99},
                {0},
                {0.99},
                {0}}

            Dim sOutput$ = m_mlp.output.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01 '0.03
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLP1XORELU()

            InitXOR()
            m_mlp.Initialize(learningRate:=0.07!)
            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.nbIterations = 500 '400 ' ELU: works
            m_mlp.SetActivationFunction(enumActivationFunction.ELU, gain:=0.1, center:=0.4)

            m_mlp.InitializeWeights(1, {
                {0.37, 0.08, 0.74},
                {0.59, 0.54, 0.32},
                {0.2, 0.78, 0.01}})
            m_mlp.InitializeWeights(2, {
                {0.5},
                {0.27},
                {0.4},
                {0.11}})

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR

            Dim sOutput$ = m_mlp.output.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLP1XORReLUSigmoid()

            InitXOR()
            m_mlp.nbIterations = 5000 ' ReLUSigmoid: works
            m_mlp.SetActivationFunction(enumActivationFunction.ReLuSigmoid)

            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {-0.5, -0.78, -0.07},
                {0.54, 0.32, -0.13},
                {-0.29, 0.89, -0.8}})
            m_mlp.InitializeWeights(2, {
                {0.28},
                {-0.94},
                {-0.5},
                {-0.36}})

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR

            Dim sOutput$ = m_mlp.output.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLP1XORDbleThreshold()

            InitXOR()
            m_mlp.nbIterations = 4000
            m_mlp.SetActivationFunction(enumActivationFunction.DoubleThreshold)

            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.21, 0.63, 0.54},
                {0.01, 0.43, 0.85},
                {0.77, 0.77, 0.27}})
            m_mlp.InitializeWeights(2, {
                {0.01},
                {0.57},
                {0.19},
                {0.66}})

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.04
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLP2XORSinus()

            Init2XOR()
            m_mlp.nbIterations = 400 ' Sinus: works
            m_mlp.SetActivationFunction(enumActivationFunction.Sinus)

            m_mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.15, 0.15, 0.7, 0.18, 0.84},
                {0.96, 0.32, 0.07, 0.01, 0.84},
                {0.81, 0.74, 0.84, 0.69, 0.98},
                {0.56, 0.27, 0.34, 0.08, 0.05},
                {0.71, 0.46, 0.4, 0.76, 0.22}})
            m_mlp.InitializeWeights(2, {
                {0.34, 0.83},
                {0.23, 0.4},
                {0.62, 0.15},
                {0.16, 0.84},
                {0.88, 0.57},
                {0.4, 0.93}})

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR

            Dim sOutput$ = m_mlp.output.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLP2XORGaussian()

            Init2XOR()
            m_mlp.nbIterations = 300 ' Gaussian: works
            m_mlp.SetActivationFunction(enumActivationFunction.Gaussian, center:=1)

            m_mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.26, 0.34, 0.51, 0.79, 0.37},
                {0.38, 0.84, 0.97, 0.65, 0.93},
                {0.2, 0.22, 0.69, 0.25, 0.68},
                {0.45, 0.58, 0.77, 0.59, 0.46},
                {0.58, 0.46, 0.93, 0.47, 0.28}})
            m_mlp.InitializeWeights(2, {
                {0.23, 0.29},
                {0.75, 0.21},
                {0.52, 0.38},
                {0.63, 0.68},
                {0.96, 0.53},
                {0.31, 0.91}})

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLP2XORTanh()

            Init2XOR()
            m_mlp.nbIterations = 300 ' HTan: works
            'm_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, center:=0.5)
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2, center:=0.5)

            m_mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.86, 0.85, 0.65, 0.68, 0.72},
                {0.56, 0.75, 0.38, 0.56, 0.33},
                {0.62, 0.48, 0.73, 0.38, 0.19},
                {0.65, 0.43, 0.81, 0.22, 0.37},
                {0.9, 0.56, 0.25, 0.31, 0.78}})
            m_mlp.InitializeWeights(2, {
                {0.95, 0.22},
                {0.25, 0.58},
                {0.86, 0.86},
                {0.35, 0.68},
                {0.46, 0.76},
                {0.06, 0.21}})

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLP2XORSigmoid()

            Init2XOR()
            m_mlp.learningRate = 3.45
            m_mlp.nbIterations = 400 ' Sigmoid: works
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, center:=0.5!)

            m_mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.02, 0.88, 0.28, 0.86, 0.3},
                {0.1, 0.03, 0.47, 0.91, 0.78},
                {0.48, 0.26, 0.17, 0.67, 0.67},
                {0.66, 0.36, 0.24, 0.21, 0.65},
                {0.04, 0.64, 0.65, 0.11, 0.28}})
            m_mlp.InitializeWeights(2, {
                {0.38, 0.31},
                {0.67, 0.2},
                {0.35, 0.97},
                {0.99, 0.79},
                {0.52, 0.31},
                {0.35, 0.44}})

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLP2XORArcTan()

            Init2XOR()
            m_mlp.learningRate = 0.05
            m_mlp.nbIterations = 500 '400 ' Arc tangent: works fine
            m_mlp.SetActivationFunction(enumActivationFunction.ArcTangent, center:=0.9)

            m_mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.92, 0.83, 0.73, 0.13, 0.63},
                {0.07, 0.44, 0.83, 0.63, 0.67},
                {0.59, 0.88, 0.17, 0.97, 0.15},
                {0.85, 0.47, 0.98, 0.53, 0.33},
                {0.08, 0.39, 0.35, 0.52, 0.38}})
            m_mlp.InitializeWeights(2, {
                {0.77, 0.39},
                {0.72, 0.38},
                {0.12, 0.06},
                {0.16, 0.55},
                {0.85, 0.84},
                {0.41, 0.47}})

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0 '0.17
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub



        <TestMethod()>
        Public Sub VecMatrixMLP2XORDbleThreshold()

            Init2XOR()
            m_mlp.learningRate = 0.15
            m_mlp.nbIterations = 900
            m_mlp.SetActivationFunction(enumActivationFunction.DoubleThreshold, center:=0.2!)

            m_mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.85, 0.95, 0.99, 0.63, 0.09},
                {0.82, 0.05, 0.68, 0.26, 0.81},
                {0.79, 0.58, 0.7, 0.95, 0.85},
                {0.78, 0.07, 0.14, 0.46, 0.83},
                {0.4, 0.81, 0.06, 0.21, 0.95}})
            m_mlp.InitializeWeights(2, {
                {0.42, 0.49},
                {0.4, 0.98},
                {0.42, 0.71},
                {0.47, 0.75},
                {0.2, 0.44},
                {0.5, 0.21}})

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLP3XORSigmoid()

            Init3XOR()
            m_mlp.learningRate = 0.4
            m_mlp.weightAdjustment = 0.35
            m_mlp.nbIterations = 400 ' Sigmoid: works
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.29, 0.55, 0.98, 0.22, 0.32, 0.41},
                {0.53, 0.09, 0.67, 0.13, 0.4, 0.43},
                {0.44, 0.46, 0.43, 0.27, 0.14, 0.18},
                {0.62, 0.56, 0.36, 0.84, 0.68, 0.69},
                {0.37, 0.34, 0.14, 0.86, 0.18, 0.17},
                {0.36, 0.78, 0.75, 0.82, 0.96, 0.92},
                {0.49, 0.94, 0.81, 0.93, 0.21, 0.78}})
            m_mlp.InitializeWeights(2, {
                {0.38, 0.45, 0.95},
                {0.37, 0.14, 0.26},
                {0.25, 0.7, 0.35},
                {0.62, 0.4, 0.79},
                {0.45, 0.87, 0.29},
                {0.85, 0.98, 0.27},
                {0.67, 0.9, 0.5}})

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = m_targetArray3XOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLP3XORGaussian()

            Init3XOR()
            m_mlp.learningRate = 0.1
            m_mlp.nbIterations = 100 ' Gaussian: works
            m_mlp.SetActivationFunction(enumActivationFunction.Gaussian, center:=1)

            m_mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.65, 0.13, 0.57, 0.28, 0.82, 0.16, 0.44},
                {0.06, 0.45, 0.31, 0.44, 0.81, 0.49, 0.88},
                {0.27, 0.93, 0.1, 0.7, 0.84, 0.82, 0.2},
                {0.54, 0.67, 0.66, 0.92, 0.44, 0.74, 0.29},
                {0.24, 0.36, 0.14, 0.66, 0.34, 0.47, 0.32},
                {0.54, 0.2, 0.71, 0.89, 0.95, 0.85, 0.96},
                {0.42, 0.41, 0.58, 0.45, 0.14, 0.37, 1.0}})
            m_mlp.InitializeWeights(2, {
                {0.14, 0.58, 0.25},
                {0.31, 0.44, 0.66},
                {0.11, 0.45, 0.91},
                {0.36, 0.38, 0.42},
                {0.15, 0.82, 0.09},
                {0.17, 0.78, 0.47},
                {0.03, 0.57, 0.73},
                {0.73, 0.4, 0.81}})

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = m_targetArray3XOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLP3XORDbleThreshold()

            Init3XOR()
            m_mlp.learningRate = 0.8
            m_mlp.nbIterations = 400
            m_mlp.SetActivationFunction(enumActivationFunction.DoubleThreshold, center:=2)

            m_mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)

            m_mlp.InitializeWeights(1, {
                {0.65, 0.98, 0.41, 0.71, 0.73, 0.69, 0.78},
                {0.82, 1.0, 0.34, 0.07, 0.08, 0.4, 0.29},
                {0.03, 0.86, 0.52, 0.98, 0.74, 0.91, 0.45},
                {0.86, 0.74, 0.63, 0.82, 0.66, 0.28, 0.76},
                {0.41, 0.47, 0.32, 0.34, 0.3, 0.16, 0.98},
                {0.8, 0.15, 0.03, 0.88, 0.58, 0.11, 0.1},
                {0.58, 0.02, 0.35, 0.75, 0.52, 0.47, 0.03}})
            m_mlp.InitializeWeights(2, {
                {0.28, 0.37, 0.8},
                {1.0, 0.13, 0.2},
                {0.11, 0.67, 0.38},
                {0.89, 0.07, 0.41},
                {0.03, 0.41, 0.53},
                {0.03, 0.73, 0.79},
                {0.23, 0.31, 0.23},
                {0.72, 0.49, 0.1}})

            'm_mlp.TrainVector()
            'm_mlp.Train(enumLearningMode.Vectorial)
            m_mlp.Train()

            Dim expectedOutput = m_targetArray3XOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLPIrisFlowerLogical()

            ' 95.6% prediction, 96.4% learning with 700  iterations in 300 msec. (64 bits)
            ' 95.6% prediction, 96.4% learning with 2500 iterations in 1.2 sec.  (32 bits)

            InitIrisFlowerLogical(m_mlp)
            m_mlp.InitializeStruct(m_neuronCountIrisFlowerLogical453, addBiasColumn:=True)
            m_mlp.Initialize(learningRate:=0.01)

            m_mlp.nbIterations = 2500
            m_mlp.minimalSuccessTreshold = 0.3
            m_mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {-0.4, -0.03, 0.49, 0.24, -0.06},
                {-0.49, -0.44, -0.05, -0.1, -0.06},
                {-0.23, 0.24, 0.11, 0.13, 0.25},
                {-0.15, 0.49, 0.11, -0.4, 0.31},
                {-0.35, 0.2, 0.02, 0.31, -0.18}})
            m_mlp.InitializeWeights(2, {
                {-0.29, 0.26, -0.16},
                {-0.33, 0.43, 0.08},
                {-0.19, -0.47, -0.29},
                {0.14, 0.3, -0.34},
                {0.29, -0.21, -0.29},
                {0.33, -0.04, -0.48}})

            m_mlp.Train(learningMode:=enumLearningMode.Vectorial)

            Dim expectedSuccess# = 0.964
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.085
            Dim loss# = m_mlp.averageError
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
            Dim expectedSuccessPrediction# = 0.956
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub VecMatrixMLPIrisFlowerLogicalSigmoid()

            ' 97.8% prediction, 96.1% learning with 1500 iterations in 1.8 sec.

            InitIrisFlowerLogical4Layers(m_mlp)
            m_mlp.Initialize(learningRate:=0.005, weightAdjustment:=0.005)

            m_mlp.nbIterations = 1500
            m_mlp.minimalSuccessTreshold = 0.3
            m_mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {-0.4, -0.36, 0.18, 0.43, 0.48, -0.34, 0.37, -0.39, -0.28, -0.09, -0.32, 0.15, 0.04, 0.24, -0.27, -0.21},
                {0.03, 0.23, -0.15, 0.32, 0.39, 0.3, 0.13, -0.13, 0.4, -0.38, -0.23, -0.07, -0.48, 0.29, 0.34, 0.19},
                {0.49, 0.32, -0.35, -0.33, -0.44, 0.43, 0.31, 0.29, -0.33, 0.48, 0.01, 0.3, -0.43, -0.03, -0.47, 0.28},
                {0.44, -0.25, -0.43, 0.4, 0.15, -0.22, -0.2, -0.18, -0.47, 0.36, 0.39, -0.06, -0.41, -0.07, 0.07, 0.17},
                {0.22, 0.09, -0.43, -0.27, 0.1, 0.41, 0.06, 0.15, 0.41, -0.22, 0.33, -0.21, -0.1, 0.16, -0.25, 0.15}})
            m_mlp.InitializeWeights(2, {
                {-0.42, 0.04, 0.27, 0.29, -0.16, 0.04, 0.35, 0.16},
                {0.25, 0.4, 0.46, 0.41, 0.23, 0.38, -0.42, -0.27},
                {0.08, 0.33, 0.25, -0.06, 0.21, 0.21, 0.15, -0.34},
                {-0.15, 0.5, -0.13, -0.48, 0.18, -0.48, 0.13, -0.3},
                {0.14, 0.29, -0.48, 0.39, 0.3, 0.2, -0.17, -0.32},
                {0.24, -0.09, -0.02, 0.2, 0.46, 0.15, 0.41, -0.47},
                {-0.3, 0.1, 0.21, 0.06, 0.4, 0.31, 0.44, -0.14},
                {0.39, 0.11, -0.06, -0.16, -0.34, 0.33, 0.48, 0.23},
                {-0.23, 0.26, -0.23, 0.45, 0.36, -0.31, -0.07, 0.38},
                {0.07, 0.19, -0.19, -0.27, 0.45, -0.3, 0.2, -0.42},
                {-0.15, 0.32, -0.42, -0.19, -0.43, -0.2, 0.33, 0.35},
                {0.15, 0.5, -0.31, -0.23, 0.49, -0.48, 0.42, -0.07},
                {0.25, -0.08, -0.3, -0.12, 0.47, -0.34, -0.49, -0.33},
                {0.33, -0.44, -0.17, -0.38, 0.13, -0.37, 0.45, -0.1},
                {0.23, -0.25, 0.32, -0.49, 0.4, -0.09, -0.29, -0.04},
                {-0.06, -0.4, -0.29, -0.31, -0.26, 0.12, 0.01, 0.05},
                {0.28, 0.38, -0.02, 0.02, -0.5, -0.17, -0.38, -0.32}})
            m_mlp.InitializeWeights(3, {
                {0.31, 0.42, -0.01},
                {-0.49, 0.46, 0.21},
                {-0.29, 0.02, -0.45},
                {0.29, 0.03, 0.24},
                {-0.3, 0.41, 0.03},
                {-0.16, -0.28, 0.24},
                {0.44, 0.37, 0.45},
                {0.32, 0.42, -0.29},
                {0.05, 0.28, -0.16}})

            m_mlp.Train(learningMode:=enumLearningMode.Vectorial)

            Dim expectedSuccess# = 0.961
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.05
            Dim loss# = m_mlp.averageError
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