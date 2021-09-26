
Imports Perceptron
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Namespace EncogMLP

    <TestClass()>
    Public Class clsMLPEncogTest

        Private m_mlp As New clsMLPEncog

        'Private m_mlp As New clsMLPClassic ' 1/13 success
        'Private m_mlp As New clsMLPOOP ' 2/12 success
        'Private m_mlp As New clsMLPAccord ' 1/13 success
        'Private m_mlp As New clsMLPTensorFlow  ' 0/14 success

        <TestInitialize()>
        Public Sub Init()
        End Sub

        Private Sub InitXOR()
            m_mlp.Initialize(learningRate:=0)
            m_mlp.inputArray = m_inputArrayXOR
            m_mlp.targetArray = m_targetArrayXOR
            m_mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
        End Sub

        Private Sub Init2XOR()
            m_mlp.Initialize(learningRate:=0)
            m_mlp.inputArray = m_inputArray2XOR
            m_mlp.targetArray = m_targetArray2XOR
            m_mlp.InitializeStruct(m_neuronCount2XOR452, addBiasColumn:=True)
        End Sub

        Private Sub Init3XOR()
            m_mlp.Initialize(learningRate:=0)
            m_mlp.inputArray = m_inputArray3XOR
            m_mlp.targetArray = m_targetArray3XOR
            m_mlp.InitializeStruct(m_neuronCount3XOR673, addBiasColumn:=True)
        End Sub

        Private Sub InitIrisFlowerLogicalOriginal()
            m_mlp.Initialize(learningRate:=0)
            'm_mlp.inputArray = m_inputArrayIrisFlower
            'm_mlp.targetArray = m_targetArrayIrisFlowerLogical
            'm_mlp.inputArray = m_inputArrayIrisFlowerTrain
            m_mlp.inputArray = m_inputArrayIrisFlowerTrainOriginal
            m_mlp.targetArray = m_targetArrayIrisFlowerLogicalTrain
            m_mlp.InitializeStruct(m_neuronCountIrisFlowerLogical, addBiasColumn:=True)
        End Sub

        <TestMethod()>
        Public Sub EncogMLP1XORSigmoidWithoutBias()

            InitXOR()
            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=False)

            m_mlp.nbIterations = 2000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {0.78, 0.18},
                {0.61, 0.61}})
            m_mlp.InitializeWeights(2, {
                {-0.76, 0.07}})

            m_mlp.Train(enumLearningMode.Vectorial)

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.03
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub EncogMLP1XORSigmoidWithoutBias231()

            InitXOR()
            m_mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=False)

            m_mlp.nbIterations = 200
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {0.46, -0.58},
                {-0.61, 0.17},
                {-0.18, 0.17}})
            m_mlp.InitializeWeights(2, {
                {0.48, -0.16, 0.3}})

            m_mlp.Train(enumLearningMode.Vectorial)

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub EncogMLP1XOR4Layers()

            TestMLP1XOR4Layers(m_mlp, weightAdjustment:=0, gain:=2,
                learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub EncogMLP1XOR5Layers()

            InitXOR()
            m_mlp.InitializeStruct(m_neuronCountXOR5Layers23331, addBiasColumn:=True)

            m_mlp.nbIterations = 300
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {-0.57, -0.7, -0.7},
                {-0.38, -0.34, 0.81},
                {0.21, 0.01, 0.04}})
            m_mlp.InitializeWeights(2, {
                {-0.91, 0.14, -0.48, 0.03},
                {0.63, 0.3, 0.44, 0.11},
                {-0.06, -0.93, 0.54, 0.26}})
            m_mlp.InitializeWeights(3, {
                {0.88, -0.92, -0.96, 0.09},
                {0.64, -0.05, 0.63, 0.37},
                {0.32, -0.65, 0.6, 0.93}})
            m_mlp.InitializeWeights(4, {
                {-0.51, -0.9, 0.8, 0.31}})

            m_mlp.Train(enumLearningMode.Vectorial)

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
        Public Sub EncogMLP1XORSigmoid()

            TestMLP1XORSigmoid231(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub EncogMLP1XORSigmoidRProp()

            TestMLP1XORSigmoidRProp(m_mlp, trainingAlgorithm:=enumTrainingAlgorithm.RProp)

        End Sub

        Private Sub TestMLP1XORSigmoid231(mlp As clsMLPGeneric,
            Optional nbIterations% = 300,
            Optional expectedLoss# = 0.01#,
            Optional learningRate! = 0.1!,
            Optional weightAdjustment! = 0,
            Optional gain! = 1,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

            InitXOR()
            mlp.Initialize(learningRate, weightAdjustment)
            mlp.nbIterations = nbIterations
            mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
            mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)

            mlp.InitializeWeights(1, {
                {-0.01, 0.26, -0.88},
                {0.8, -0.2, 0.46},
                {-0.31, -0.2, -0.39}})
            mlp.InitializeWeights(2, {
                {0.05, 0.99, 0.35, 0.35}})

            mlp.Train(learningMode)

            Dim expectedOutput = m_targetArrayXOR

            Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub EncogMLP1XORTanh()

            TestMLP1XORTanh(m_mlp, weightAdjustment:=0, gain:=2,
                learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub EncogMLP1XORTanhRProp()

            TestMLP1XORTanhRProp(m_mlp, trainingAlgorithm:=enumTrainingAlgorithm.RProp)

        End Sub

        <TestMethod()>
        Public Sub EncogMLP1XORTanh261()

            TestMLP1XORTanh261(m_mlp, nbIterations:=200, gain:=2,
                learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub EncogMLP2XORSigmoidStdr()

            TestMLP2XORSigmoid(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub EncogMLP2XORSigmoid()

            Init2XOR()

            m_mlp.nbIterations = 200
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {-0.28, -0.99, 0.92, -0.54, -0.54},
                {-0.91, -0.29, 0.21, -0.12, -0.47},
                {0.59, -0.3, 0.57, -0.13, -0.16},
                {0.62, -0.04, -0.83, -0.08, 0.91},
                {-0.77, 0.61, 0.98, 0.65, 0.55}})
            m_mlp.InitializeWeights(2, {
                {-0.42, 0.01, 0.77, -0.39, -0.74, -0.66},
                {-0.44, -0.27, 0.12, 0.36, 0.9, -0.97}})

            m_mlp.Train(enumLearningMode.Vectorial)

            Dim expectedOutput = m_targetArray2XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub EncogMLP2XORTanh()

            TestMLP2XORTanh(m_mlp, weightAdjustment:=0, gain:=2,
                learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub EncogMLP2XORTanh462()

            TestMLP2XORTanh462(m_mlp, gain:=2,
                learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub EncogMLP2XORSinus()

            Init2XOR()

            m_mlp.nbIterations = 200
            m_mlp.SetActivationFunction(enumActivationFunction.Sinus)

            m_mlp.InitializeWeights(1, {
                {-0.07, -0.39, -0.31, -0.16, -0.15},
                {0.78, -0.22, 0.84, -0.79, -0.61},
                {0.53, -0.7, 0.7, -0.01, -0.54},
                {0.54, -0.5, -0.9, 0.1, -0.74},
                {0.62, -0.4, 0.06, 0.52, 0.42}})
            m_mlp.InitializeWeights(2, {
                {0.99, 0.69, -0.17, -0.63, 0.42, -0.28},
                {-0.49, 0.7, 0.25, -0.25, -0.71, 0.99}})

            m_mlp.Train(enumLearningMode.Vectorial)

            Dim expectedOutput = m_targetArray2XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub EncogMLP3XORSigmoid()

            Init3XOR()

            m_mlp.nbIterations = 300
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {1.0, 0.68, -0.01, -0.58, -0.27, -0.56, 0.55},
                {0.71, -0.96, -0.45, 0.58, 0.71, -0.28, -0.95},
                {0.66, -0.32, -0.08, 0.61, 0.8, 0.23, -0.06},
                {0.8, -0.05, -0.42, -0.73, 0.64, -0.98, 0.91},
                {0.35, 0.36, -0.47, 0.04, -0.7, -0.18, 0.19},
                {0.39, 0.56, -0.32, -0.47, -0.58, -0.87, 0.14},
                {0.7, -0.76, -0.16, 0.33, 0.42, 0.25, -0.17}})
            m_mlp.InitializeWeights(2, {
                {-0.14, -0.99, 0.57, 0.54, -0.63, -0.12, 0.14, 0.83},
                {-0.71, -0.86, -0.87, -0.69, -0.06, 0.3, -0.17, 0.68},
                {-0.9, 0.07, -0.94, -0.14, -0.67, 0.06, -0.46, -0.45}})

            m_mlp.Train(enumLearningMode.Vectorial)

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        '<TestMethod()>
        'Public Sub EncogMLP3XORSigmoidStdr()

        '    TestMLP3XORSigmoid(m_mlp, nbIterations:=10000, learningMode:=enumLearningMode.Vectorial)

        'End Sub

        '<TestMethod()>
        'Public Sub EncogMLP3XORSigmoid2Stdr()

        '    TestMLP3XORSigmoid2(m_mlp, nbIterations:=10000, learningMode:=enumLearningMode.Vectorial)

        'End Sub

        <TestMethod()>
        Public Sub EncogMLP3XORTanhStdr()

            TestMLP3XORTanh(m_mlp, gain:=2,
                learningMode:=enumLearningMode.Vectorial)

        End Sub

        '<TestMethod()>
        'Public Sub EncogMLP3XORTanh2Stdr()

        '    TestMLP3XORTanh2(m_mlp, gain:=2, nbIterations:=10000,
        '        learningMode:=enumLearningMode.Vectorial)

        'End Sub

        '<TestMethod()>
        'Public Sub EncogMLP3XORGaussianStdr()

        '    TestMLP3XORGaussian(m_mlp, nbIterations:=10000,
        '        learningMode:=enumLearningMode.Vectorial)

        'End Sub

        '<TestMethod()>
        'Public Sub EncogMLP3XORSinusStdr()

        '    TestMLP3XORSinus(m_mlp, nbIterations:=20000,
        '        learningMode:=enumLearningMode.Vectorial)

        'End Sub

        <TestMethod()>
        Public Sub EncogMLP3XORSinus()

            Init3XOR()

            m_mlp.nbIterations = 3000
            m_mlp.SetActivationFunction(enumActivationFunction.Sinus)

            m_mlp.InitializeWeights(1, {
                {-0.02, -0.99, -0.16, -0.12, -0.73, 0.79, 0.65},
                {0.04, 0.83, -0.42, -0.23, -0.7, 0.44, 0.1},
                {-0.25, -0.86, 0.45, 0.15, -0.3, -0.53, -0.62},
                {-0.16, -0.79, 0.56, 0.15, -0.38, 0.9, 0.81},
                {0.19, 0.25, 0.41, -0.75, 0.61, -0.96, 0.29},
                {-0.59, 0.95, 0.44, -0.21, 0.05, -0.69, -0.7},
                {0.26, 0.68, -0.07, 0.53, -0.8, -0.72, -0.08}})
            m_mlp.InitializeWeights(2, {
                {0.0, -0.03, -0.91, 0.69, 0.73, 0.45, -0.1, 0.14},
                {-0.71, 0.01, -0.72, -0.05, 0.73, 0.6, 0.48, -0.02},
                {-0.22, 0.44, 0.6, 0.62, 0.53, 0.64, 0.06, 0.66}})

            m_mlp.Train(enumLearningMode.Vectorial)

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub EncogMLPIrisFlowerAnalogTanh2()

            ' 96.7% prediction, 98.3% learning with 50 iterations in 9 msec.

            TestMLPIrisFlowerAnalogTanh2(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub EncogMLPIrisFlowerAnalogTanh()

            ' 96.7% prediction, 98.3% learning with 300 iterations in 40 msec.

            TestMLPIrisFlowerAnalogTanh(m_mlp,
                learningMode:=enumLearningMode.Vectorial, nbIterations:=300)

        End Sub

        <TestMethod()>
        Public Sub EncogMLPIrisFlowerLogicalStdr()

            ' 97.8% prediction, 99.4% learning with 500 iterations in 3.7 sec.

            TestMLPIrisFlowerLogicalTanh(m_mlp, nbIterations:=500)

        End Sub

        <TestMethod()>
        Public Sub EncogMLPIrisFlowerLogical3L()

            ' 95.6% prediction, 96.4% learning with 1500 iterations in 74 msec.

            InitIrisFlowerLogical4Layers(m_mlp)
            m_mlp.InitializeStruct(m_neuronCountIrisFlowerLogical443, addBiasColumn:=True)

            m_mlp.nbIterations = 1500
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent)

            m_mlp.InitializeWeights(1, {
                {-0.51, -0.75, -0.43, -0.24, -0.09},
                {0.25, -0.08, -0.48, 0.33, -0.99},
                {0.1, 0.24, -0.68, -0.58, 0.36},
                {0.25, -0.57, 0.15, -0.37, 0.44}})
            m_mlp.InitializeWeights(2, {
                {0.11, -0.84, 0.99, -0.15, 0.69},
                {0.23, 0.81, 0.42, -0.53, -0.47},
                {0.97, 0.98, -0.16, 0.09, -0.25}})

            m_mlp.minimalSuccessTreshold = 0.3
            m_mlp.Train(learningMode:=enumLearningMode.Vectorial)

            Dim expectedSuccess# = 0.964
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.052
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            If m_mlp.successPC = 1 AndAlso m_mlp.minimalSuccessTreshold <= 0.05! Then
                Dim expectedOutput = m_targetArrayIrisFlowerLogical
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
        Public Sub EncogMLPIrisFlowerLogical4L()

            ' 97.8% prediction, 100% learning with 1500 iterations in 280 msec.

            InitIrisFlowerLogicalOriginal()
            m_mlp.InitializeStruct(m_neuronCountIrisFlowerLogical4_16_83, addBiasColumn:=True)

            'm_mlp.minimalSuccessTreshold = 0.05
            'm_mlp.nbIterations = 25000 '5000

            m_mlp.nbIterations = 1500
            m_mlp.minimalSuccessTreshold = 0.3

            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent)

            m_mlp.InitializeWeights(1, {
                {0.05, 0.55, -0.74, -0.84, 0.03},
                {-0.62, 0.31, 0.51, -0.1, 0.18},
                {-0.11, -0.38, 0.23, -0.9, 0.2},
                {-0.01, -0.88, 0.94, -0.95, -0.12},
                {0.7, 0.05, -0.04, 0.43, -0.33},
                {0.01, 0.24, -0.72, 0.87, 0.85},
                {0.17, 0.65, 0.56, -0.1, 0.15},
                {0.19, -0.88, 0.25, 0.82, 0.27},
                {-0.55, 0.51, 0.37, -0.27, 0.75},
                {0.6, -0.39, -0.55, -0.22, 0.37},
                {-0.11, 0.3, -0.99, 0.48, -0.24},
                {0.46, -0.67, 0.31, 0.09, -0.4},
                {-0.61, 0.64, -0.34, 0.71, 0.23},
                {0.9, -0.12, -0.81, -0.17, -0.66},
                {0.79, -0.42, 0.81, 0.55, 0.45},
                {-0.89, 0.71, -0.39, 0.57, 0.31}})
            m_mlp.InitializeWeights(2, {
                {-0.19, -0.38, -0.1, 0.75, 0.32, -0.1, 0.83, -0.25, -0.17, 0.64, -0.63, 0.84, 0.55, -0.65, -0.8, -0.5, 0.51},
                {-0.31, 0.41, -0.57, -0.16, -0.84, -0.18, -0.77, 0.87, -0.17, 0.79, 0.03, -0.6, 0.94, 0.63, -0.79, 0.52, 0.9},
                {-0.82, 0.21, -0.71, -0.7, -0.72, -0.74, 0.71, -0.01, -0.14, -0.42, -0.78, -0.73, -0.58, 0.16, 0.57, -0.2, -0.38},
                {-0.26, -0.82, 0.48, -0.94, 0.02, 0.25, -0.2, 0.94, -0.36, -0.83, -0.11, -0.22, 0.81, -0.73, 0.19, 0.8, -0.28},
                {-0.22, -0.2, -0.01, 0.11, -0.34, -0.36, -0.94, -0.78, -0.8, -0.34, -0.42, 0.83, -0.89, 0.4, -1.0, -0.75, -0.73},
                {-0.47, -0.52, -0.07, -0.71, -0.18, -0.03, 0.98, -0.78, -0.48, -0.31, 0.79, -0.15, 0.83, -0.84, -0.62, 0.09, 0.86},
                {-0.72, 0.54, 0.94, -0.42, -0.21, -0.53, -0.67, 0.27, 0.88, 0.67, -0.67, -0.73, 0.07, 0.22, 0.74, -0.75, -0.33},
                {-0.61, -0.47, -0.69, -0.61, -0.5, 0.27, 0.88, -0.68, -0.92, -0.22, 0.35, -0.32, 0.32, -0.95, 0.04, -0.21, -0.46}})
            m_mlp.InitializeWeights(3, {
                {0.96, -0.04, 0.95, -0.78, 0.28, 0.17, 0.31, 0.27, 0.37},
                {-0.09, -0.37, 0.68, -0.42, 0.08, -0.8, -0.4, 0.07, -0.63},
                {-0.46, -0.94, 0.18, 0.62, -0.32, 0.38, 0.03, 0.26, -0.24}})

            m_mlp.Train(learningMode:=enumLearningMode.Vectorial)

            Dim expectedSuccess# = 1
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.013
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
            Dim expectedSuccessPrediction# = 0.978 ' 280 msec.
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub EncogMLPIrisFlowerLogical4L4883()

            ' 97.8% prediction, 100% learning with 2500 iterations in 309 msec. (original)
            ' 96.7% prediction, 100% learning with 2500 iterations in 309 msec. (corrected)

            InitIrisFlowerLogicalOriginal()
            m_mlp.InitializeStruct(m_neuronCountIrisFlowerLogical4883, addBiasColumn:=True)

            'm_mlp.minimalSuccessTreshold = 0.05
            'm_mlp.nbIterations = 15000

            m_mlp.nbIterations = 2500
            m_mlp.minimalSuccessTreshold = 0.3

            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent)

            m_mlp.InitializeWeights(1, {
                {-0.09, 0.08, -0.13, 0.93, 0.72},
                {-0.48, -0.98, 0.18, -0.99, 0.69},
                {0.41, 0.1, 0.21, -0.92, 0.78},
                {0.47, 0.36, -0.16, -0.32, 0.62},
                {0.11, -0.3, -0.23, 0.29, -0.42},
                {-0.9, 0.34, 0.5, -0.43, 0.53},
                {0.24, -0.3, -0.13, 0.09, -0.18},
                {0.42, -0.42, -0.95, -0.32, -0.76}})
            m_mlp.InitializeWeights(2, {
                {-0.11, 0.34, 0.65, -0.08, -0.64, -0.34, 0.92, 0.01, -0.52},
                {0.14, -0.53, 0.35, -0.58, -0.03, -0.57, -0.63, -0.35, 0.64},
                {0.15, -0.02, 0.54, 0.64, 0.95, 0.34, 0.97, 0.24, 0.21},
                {-0.84, 0.57, 0.1, 0.01, 0.63, -0.18, 0.23, 0.15, -0.14},
                {0.57, 0.82, -0.98, 0.56, -0.65, 0.59, 0.99, 0.47, 0.92},
                {0.1, 0.87, -0.44, 0.07, -0.29, 0.52, 0.72, -0.66, -0.86},
                {0.34, 0.77, -0.87, 1.0, -0.52, -0.14, -0.78, -0.92, 0.05},
                {0.65, 0.4, -0.49, 0.34, -0.18, 0.85, -0.66, -0.74, -0.06}})
            m_mlp.InitializeWeights(3, {
                {-0.13, 0.59, -0.09, 0.79, 0.1, -0.36, -0.38, 0.33, -0.65},
                {-0.28, 0.21, 0.4, 0.08, -0.23, -0.25, 0.7, 0.36, 0.08},
                {-0.57, 0.31, 0.56, 0.19, 0.35, 0.61, 0.48, -0.67, -0.66}})

            m_mlp.Train(learningMode:=enumLearningMode.Vectorial)

            Dim expectedSuccess# = 1
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.014
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
            Dim expectedSuccessPrediction# = 0.967 '0.978 ' 309 msec.
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub EncogMLPIrisFlowerLogical4L4773()

            ' 97.8% prediction, 100% learning with 3500 iterations in 383 msec.

            InitIrisFlowerLogicalOriginal()
            m_mlp.InitializeStruct(m_neuronCountIrisFlowerLogical4773, addBiasColumn:=True)

            'm_mlp.minimalSuccessTreshold = 0.05
            'm_mlp.nbIterations = 6000

            m_mlp.nbIterations = 3500
            m_mlp.minimalSuccessTreshold = 0.3

            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent)

            m_mlp.InitializeWeights(1, {
                {0.37, -0.96, -0.19, 0.68, -0.51},
                {0.65, -0.73, -0.8, 0.74, 0.83},
                {-0.16, -0.3, -0.92, -0.87, -0.25},
                {0.75, 0.6, 0.3, -0.54, -0.06},
                {-0.12, 0.08, -0.33, 0.29, 0.94},
                {0.29, 0.03, 0.91, 0.9, 0.98},
                {0.56, -0.16, 0.34, -0.29, 0.1}})
            m_mlp.InitializeWeights(2, {
                {0.76, -0.8, -0.31, 0.91, -0.56, -0.48, -0.12, 0.95},
                {0.68, -0.14, 0.88, -0.29, 0.95, 0.42, 0.95, -0.16},
                {-0.33, -0.81, 0.91, 0.29, -0.34, 0.01, -0.44, 0.97},
                {-0.71, 0.9, -0.03, 0.59, 0.14, -0.13, 0.9, 0.74},
                {0.8, 0.39, -0.81, 0.39, 0.05, 0.39, -0.28, -0.84},
                {-0.49, 0.18, 0.82, -0.14, 0.73, -0.65, -0.57, 0.42},
                {-0.01, -0.73, 0.68, 0.55, 0.05, 0.49, 0.8, 0.85}})
            m_mlp.InitializeWeights(3, {
                {-0.03, -0.6, 0.76, -0.24, 0.22, -0.1, 0.47, -0.24},
                {0.38, 0.66, -0.09, -0.66, -0.83, -0.9, -0.81, -0.27},
                {-0.3, 0.44, 0.53, 0.45, -0.54, -0.2, 0.03, 0.69}})

            m_mlp.Train(learningMode:=enumLearningMode.Vectorial)

            Dim expectedSuccess# = 1
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.01
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
            Dim expectedSuccessPrediction# = 0.978 ' 383 msec.
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub EncogMLPIrisFlowerLogical4L4663()

            ' 97.8% prediction, 100% learning with 3500 iterations in 350 msec.

            InitIrisFlowerLogicalOriginal()
            m_mlp.InitializeStruct(m_neuronCountIrisFlowerLogical4663, addBiasColumn:=True)

            'm_mlp.minimalSuccessTreshold = 0.05
            'm_mlp.nbIterations = 30000 ' 99.7 %

            m_mlp.nbIterations = 3500
            m_mlp.minimalSuccessTreshold = 0.3

            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent)

            m_mlp.InitializeWeights(1, {
                {-0.28, 0.53, -0.5, -0.67, -0.4},
                {-0.53, 0.0, 0.62, -0.81, -0.47},
                {0.47, 0.78, -0.09, -0.16, -0.59},
                {0.55, -0.11, 0.67, -0.36, 0.81},
                {-0.63, -0.94, -0.38, -0.27, 0.14},
                {0.0, 0.99, 0.57, -0.23, -0.96}})
            m_mlp.InitializeWeights(2, {
                {-0.04, -0.25, -0.44, 0.66, -0.24, -0.41, -0.77},
                {-0.89, 0.91, -0.21, 0.27, -0.92, 0.72, 0.12},
                {0.75, 0.14, -0.37, -0.59, 0.74, 0.86, 0.03},
                {0.38, 0.08, 0.16, 0.77, -0.3, 0.75, 0.64},
                {-0.41, -0.53, 0.95, -0.04, -0.05, 0.93, -0.09},
                {0.27, 0.32, 0.9, -0.42, 0.84, 0.96, -0.29}})
            m_mlp.InitializeWeights(3, {
                {-0.57, 0.52, -0.86, -0.49, 0.63, 0.17, 0.53},
                {0.25, -0.92, 0.44, -0.85, 0.2, -0.38, -0.95},
                {0.72, 0.94, -0.42, 0.86, 0.14, -0.49, -0.04}})

            m_mlp.Train(learningMode:=enumLearningMode.Vectorial)

            Dim expectedSuccess# = 1
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.011
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
            Dim expectedSuccessPrediction# = 0.978 ' 350 msec.
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub EncogMLPIrisFlowerLogical4LSigmoid()

            ' 94.4% prediction, 99.4% learning with 400 iterations in 74 msec.

            InitIrisFlowerLogical4Layers(m_mlp)

            'm_mlp.minimalSuccessTreshold = 0.05
            'm_mlp.nbIterations = 25000 '5000

            m_mlp.nbIterations = 400
            m_mlp.minimalSuccessTreshold = 0.3

            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {0.48, -0.49, 0.12, 0.45, 0.25},
                {-0.25, 0.76, -0.74, 0.29, -0.96},
                {-0.79, 0.86, -0.13, -0.64, -0.8},
                {0.55, -0.76, 0.5, 0.57, -0.27},
                {0.11, 0.55, 0.96, 0.62, -0.29},
                {0.21, 0.84, 0.15, 0.48, 0.84},
                {-0.36, 0.15, 0.59, -0.39, 0.81},
                {0.28, -0.22, 0.93, 0.64, 0.36},
                {-0.14, -0.9, 0.89, 0.61, -0.84},
                {0.51, 0.4, -0.16, 0.43, 0.94},
                {0.91, -0.96, -0.98, 0.45, -0.65},
                {0.32, 0.05, 1.0, 0.7, -0.29},
                {-0.01, 0.28, -0.58, 0.17, 0.82},
                {0.58, 0.62, -0.11, 0.59, 0.15},
                {0.34, -0.74, -0.93, -1.0, -0.65},
                {0.77, 0.87, -0.43, 0.24, 0.21}})
            m_mlp.InitializeWeights(2, {
                {0.33, 0.65, -0.16, 0.48, 0.08, 0.78, 0.18, 0.61, 0.4, -0.21, 0.34, 0.71, -0.93, 0.73, -0.68, 0.38, -0.07},
                {0.47, 0.23, -0.22, 0.0, -0.77, -0.43, -0.95, 0.99, 0.43, -0.81, 0.13, -0.63, 0.32, 0.05, -0.68, -0.67, 0.39},
                {0.6, 0.89, -0.25, 0.4, -0.63, 0.08, 0.88, -0.49, 0.44, -0.4, -0.49, -0.92, -0.92, -0.76, -0.08, 0.76, -0.96},
                {0.99, -0.48, -0.61, 0.68, 0.69, 0.48, 0.03, -0.74, 0.47, -0.88, 0.09, 1.0, -0.49, -0.79, -0.84, 0.15, -0.06},
                {0.26, 0.7, -0.31, 0.14, 0.99, 0.94, -0.6, 0.29, 0.52, -0.74, -0.98, 0.51, 0.85, 0.36, -0.93, -0.01, -0.7},
                {-0.3, 0.89, 0.34, 0.89, -0.99, 0.29, 0.97, -0.43, 0.07, -0.62, -0.21, 0.73, -0.81, -0.59, 0.41, -0.68, 0.16},
                {-0.74, -0.24, 0.86, 0.82, 0.99, 0.66, -0.47, -0.5, -0.69, 0.67, -0.94, 0.26, -0.25, 0.73, 0.74, 0.32, -0.66},
                {0.63, -0.7, 0.61, -0.54, -0.44, -0.73, 0.99, 0.76, -0.18, -0.81, 0.11, 0.21, -0.89, 0.23, -0.39, 0.06, 0.15}})
            m_mlp.InitializeWeights(3, {
                {-0.61, -0.03, 0.75, -0.15, -0.19, -0.34, 0.13, 0.6, 0.74},
                {-0.6, -0.45, 0.29, 0.68, 0.1, 0.58, 0.52, 0.12, 0.18},
                {0.36, 0.85, -0.36, 0.19, 0.14, 0.47, -0.57, 0.42, 0.28}})

            m_mlp.Train(learningMode:=enumLearningMode.Vectorial)

            Dim expectedSuccess# = 0.994
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.009
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
            Dim expectedSuccessPrediction# = 0.944 ' 74 msec.
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub EncogMLPIrisFlowerLogical4LSigmoid3()

            ' 97.8% prediction, 93.3% learning with 300 iterations in 57 msec.

            InitIrisFlowerLogical4Layers(m_mlp)

            'm_mlp.minimalSuccessTreshold = 0.05
            'm_mlp.nbIterations = 25000 '5000

            m_mlp.nbIterations = 300
            m_mlp.minimalSuccessTreshold = 0.3

            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {-0.5, -0.23, -0.58, 0.86, -0.13},
                {-0.48, 0.38, -0.77, -0.1, 0.53},
                {0.69, -0.35, -0.14, -0.54, 0.91},
                {-0.48, 0.73, 0.43, -0.38, -0.18},
                {-0.96, -0.95, 0.94, 0.37, 0.31},
                {0.11, -0.02, -0.96, -0.93, -0.06},
                {-0.86, 0.11, 0.49, -0.52, -0.49},
                {0.89, 0.69, -0.85, 0.52, -0.27},
                {0.8, 0.31, -0.52, 0.41, 0.89},
                {-0.46, 0.5, 0.46, 1.0, -0.51},
                {0.24, -0.75, -0.35, 0.2, 0.25},
                {-0.67, -0.12, -0.64, -0.46, -0.41},
                {0.05, 0.47, -0.25, -0.14, -0.67},
                {-0.83, -0.93, -0.09, 0.25, -0.26},
                {-0.64, 0.67, 0.98, -0.59, -0.06},
                {-0.55, 0.71, -0.5, 0.73, -0.59}})
            m_mlp.InitializeWeights(2, {
                {0.52, 0.74, 0.05, -0.61, 0.06, 0.19, 0.2, -0.43, -0.86, -0.19, 0.83, 0.38, -0.97, 0.02, -0.89, 0.93, -0.86},
                {0.76, 0.34, -0.08, 0.73, -0.2, 0.57, -0.97, -0.89, 0.34, 0.67, 0.81, -0.33, 0.23, -0.67, -0.64, 0.55, -0.7},
                {0.03, -0.99, 0.49, -0.26, 0.02, 0.08, 0.54, -0.82, -0.11, -0.97, 0.68, -0.47, 0.84, -0.4, -0.96, -0.34, 0.4},
                {-0.43, 0.94, 0.22, -0.12, 0.1, 0.98, 0.4, 0.88, -0.96, 0.43, -0.21, -0.88, 0.58, 0.59, -0.5, 0.26, -0.15},
                {0.61, -0.16, -0.01, -0.19, -0.78, -0.08, 0.25, -0.69, 0.93, -0.92, -0.64, 0.66, 0.68, 0.69, 0.79, 0.67, 0.37},
                {0.04, -0.05, 0.16, 0.9, 0.84, -0.43, 0.72, 0.46, -0.8, 0.21, 0.79, -0.76, 0.86, 0.99, -0.25, -0.11, -0.58},
                {-0.66, 0.3, 0.19, 0.19, -0.88, 0.9, 0.41, 0.54, 0.00, 1.0, 0.42, 0.52, 0.00, -0.73, 0.49, 0.55, -0.85},
                {-0.65, 0.23, -0.08, -0.33, 0.83, -0.07, -0.65, 0.85, 0.21, 0.82, 0.34, 0.73, 0.00, 0.36, -0.65, -0.44, 0.98}})
            m_mlp.InitializeWeights(3, {
                {-0.19, -0.97, 0.65, 0.16, -0.8, 0.28, 0.65, 0.74, -0.22},
                {-0.32, 0.2, 0.4, -0.99, 0.02, 0.74, -0.87, 0.33, -0.97},
                {0.77, 0.83, -0.6, 0.3, 0.94, 0.49, 0.2, -0.86, 0.44}})

            m_mlp.Train(learningMode:=enumLearningMode.Vectorial)

            Dim expectedSuccess# = 0.933
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.065
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
            Dim expectedSuccessPrediction# = 0.978 ' 57 msec.
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub EncogMLPIrisFlowerLogical4LSigmoid2()

            ' 97.8% prediction, 97.2% learning with 200 iterations in 103 msec.
            TestMLPIrisFlowerLogicalSigmoid(m_mlp,
                nbIterations:=500, expectedSuccess:=0.972)

        End Sub

        <TestMethod()>
        Public Sub EncogMLPIrisFlowerAnalogSigmoidStdr()

            ' 93.3% prediction, 96.7% learning with 200 iterations in 103 msec.
            TestMLPIrisFlowerAnalogSigmoid(m_mlp,
                expectedSuccess:=0.967, expectedLoss:=0.026,
                expectedSuccessPrediction:=0.933)

        End Sub

        <TestMethod()>
        Public Sub EncogMLPIrisFlowerAnalogSigmoid()

            ' 96.7% prediction, 95.8% learning with 250 iterations in x msec.

            InitIrisFlowerAnalog4Layers(m_mlp)
            m_mlp.InitializeStruct({4, 12, 10, 1}, addBiasColumn:=True)

            m_mlp.nbIterations = 250 '1500
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {0.09, 0.48, 0.41, -0.36, -0.43},
                {-0.19, 0.01, -0.28, -0.33, 0.17},
                {0.35, 0.25, -0.22, 0.18, -0.17},
                {-0.29, -0.4, -0.24, -0.39, 0.17},
                {0.08, 0.31, -0.22, 0.28, 0.29},
                {-0.33, -0.36, 0.03, 0.36, 0.41},
                {0.08, -0.2, -0.37, -0.17, -0.24},
                {0.05, 0.4, -0.36, -0.45, -0.29},
                {0.16, -0.22, -0.35, -0.26, 0.24},
                {-0.06, 0.22, -0.43, 0.33, 0.14},
                {-0.19, -0.5, 0.36, -0.48, -0.33},
                {0.13, 0.21, -0.15, 0.4, -0.33}})
            m_mlp.InitializeWeights(2, {
                {0.48, -0.14, -0.24, -0.4, 0.05, 0.12, 0.45, -0.07, 0.28, -0.19, 0.46, -0.3, 0.41},
                {-0.49, -0.21, 0.16, -0.46, -0.46, -0.16, -0.04, -0.44, 0.2, -0.28, -0.09, -0.24, -0.2},
                {-0.23, -0.19, -0.42, 0.42, -0.31, 0.29, -0.35, -0.47, 0.18, 0.42, 0.39, -0.35, -0.31},
                {0.32, 0.26, 0.15, -0.45, -0.17, 0.04, -0.39, -0.41, 0.35, 0.22, 0.01, 0.09, 0.18},
                {-0.49, -0.06, -0.39, -0.23, -0.37, 0.35, 0.33, -0.25, -0.15, 0.14, -0.16, 0.36, -0.38},
                {-0.33, -0.45, 0.38, -0.17, -0.13, 0.27, 0.39, 0.34, 0.02, 0.19, -0.08, 0.15, 0.4},
                {0.37, -0.35, -0.3, -0.08, 0.09, 0.07, -0.18, 0.28, 0.21, 0.42, -0.09, 0.29, -0.46},
                {-0.18, 0.44, -0.04, -0.38, -0.2, -0.31, -0.29, -0.13, -0.44, -0.28, 0.02, -0.15, 0.24},
                {0.21, 0.34, -0.01, 0.25, 0.19, 0.12, -0.26, 0.47, 0.18, -0.45, 0.44, -0.44, 0.27},
                {0.03, 0.11, -0.11, -0.16, 0.45, 0.42, 0.08, 0.23, 0.07, 0.4, -0.44, 0.07, -0.38}})
            m_mlp.InitializeWeights(3, {
                {-0.04, 0.19, -0.41, 0.48, 0.39, -0.27, 0.34, 0.12, 0.17, -0.21, 0.47}})

            m_mlp.minimalSuccessTreshold = 0.2

            'm_mlp.PrintWeights()
            'm_mlp.printOutput_ = True

            m_mlp.Train(learningMode:=enumLearningMode.Defaut)

            Dim expectedSuccess# = 0.958
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.025
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            If m_mlp.successPC = 1 AndAlso m_mlp.minimalSuccessTreshold <= 0.05! Then
                Dim expectedOutput = m_targetArrayIrisFlowerLogical
                Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
                Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
                Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
                Assert.AreEqual(sExpectedOutput, sOutput)
            End If

            m_mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
                m_targetArrayIrisFlowerAnalogTest, nbOutputs:=1)
            Dim expectedSuccessPrediction# = 0.967
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub EncogMLPSunspot1Sigmoid()

            ' 90.0% prediction, 95.9% learning with 100 iterations in 5 msec.

            TestMLPSunspot1Sigmoid(m_mlp, nbIterations:=100, expectedSuccess:=0.959,
                expectedSuccessPrediction:=0.9, expectedLoss:=0.04)

        End Sub

        <TestMethod()>
        Public Sub EncogMLPSunspot1Tanh()

            ' 80.0% prediction, 75% learning with 200 iterations in 9 msec.

            TestMLPSunspot1Tanh(m_mlp, expectedSuccess:=0.75, expectedSuccessPrediction:=0.8)

        End Sub

        <TestMethod()>
        Public Sub EncogMLPSunspot2Tanh()

            ' 95.2% prediction, 94.8% learning with 400 iterations in 33 msec.

            TestMLPSunspotTanh2(m_mlp, nbIterations:=400, expectedPredictionAccuracy:=0.952,
                expectedLearningAccuracy:=0.948, learningMode:=enumLearningMode.Vectorial)

        End Sub

    End Class

End Namespace