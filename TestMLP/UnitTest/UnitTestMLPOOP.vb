
Imports Perceptron
Imports Perceptron.Utility ' Matrix
'Imports Perceptron.clsMLPGeneric ' enumLearningMode
Imports Microsoft.VisualStudio.TestTools.UnitTesting

Namespace OOPMLP

    <TestClass()>
    Public Class clsMLPOOPTest

        Private m_mlp As New clsMLPOOP

        ' Weights are quite the same as MLP Classic, but not exactly:
        'Private m_mlp As New clsMLPClassic ' 13 success, 4 fails
        'Private m_mlp As New clsMLPAccord ' 10 success, 7 fails
        'Private m_mlp As New clsMLPTensorFlow  ' 1 success, 16 fails

        ' Weights are not stored in the same way:
        'Private m_mlp As New clsMPLMatrix ' 15/15 fails
        'Private m_mlp As New clsVectorizedMatrixMLP ' 15/15 fails
        'Private m_mlp As New clsMLPTensor ' 15/15 fails

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
        Public Sub MLPOOP1XOR4Layers()

            TestMLP1XOR4Layers(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP1XOR5Layers()

            TestMLP1XOR5Layers(m_mlp, nbIterations:=1500)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP1XORSemiStochastic()

            TestMLP1XORSemiStochastic(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP1XORStochastic()

            TestMLP1XORStochastic(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP1XORSigmoid()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New Sigmoid(alpha:=2.5)

            TestMLP1XORSigmoid(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP1XORSigmoidWithoutBias()

            TestMLP1XORWithoutBias(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP1XORSigmoidWithoutBias231()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New Sigmoid(alpha:=1)

            TestMLP1XORWithoutBias231(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP1XORSigmoidWithoutBias2()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New Sigmoid(alpha:=1, center:=2.2)

            InitXOR()
            m_mlp.Initialize(learningRate:=0.9!, weightAdjustment:=0.05!)
            m_mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=False)

            m_mlp.nbIterations = 60000
            m_mlp.SetActivationFunction(
                enumActivationFunction.Sigmoid, center:=2.2!)

            m_mlp.InitializeWeights(1, {
                {0.66, 0.53},
                {0.65, 0.69},
                {0.82, 0.56}})
            m_mlp.InitializeWeights(2, {
                {0.62, 0.54, 0.5}})

            m_mlp.Train()

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
        Public Sub MLPOOP1XORTanh()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New HyperbolicTangent(alpha:=2.4#)

            TestMLP1XORTanh(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP1XORTanh261()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New HyperbolicTangent(alpha:=2.0#)

            TestMLP1XORTanh261(m_mlp, nbIterations:=500)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP1XORELU()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New ELU(alpha:=0.6!, center:=0.3!)

            TestMLP1XORELU(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP2XORSigmoid()

            TestMLP2XORSigmoid(m_mlp, nbIterations:=6000)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP2XORTanh()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New HyperbolicTangent(alpha:=2.0#)

            TestMLP2XORTanh(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP2XORTanh2()

            TestMLP2XORTanh2(m_mlp, nbIterations:=600)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP2XORTanh462()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New HyperbolicTangent(alpha:=2.0#)

            TestMLP2XORTanh462(m_mlp, nbIterations:=9000)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP3XORSigmoidStdr()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New Sigmoid(alpha:=2)

            TestMLP3XORSigmoid(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP3XORSigmoidStdr2()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New Sigmoid(alpha:=1)

            TestMLP3XORSigmoid2(m_mlp, nbIterations:=1500)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP3XORTanh()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New HyperbolicTangent(alpha:=1.0#)

            TestMLP3XORTanh(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP3XORTanh2()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New HyperbolicTangent(alpha:=1.0#)

            TestMLP3XORTanh2(m_mlp, nbIterations:=10000)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP3XORGaussianStdr()

            TestMLP3XORGaussian(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP3XORSinusStdr()

            TestMLP3XORSinus(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP3XORELUStdr() ' 166 msec

            TestMLP3XORELU(m_mlp, nbIterations:=400)

        End Sub

        '<TestMethod()>
        'Public Sub MLPOOP3XORMishStdr()

        '    TestMLP3XORMish(m_mlp)

        'End Sub

        <TestMethod()>
        Public Sub MLPOOP3XORMish()

            Init3XOR()
            m_mlp.Initialize(learningRate:=0.01!, weightAdjustment:=0.005!)

            m_mlp.nbIterations = 1800
            m_mlp.SetActivationFunction(enumActivationFunction.Mish)

            m_mlp.InitializeWeights(1, {
                {0.28, 0.23, 0.13, 0.18, 0.05, 0.07, 0.16},
                {0.21, 0.07, 0.25, 0.27, 0.04, 0.1, 0.12},
                {0.04, 0.12, 0.26, 0.12, 0.26, 0.22, 0.19},
                {0.05, 0.03, 0.07, 0.14, 0.14, 0.01, 0.14},
                {0.26, 0.06, 0.26, 0.09, 0.04, 0.03, 0.05},
                {0.15, 0.03, 0.22, 0.05, 0.26, 0.04, 0.03}})
            m_mlp.InitializeWeights(2, {
                 {0.17, 0.03, 0.23, 0.13, 0.26, 0.29, 0.19},
                 {0.11, 0.21, 0.12, 0.1, 0.14, 0.01, 0.1},
                 {0.2, 0.03, 0.02, 0.1, 0.12, 0.23, 0.03}})

            m_mlp.Train()

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
        Public Sub MLPOOP3XORGaussian()

            Init3XOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.09!)

            m_mlp.nbIterations = 150
            m_mlp.SetActivationFunction(enumActivationFunction.Gaussian, center:=1)

            m_mlp.InitializeWeights(1, {
                {0.64, 0.03, 0.82, 0.43, 0.84, 0.69, 0.69},
                {0.63, 0.08, 0.77, 0.86, 0.23, 0.16, 0.71},
                {0.56, 0.7, 0.63, 0.51, 0.14, 0.5, 0.39},
                {0.86, 0.98, 0.6, 0.64, 0.28, 0.05, 0.69},
                {0.13, 0.83, 0.03, 0.1, 0.68, 0.36, 0.45},
                {1.0, 0.31, 0.75, 0.49, 0.25, 0.03, 0.16}})
            m_mlp.InitializeWeights(2, {
                {0.52, 0.31, 0.46, 0.99, 0.34, 0.07, 0.86},
                {0.42, 0.57, 0.38, 0.72, 1.0, 0.03, 0.75},
                {0.29, 0.32, 0.14, 0.32, 0.53, 0.91, 0.41}})

            m_mlp.Train()

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
        Public Sub MLPOOPIrisFlowerAnalogSigmoid()

            ' 93.3% prediction, 94.2% learning with 150 iterations in 315 msec.

            TestMLPIrisFlowerAnalogSigmoid(m_mlp, expectedSuccess:=0.942,
                expectedSuccessPrediction:=0.933)

        End Sub

        <TestMethod()>
        Public Sub MLPOOPIrisFlowerAnalogTanh()

            ' 96.7% prediction, 96.7% learning with 200 iterations in 380 msec.

            TestMLPIrisFlowerAnalogTanh(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPOOPIrisFlowerLogicalSinus()

            ' 97.8% prediction, 93.9% learning with 200 iterations in 410 msec.

            TestMLPIrisFlowerLogicalSinus(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPOOPIrisFlowerLogicalTanh()

            ' 97.8% prediction, 98.9% learning with 1200 iterations in 3.7 sec.

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New HyperbolicTangent(alpha:=2.0#)

            TestMLPIrisFlowerLogicalTanh(m_mlp,
                nbIterations:=1200, expectedSuccess:=0.989#)

        End Sub

        <TestMethod()>
        Public Sub MLPOOPIrisFlowerLogicalSigmoid()

            ' 97.8% prediction, 98.6% learning with 800 iterations in 2.3 sec.

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New Sigmoid(alpha:=1.0#)

            TestMLPIrisFlowerLogicalSigmoid(m_mlp, nbIterations:=800)

        End Sub

        <TestMethod()>
        Public Sub MLPOOPSunspot1Sigmoid()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New Sigmoid(alpha:=1.0#)

            ' 90.0% prediction, 73.5% learning with 400 iterations in 220 msec.

            TestMLPSunspot1Sigmoid(m_mlp, nbIterations:=400, expectedSuccess:=0.735,
                expectedSuccessPrediction:=0.9, expectedLoss:=0.07)

        End Sub

        <TestMethod()>
        Public Sub MLPOOPSunspot1Tanh()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New HyperbolicTangent(alpha:=1.0#)

            ' 90.0% prediction, 75% learning with 200 iterations in 105 msec.

            TestMLPSunspot1Tanh(m_mlp, expectedSuccess:=0.75, expectedSuccessPrediction:=0.9)

        End Sub

        <TestMethod()>
        Public Sub MLPOOPSunspot2Tanh()

            ' 92.8% prediction, 91.6% learning with 200 iterations in 210 msec.

            TestMLPSunspotTanh2(m_mlp, nbIterations:=200, expectedSuccess:=0.695,
                expectedLearningAccuracy:=0.916, expectedLoss:=0.084,
                expectedPredictionAccuracy:=0.928, expectedSuccessPrediction:=0.75)

        End Sub

    End Class

End Namespace