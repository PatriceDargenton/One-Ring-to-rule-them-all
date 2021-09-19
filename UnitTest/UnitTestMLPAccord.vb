
Imports Perceptron
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPClassic ' enumLearningMode

Namespace AccordMLP

    <TestClass()>
    Public Class clsMLPAccordTest

        Private m_mlp As New clsMLPAccord

        'Private m_mlp As New clsMLPClassic ' 18/18 success
        'Private m_mlp As New clsMLPOOP ' 14 success, 4 fails
        'Private m_mlp As New clsMLPEncog ' 8 success, 10 fails
        'Private m_mlp As New clsMLPTensorFlow  ' 0 success, 18 fails

        <TestInitialize()>
        Public Sub Init()
        End Sub

        Private Sub InitXOR()
            m_mlp.Initialize(learningRate:=0.01!)
            m_mlp.inputArray = m_inputArrayXOR
            m_mlp.targetArray = m_targetArrayXOR
        End Sub

        Private Sub Init2XOR()
            m_mlp.Initialize(learningRate:=0.01!)
            m_mlp.inputArray = m_inputArray2XOR
            m_mlp.targetArray = m_targetArray2XOR
        End Sub

        Private Sub Init3XOR()
            m_mlp.Initialize(learningRate:=0.01!)
            m_mlp.inputArray = m_inputArray3XOR
            m_mlp.targetArray = m_targetArray3XOR
        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XORSemiStochastic()

            TestMLP1XORSemiStochastic(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XORStochastic()

            TestMLP1XORStochastic(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XOR4Layers()

            TestMLP1XOR4Layers(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XOR4LayersVect()

            TestMLP1XOR4Layers(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XOR5Layers()

            TestMLP1XOR5Layers(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XOR5LayersVect()

            TestMLP1XOR5Layers(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XORSigmoidStdr()

            TestMLP1XORSigmoid(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XORSigmoidVect()

            TestMLP1XORSigmoid(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XORSigmoidRProp()

            TestMLP1XORSigmoidRProp(m_mlp, trainingAlgorithm:=enumTrainingAlgorithm.RProp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XORTanh()

            TestMLP1XORTanh(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XORTanhVect()

            TestMLP1XORTanh(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XORTanhRProp()

            'm_mlp.RBPLAlgo = True
            TestMLP1XORTanhRProp(m_mlp, nbIterations:=350, trainingAlgorithm:=enumTrainingAlgorithm.RProp)
            'TestMLP1XORTanhRProp(m_mlp, nbIterations:=350, learningMode:=enumLearningMode.VectorialBatch)
            'm_mlp.RBPLAlgo = False

        End Sub

        ' useBias=False not implemented
        '<TestMethod()>
        'Public Sub MLP1XORTanh261()

        '    TestMLP1XORTanh261(m_mlp, nbIterations:=500,
        '        learningMode:=enumLearningMode.Vectorial)

        'End Sub

        <TestMethod()>
        Public Sub AccordMLP2XORSigmoid()

            TestMLP2XORSigmoid(m_mlp, nbIterations:=6000)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP2XORTanh()

            TestMLP2XORTanh(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP2XORTanhVect()

            TestMLP2XORTanh(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP2XORTanh2()

            TestMLP2XORTanh2(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP2XORTanh2Vect()

            TestMLP2XORTanh2(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        ' useBias=False not implemented
        '<TestMethod()>
        'Public Sub AccordMLP2XORTanh462()

        '    TestMLP2XORTanh462(m_mlp)

        'End Sub

        <TestMethod()>
        Public Sub AccordMLP2XORTanhVectRProp()

            Init2XOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.05!)

            m_mlp.nbIterations = 700
            m_mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.RProp
            m_mlp.PRBPLAlgo = True
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=1.9!)

            m_mlp.InitializeWeights(1, {
                {-0.08, 0.24, 0.67, 0.65, 0.21},
                {-0.68, 0.38, 0.41, -0.45, -0.05},
                {0.52, 0.7, 0.11, 0.31, 0.34},
                {0.35, 0.58, 0.58, -0.33, 0.29}})
            m_mlp.InitializeWeights(2, {
                {0.76, 0.13, -0.57, 0.2, -0.16},
                {-0.78, 0.18, -0.3, 0.18, -0.47}})

            'm_mlp.TrainVector()
            m_mlp.Train(learningMode:=enumLearningMode.Vectorial)
            'm_mlp.Train() Does not work

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
        Public Sub AccordMLP3XORSigmoid()

            TestMLP3XORSigmoid(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP3XORSigmoidVect()

            TestMLP3XORSigmoid(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP3XORSigmoid2()

            TestMLP3XORSigmoid2(m_mlp, nbIterations:=1600)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP3XORTanh()

            TestMLP3XORTanh(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP3XORTanhVect()

            TestMLP3XORTanh(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP3XORTanh2()

            TestMLP3XORTanh2(m_mlp, nbIterations:=11000)

        End Sub

        <TestMethod()>
        Public Sub AccordMLPIrisFlowerAnalogTanh()

            ' 96.7% prediction, 96.7% learning with 200 iterations in 45 msec.

            TestMLPIrisFlowerAnalogTanh(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLPIrisFlowerAnalogTanh2()

            ' 96.7% prediction, 98.3% learning with 900 iterations in 200 msec.

            TestMLPIrisFlowerAnalogTanh2(m_mlp, nbIterations:=900)

        End Sub

        <TestMethod()>
        Public Sub AccordMLPIrisFlowerAnalogSigmoid()

            ' 96.7% prediction, 95.8% learning with 150 iterations in 23 msec.

            TestMLPIrisFlowerAnalogSigmoid(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub AccordMLPIrisFlowerLogical4LTanh()

            ' 97.8% prediction, 99.4% learning with 1500 iterations in 500 msec.

            TestMLPIrisFlowerLogicalTanh(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLPIrisFlowerLogicalSigmoid()

            ' 97.8% prediction, 98.6% learning with 1000 iterations in 200 msec.

            TestMLPIrisFlowerLogicalSigmoid(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLPSunspot1Sigmoid()

            ' 90.0% prediction, 73.5% learning with 500 iterations in 40 msec.

            TestMLPSunspot1Sigmoid(m_mlp, nbIterations:=500, expectedSuccess:=0.735,
                expectedSuccessPrediction:=0.9, expectedLoss:=0.07)

        End Sub

        <TestMethod()>
        Public Sub AccordMLPSunspot1Tanh()

            ' 90.0% prediction, 75% learning with 200 iterations in 15 msec.

            TestMLPSunspot1Tanh(m_mlp, expectedSuccess:=0.75, expectedSuccessPrediction:=0.9)

        End Sub

        <TestMethod()>
        Public Sub AccordMLPSunspot2Tanh()

            ' 93.4% prediction, 93.1% learning with 100 iterations in 9 msec.

            TestMLPSunspotTanh2(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

    End Class

End Namespace