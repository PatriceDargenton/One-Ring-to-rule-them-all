
Imports Perceptron
Imports Perceptron.Utility ' Matrix
'Imports Perceptron.clsMLPClassic ' enumLearningMode

Namespace ClassicMLP

    <TestClass()>
    Public Class MultiLayerPerceptronTest

        Private m_mlp As New clsMLPClassic

        ' Weights are quite the same as MLP Classic, but not exactly:
        'Private m_mlp As New clsMLPOOP ' 33 success, 28 fails
        'Private m_mlp As New clsMLPAccord ' 22 success, 39 fails
        'Private m_mlp As New clsMLPEncog  ' 14 success, 47 fails
        'Private m_mlp As New clsMPLMatrix ' 10 success, 51 fails
        'Private m_mlp As New clsMLPRProp  ' 8 success, 53 fails
        'Private m_mlp As New clsMLPNeuralNet ' 5 success, 56 fails
        'Private m_mlp As New clsMLPTensorFlow ' 2 success, 59 fails

        ' Weights are not stored in the same way:
        'Private m_mlp As New clsVectorizedMatrixMLP ' 61/61 fails
        'Private m_mlp As New clsMLPTensor ' 61/61 fails

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

        Private Sub InitXORAnalog()
            m_mlp.inputArray = m_inputArrayXORAnalog
            m_mlp.targetArray = m_targetArrayXORAnalog
            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
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

#Region "Animated gifs"

        <TestMethod()>
        Public Sub MLP1XORGifTanh221()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-tanh-221.gif

            InitXOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 1000
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {0.47, 0.13, 0.39},
                {0.46, 0.49, 0.19}})
            m_mlp.InitializeWeights(2, {
                {0.22, 0.41, 0.23}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MLP1XORGifTanh231()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-tanh-231.gif

            InitXOR()
            m_mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 1000
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {0.05, 0.07, 0.43},
                {0.08, 0.46, 0.04},
                {0.04, 0.43, 0.54}})
            m_mlp.InitializeWeights(2, {
                {0.21, 0.51, 0.33, 0.15}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MLP1XORGifTanh281()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-tanh-281.gif

            InitXOR()
            m_mlp.InitializeStruct({2, 8, 1}, addBiasColumn:=True)
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 1000
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {0.07, 0.11, 0.43},
                {0.3, 0.49, 0.39},
                {0.06, 0.34, 0.14},
                {0.17, 0.13, 0.51},
                {0.31, 0.31, 0.19},
                {0.21, 0.43, 0.12},
                {0.06, 0.08, 0.33},
                {0.3, 0.38, 0.07}})
            m_mlp.InitializeWeights(2, {
                {0.02, 0.13, 0.05, 0.26, 0.18, 0.5, 0.14, 0.1, 0.1}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MLP1XORGifTanh2221()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-tanh-2221.gif

            InitXOR()
            m_mlp.InitializeStruct(m_neuronCountXOR4Layers, addBiasColumn:=True)
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 1000
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {0.36, 0.2, 0.16},
                {0.18, 0.48, 0.02}})
            m_mlp.InitializeWeights(2, {
                {0.2, 0.51, 0.16},
                {0.47, 0.41, 0.49}})
            m_mlp.InitializeWeights(3, {
                {0.05, 0.31, 0.38}})

            m_mlp.Train()

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
        Public Sub MLP1XORGifSigmoid221()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-sigmoid-221.gif

            InitXOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 13000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {-0.18, -0.09, 0.46},
                {-0.46, -0.46, -0.14}})
            m_mlp.InitializeWeights(2, {
                {-0.41, -0.44, -0.1}})

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
        Public Sub MLP1XORGifElu221()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-elu-221.gif

            InitXOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 600
            m_mlp.SetActivationFunction(enumActivationFunction.ELU, gain:=0.3!)

            m_mlp.InitializeWeights(1, {
                {0.26, 0.18, 0.49},
                {0.42, 0.43, 0.43}})
            m_mlp.InitializeWeights(2, {
                {0.12, 0.11, 0.24}})

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

        ' useAlternateDerivativeFunction = True
        '<TestMethod()>
        'Public Sub MLP1XORGifDbleThreshold()

        '    InitXOR()
        '    m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

        '    m_mlp.nbIterations = 5000
        '    m_mlp.SetActivationFunction(enumActivationFunction.DoubleThreshold)

        '    m_mlp.InitializeWeights(1, {
        '        {0.86, 0.37, 0.8},
        '        {1.0, 0.62, 0.17}})
        '    m_mlp.InitializeWeights(2, {
        '        {0.54, 0.83, 0.41}})

        '    m_mlp.Train()

        '    Dim expectedOutput = m_targetArrayXOR
        '    Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

        '    Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

        '    Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        '    Assert.AreEqual(sExpectedOutput, sOutput)

        '    Const expectedLoss# = 0
        '    Dim loss# = m_mlp.averageError
        '    Dim lossRounded# = Math.Round(loss, 2)
        '    Assert.AreEqual(True, lossRounded <= expectedLoss)

        'End Sub

        <TestMethod()>
        Public Sub MLP1XORGifDbleThreshold221()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-double-threshold-221.gif

            InitXOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 4000
            m_mlp.SetActivationFunction(enumActivationFunction.DoubleThreshold)

            m_mlp.InitializeWeights(1, {
                {-0.44, -0.2, -0.07},
                {-0.04, 0.22, 0.13}})
            m_mlp.InitializeWeights(2, {
                {-0.21, 0.28, -0.03}})

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
        Public Sub MLP1XORGifGaussian221()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-gaussian-221.gif

            InitXOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 700
            m_mlp.SetActivationFunction(enumActivationFunction.Gaussian)

            m_mlp.InitializeWeights(1, {
                {0.22, 0.76, 0.51},
                {0.71, 0.5, 0.46}})
            m_mlp.InitializeWeights(2, {
                {0.05, 0.87, 0.64}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MLP1XORGifSinus221()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-sinus-221.gif
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-sinus-221-zoomed-out.gif

            InitXOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 200
            m_mlp.SetActivationFunction(enumActivationFunction.Sinus)

            m_mlp.InitializeWeights(1, {
                {0.22, 0.39, 0.15},
                {0.2, 0.37, 0.39}})
            m_mlp.InitializeWeights(2, {
                {0.34, 0.12, 0.15}})

            m_mlp.Train()

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
        Public Sub MLP1XORGifReLu221()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-relu-221.gif

            InitXOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 700
            m_mlp.SetActivationFunction(enumActivationFunction.ReLu)

            m_mlp.InitializeWeights(1, {
                {0.1, 0.11, 0.14},
                {0.28, 0.04, 0.44}})
            m_mlp.InitializeWeights(2, {
                {0.45, 0.48, 0.24}})

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
        Public Sub MLP1XORGifArctan221()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-arctan-221.gif

            InitXOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 600
            m_mlp.SetActivationFunction(enumActivationFunction.ArcTangent)

            m_mlp.InitializeWeights(1, {
                {0.19, 0.25, 0.2},
                {0.47, 0.33, 0.49}})
            m_mlp.InitializeWeights(2, {
                {0.42, 0.23, 0.31}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MLP1XORGifMish221()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-mish-221.gif

            InitXOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 400
            m_mlp.SetActivationFunction(enumActivationFunction.Mish)

            m_mlp.InitializeWeights(1, {
                {0.52, 0.13, 0.49},
                {0.15, 0.16, 0.35}})
            m_mlp.InitializeWeights(2, {
                {0.35, 0.4, 0.11}})

            m_mlp.Train()

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
        Public Sub MLP1XORAnalogGifElu221()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-analog-elu-221.gif

            InitXORAnalog()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 700
            m_mlp.SetActivationFunction(enumActivationFunction.ELU, gain:=0.5!)

            m_mlp.InitializeWeights(1, {
                {0.12, 0.2, 0.06},
                {0.13, 0.11, 0.25}})
            m_mlp.InitializeWeights(2, {
                {0.25, 0.23, 0.22}})

            m_mlp.Train()

            'Dim expectedOutput = m_targetArrayXORAnalog
            ' The structure is too small to perform full learning (underfitting):
            Dim expectedOutput = New Double(,) {
                {1.1},
                {0.2},
                {1.1},
                {0.2},
                {0.9},
                {0.2},
                {0.9},
                {0.2},
                {0.2}}
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.13
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MLP1XORAnalogGifElu231()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-analog-elu-231.gif

            InitXORAnalog()
            m_mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 1000
            m_mlp.SetActivationFunction(enumActivationFunction.ELU, gain:=0.5!)

            m_mlp.InitializeWeights(1, {
                {0.22, 0.05, 0.16},
                {0.07, 0.03, 0.24},
                {0.1, 0.04, 0.2}})
            m_mlp.InitializeWeights(2, {
                {0.08, 0.12, 0.11, 0.24}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXORAnalog
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
        Public Sub MLP1XORAnalogGifDoubleThreshold241()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-analog-double-threshold-241.gif

            InitXORAnalog()
            m_mlp.InitializeStruct(m_neuronCountXOR241, addBiasColumn:=True)
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 5000
            m_mlp.SetActivationFunction(enumActivationFunction.DoubleThreshold)

            m_mlp.InitializeWeights(1, {
                {0.47, -0.27, -0.1},
                {0.2, 0.4, 0.22},
                {-0.08, 0.22, -0.02},
                {0.48, 0.29, -0.29}})
            m_mlp.InitializeWeights(2, {
                {0.22, 0.34, -0.31, -0.47, 0.34}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXORAnalog
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
        Public Sub MLP1XORAnalogGifReLu241()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-analog-relu-241.gif

            InitXORAnalog()
            m_mlp.InitializeStruct(m_neuronCountXOR241, addBiasColumn:=True)
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 500
            m_mlp.SetActivationFunction(enumActivationFunction.ReLu)

            m_mlp.InitializeWeights(1, {
                {0.02, 0.29, 0.11},
                {0.09, 0.05, 0.3},
                {0.22, 0.31, 0.12},
                {0.15, 0.21, 0.11}})
            m_mlp.InitializeWeights(2, {
                {0.19, 0.05, 0.17, 0.02, 0.34}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXORAnalog
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
        Public Sub MLP1XORAnalogGifTanh231()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-analog-tanh-231.gif

            InitXORAnalog()
            m_mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 1000
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {0.51, 0.34, 0.03},
                {0.33, 0.13, 0.28},
                {0.14, 0.35, 0.43}})
            m_mlp.InitializeWeights(2, {
                {0.36, 0.05, 0.04, 0.31}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXORAnalog
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MLP1XORAnalogGifTanh2331()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-analog-tanh-2331.gif

            InitXORAnalog()
            m_mlp.InitializeStruct(m_neuronCountXOR4Layers2331, addBiasColumn:=True)
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 2000
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {0.17, 0.55, 0.31},
                {0.41, 0.04, 0.06},
                {0.49, 0.16, 0.03}})
            m_mlp.InitializeWeights(2, {
                {0.03, 0.51, 0.52, 0.35},
                {0.24, 0.26, 0.31, 0.11},
                {0.26, 0.16, 0.3, 0.24}})
            m_mlp.InitializeWeights(3, {
                {0.51, 0.18, 0.08, 0.25}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXORAnalog
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MLP1XORAnalogGifArctan231()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-analog-arctan-231.gif

            InitXORAnalog()
            m_mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 1000
            m_mlp.SetActivationFunction(enumActivationFunction.ArcTangent, gain:=1)

            m_mlp.InitializeWeights(1, {
                {0.03, 0.38, 0.25},
                {0.08, 0.39, 0.23},
                {0.13, 0.27, 0.37}})
            m_mlp.InitializeWeights(2, {
                {0.47, 0.07, 0.27, 0.48}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXORAnalog
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
        Public Sub MLP1XORAnalogGifGaussian231()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-analog-gaussian-231.gif

            InitXORAnalog()
            m_mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 300
            m_mlp.SetActivationFunction(enumActivationFunction.Gaussian, gain:=1)

            m_mlp.InitializeWeights(1, {
                {0.33, 0.14, 0.56},
                {0.29, 0.8, 0.22},
                {0.28, 0.48, 0.09}})
            m_mlp.InitializeWeights(2, {
                {0.9, 0.71, 0.05, 0.12}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXORAnalog
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MLP1XORAnalogGifMish231()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-analog-mish-231.gif

            InitXORAnalog()
            m_mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 1000
            m_mlp.SetActivationFunction(enumActivationFunction.Mish, gain:=1)

            m_mlp.InitializeWeights(1, {
                {0.15, 0.52, 0.16},
                {0.51, 0.27, 0.14},
                {0.19, 0.34, 0.39}})
            m_mlp.InitializeWeights(2, {
                {0.12, 0.26, 0.27, 0.28}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXORAnalog
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MLP1XORAnalogGifSigmoid231()

            ' Video of the learning process:
            ' http://patrice.dargenton.free.fr/ai/perceptron/xor/xor-analog-sigmoid-231.gif

            InitXORAnalog()
            m_mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 20000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=1)

            m_mlp.InitializeWeights(1, {
                {0.01, -0.39, 0.28},
                {0.19, 0.19, -0.37},
                {-0.1, 0.44, 0.25}})
            m_mlp.InitializeWeights(2, {
                {0.31, -0.11, -0.37, -0.33}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXORAnalog
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.03
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

#End Region

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

        ' useAlternateDerivativeFunction = True
        '<TestMethod()>
        'Public Sub MLP2XORDbleThreshold()

        '    Init2XOR()
        '    m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

        '    m_mlp.nbIterations = 3000
        '    m_mlp.SetActivationFunction(enumActivationFunction.DoubleThreshold)

        '    m_mlp.InitializeWeights(1, {
        '        {0.43, 0.21, 0.16, 0.85, 0.2},
        '        {0.25, 0.5, 0.87, 0.5, 0.36},
        '        {0.21, 0.78, 0.68, 0.47, 0.35},
        '        {0.42, 0.09, 0.25, 0.87, 0.17}})
        '    m_mlp.InitializeWeights(2, {
        '        {0.88, 0.37, 0.12, 0.17, 0.79},
        '        {0.71, 0.88, 0.7, 0.83, 0.02}})

        '    m_mlp.Train()

        '    Dim expectedOutput = m_targetArray2XOR
        '    Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

        '    Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

        '    Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        '    Assert.AreEqual(sExpectedOutput, sOutput)

        '    Const expectedLoss# = 0
        '    Dim loss# = m_mlp.averageError
        '    Dim lossRounded# = Math.Round(loss, 2)
        '    Assert.AreEqual(True, lossRounded <= expectedLoss)

        'End Sub

        <TestMethod()>
        Public Sub MLP2XORDbleThreshold()

            Init2XOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.3!)

            m_mlp.nbIterations = 600
            m_mlp.SetActivationFunction(enumActivationFunction.DoubleThreshold)

            m_mlp.InitializeWeights(1, {
               {0.42, 0.25, -0.15, -0.3, -0.41},
                {-0.31, -0.07, 0.46, -0.44, -0.43},
                {0.39, 0.11, -0.35, 0.18, -0.13},
                {-0.14, -0.47, -0.2, -0.26, 0.42}})
            m_mlp.InitializeWeights(2, {
                {0.04, 0.39, -0.23, 0.19, -0.28},
                {-0.25, 0.11, 0.47, 0.11, 0.37}})

            m_mlp.Train()

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
        Public Sub MLP3XORMishStdr()

            TestMLP3XORMish(m_mlp)

        End Sub

        ' useAlternateDerivativeFunction = True
        '<TestMethod()>
        'Public Sub MLP3XORDbleThreshold()

        '    Init3XOR()
        '    m_mlp.Initialize(learningRate:=2.0!, weightAdjustment:=0.1!)

        '    m_mlp.nbIterations = 100
        '    m_mlp.SetActivationFunction(enumActivationFunction.DoubleThreshold, center:=2)

        '    m_mlp.InitializeWeights(1, {
        '        {0.12, 0.3, 0.12, 0.28, 0.12, 0.2, 0.32},
        '        {0.05, 0.6, 0.24, 0.41, 0.75, 0.57, 0.76},
        '        {0.51, 0.62, 0.39, 0.7, 0.29, 0.53, 0.28},
        '        {0.51, 0.43, 0.45, 0.4, 0.36, 0.56, 0.74},
        '        {0.77, 0.82, 0.95, 0.06, 0.84, 0.71, 0.27},
        '        {0.66, 0.29, 0.85, 0.32, 0.92, 0.48, 0.29}})
        '    m_mlp.InitializeWeights(2, {
        '        {0.71, 0.99, 0.73, 0.06, 0.95, 0.55, 0.57},
        '        {0.38, 0.5, 0.37, 0.85, 0.78, 1.0, 0.61},
        '        {0.87, 0.67, 0.87, 0.76, 0.64, 0.59, 0.27}})
        '    m_mlp.Train()

        '    Dim sOutput = m_mlp.output.ToString() 'WithFormat(dec:="0.0")

        '    Dim expectedOutput = m_targetArray3XOR
        '    Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        '    Dim sExpectedOutput = expectedMatrix.ToString() 'WithFormat(dec:="0.0")
        '    Assert.AreEqual(sExpectedOutput, sOutput)

        '    Const expectedLoss# = 0
        '    Dim loss# = m_mlp.averageError
        '    Dim lossRounded# = Math.Round(loss, 2)
        '    Assert.AreEqual(True, lossRounded <= expectedLoss)

        'End Sub

        <TestMethod()>
        Public Sub MLP3XORDbleThreshold()

            Init3XOR()
            m_mlp.Initialize(learningRate:=0.15!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 200
            m_mlp.SetActivationFunction(enumActivationFunction.DoubleThreshold)

            m_mlp.InitializeWeights(1, {
                {-0.42, 0.36, 0.34, 0.34, 0.38, 0.21, -0.38},
                {-0.35, -0.15, 0.41, -0.47, 0.4, -0.24, 0.44},
                {0.35, -0.13, 0.04, 0.39, 0.25, 0.13, 0.34},
                {-0.17, -0.46, -0.47, 0.14, 0.39, -0.14, 0.16},
                {-0.17, 0.06, 0.29, -0.34, 0.38, 0.46, -0.26},
                {-0.06, 0.17, -0.33, 0.4, -0.24, 0.2, -0.37}})
            m_mlp.InitializeWeights(2, {
                {0.45, -0.43, 0.33, -0.17, -0.35, -0.12, -0.35},
                {0.08, 0.08, 0.14, 0.37, -0.16, -0.17, 0.25},
                {0.33, 0.32, -0.29, 0.48, -0.15, -0.05, 0.31}})
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