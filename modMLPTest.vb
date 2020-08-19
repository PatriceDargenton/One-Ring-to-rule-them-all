
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode
Imports Microsoft.VisualStudio.TestTools.UnitTesting ' Assert.AreEqual

Module modMLPTest

#Region "Initialization"

    Public ReadOnly m_neuronCountXOR%() = {2, 2, 1}
    Public ReadOnly m_neuronCountXOR231%() = {2, 3, 1} ' With bias
    Public ReadOnly m_neuronCountXOR261%() = {2, 6, 1} ' TensorFlow minimal size
    Public ReadOnly m_neuronCount2XOR%() = {4, 4, 2}
    Public ReadOnly m_neuronCount2XOR452%() = {4, 5, 2}
    Public ReadOnly m_neuronCount2XOR462%() = {4, 6, 2} ' TensorFlow minimal size
    Public ReadOnly m_neuronCount3XOR%() = {6, 6, 3}
    Public ReadOnly m_neuronCount3XOR673%() = {6, 7, 3}
    Public ReadOnly m_neuronCountXOR4Layers%() = {2, 2, 2, 1}
    Public ReadOnly m_neuronCountXOR5Layers%() = {2, 2, 2, 2, 1}
    Public ReadOnly m_neuronCountXOR4Layers2331%() = {2, 3, 3, 1}
    Public ReadOnly m_neuronCountXOR5Layers23331%() = {2, 3, 3, 3, 1}

    Public ReadOnly m_inputArrayXOR!(,) = {
        {1, 0},
        {0, 0},
        {0, 1},
        {1, 1}}

    Public ReadOnly m_targetArrayXOR!(,) = {
        {1},
        {0},
        {1},
        {0}}

    Public ReadOnly m_inputArray2XOR!(,) = {
        {1, 0, 1, 0},
        {1, 0, 0, 0},
        {1, 0, 0, 1},
        {1, 0, 1, 1},
        {0, 0, 1, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 1},
        {0, 0, 1, 1},
        {0, 1, 1, 0},
        {0, 1, 0, 0},
        {0, 1, 0, 1},
        {0, 1, 1, 1},
        {1, 1, 1, 0},
        {1, 1, 0, 0},
        {1, 1, 0, 1},
        {1, 1, 1, 1}}

    Public ReadOnly m_targetArray2XOR!(,) = {
        {1, 1},
        {1, 0},
        {1, 1},
        {1, 0},
        {0, 1},
        {0, 0},
        {0, 1},
        {0, 0},
        {1, 1},
        {1, 0},
        {1, 1},
        {1, 0},
        {0, 1},
        {0, 0},
        {0, 1},
        {0, 0}}

    Public ReadOnly m_inputArray3XOR!(,) = {
        {1, 0, 1, 0, 1, 0},
        {1, 0, 1, 0, 0, 0},
        {1, 0, 1, 0, 0, 1},
        {1, 0, 1, 0, 1, 1},
        {1, 0, 0, 0, 1, 0},
        {1, 0, 0, 0, 0, 0},
        {1, 0, 0, 0, 0, 1},
        {1, 0, 0, 0, 1, 1},
        {1, 0, 0, 1, 1, 0},
        {1, 0, 0, 1, 0, 0},
        {1, 0, 0, 1, 0, 1},
        {1, 0, 0, 1, 1, 1},
        {1, 0, 1, 1, 1, 0},
        {1, 0, 1, 1, 0, 0},
        {1, 0, 1, 1, 0, 1},
        {1, 0, 1, 1, 1, 1},
        {0, 0, 1, 0, 1, 0},
        {0, 0, 1, 0, 0, 0},
        {0, 0, 1, 0, 0, 1},
        {0, 0, 1, 0, 1, 1},
        {0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1, 1},
        {0, 0, 0, 1, 1, 0},
        {0, 0, 0, 1, 0, 0},
        {0, 0, 0, 1, 0, 1},
        {0, 0, 0, 1, 1, 1},
        {0, 0, 1, 1, 1, 0},
        {0, 0, 1, 1, 0, 0},
        {0, 0, 1, 1, 0, 1},
        {0, 0, 1, 1, 1, 1},
        {0, 1, 1, 0, 1, 0},
        {0, 1, 1, 0, 0, 0},
        {0, 1, 1, 0, 0, 1},
        {0, 1, 1, 0, 1, 1},
        {0, 1, 0, 0, 1, 0},
        {0, 1, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 1},
        {0, 1, 0, 0, 1, 1},
        {0, 1, 0, 1, 1, 0},
        {0, 1, 0, 1, 0, 0},
        {0, 1, 0, 1, 0, 1},
        {0, 1, 0, 1, 1, 1},
        {0, 1, 1, 1, 1, 0},
        {0, 1, 1, 1, 0, 0},
        {0, 1, 1, 1, 0, 1},
        {0, 1, 1, 1, 1, 1},
        {1, 1, 1, 0, 1, 0},
        {1, 1, 1, 0, 0, 0},
        {1, 1, 1, 0, 0, 1},
        {1, 1, 1, 0, 1, 1},
        {1, 1, 0, 0, 1, 0},
        {1, 1, 0, 0, 0, 0},
        {1, 1, 0, 0, 0, 1},
        {1, 1, 0, 0, 1, 1},
        {1, 1, 0, 1, 1, 0},
        {1, 1, 0, 1, 0, 0},
        {1, 1, 0, 1, 0, 1},
        {1, 1, 0, 1, 1, 1},
        {1, 1, 1, 1, 1, 0},
        {1, 1, 1, 1, 0, 0},
        {1, 1, 1, 1, 0, 1},
        {1, 1, 1, 1, 1, 1}}

    Public ReadOnly m_targetArray3XOR!(,) = {
        {1, 1, 1},
        {1, 1, 0},
        {1, 1, 1},
        {1, 1, 0},
        {1, 0, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 0, 0},
        {1, 1, 1},
        {1, 1, 0},
        {1, 1, 1},
        {1, 1, 0},
        {1, 0, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 0, 0},
        {0, 1, 1},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 0},
        {0, 0, 1},
        {0, 0, 0},
        {0, 0, 1},
        {0, 0, 0},
        {0, 1, 1},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 0},
        {0, 0, 1},
        {0, 0, 0},
        {0, 0, 1},
        {0, 0, 0},
        {1, 1, 1},
        {1, 1, 0},
        {1, 1, 1},
        {1, 1, 0},
        {1, 0, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 0, 0},
        {1, 1, 1},
        {1, 1, 0},
        {1, 1, 1},
        {1, 1, 0},
        {1, 0, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 0, 0},
        {0, 1, 1},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 0},
        {0, 0, 1},
        {0, 0, 0},
        {0, 0, 1},
        {0, 0, 0},
        {0, 1, 1},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 0},
        {0, 0, 1},
        {0, 0, 0},
        {0, 0, 1},
        {0, 0, 0}}

    Private Sub InitXOR(mlp As clsMLPGeneric)
        mlp.Initialize(learningRate:=0.01!, weightAdjustment:=0)
        mlp.inputArray = m_inputArrayXOR
        mlp.targetArray = m_targetArrayXOR
    End Sub

    Private Sub Init2XOR(mlp As clsMLPGeneric)
        mlp.Initialize(learningRate:=0.01!, weightAdjustment:=0)
        mlp.inputArray = m_inputArray2XOR
        mlp.targetArray = m_targetArray2XOR
    End Sub

    Private Sub Init3XOR(mlp As clsMLPGeneric)
        mlp.Initialize(learningRate:=0.01!, weightAdjustment:=0)
        mlp.inputArray = m_inputArray3XOR
        mlp.targetArray = m_targetArray3XOR
    End Sub

#End Region

#Region "Standard tests"

    Public Sub TestMLP1XORSemiStochastic(mlp As clsMLPGeneric,
        Optional nbIterations% = 8000,
        Optional expectedLoss# = 0.03#,
        Optional learningRate! = 0.05!,
        Optional weightAdjustment! = 0.1!,
        Optional gain! = 2)

        InitXOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain, center:=0)

        mlp.InitializeWeights(1, {
            {-0.75, 0.64, -0.09},
            {0.12, 0.75, -0.63}})
        mlp.InitializeWeights(2, {
            {-0.79, -0.13, 0.58}})

        mlp.Train(enumLearningMode.SemiStochastic)

        Dim expectedOutput = m_targetArrayXOR
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

        Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")

        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP1XORStochastic(mlp As clsMLPGeneric,
        Optional nbIterations% = 25000,
        Optional expectedLoss# = 0.04#,
        Optional learningRate! = 0.05!,
        Optional weightAdjustment! = 0.1!,
        Optional gain! = 2)

        InitXOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain, center:=0)

        mlp.InitializeWeights(1, {
            {-0.75, 0.64, -0.09},
            {0.12, 0.75, -0.63}})
        mlp.InitializeWeights(2, {
            {-0.79, -0.13, 0.58}})

        mlp.Train(enumLearningMode.Stochastic)

        Dim expectedOutput = m_targetArrayXOR
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

        Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")

        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP1XORWithoutBias(mlp As clsMLPGeneric,
        Optional nbIterations% = 40000,
        Optional expectedLoss# = 0.11#,
        Optional learningRate! = 0.1!,
        Optional weightAdjustment! = 0.02!,
        Optional gain! = 1)

        InitXOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=False)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain, center:=0)

        mlp.InitializeWeights(1, {
            {0.42, 0.79},
            {0.55, 0.02}})
        mlp.InitializeWeights(2, {
            {0.51, 0.31}})

        mlp.Train()

        'Dim expectedOutput = m_targetArrayXOR
        Dim expectedOutput = New Double(,) {
            {0.9},
            {0.1},
            {0.9},
            {0.1}}
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

        Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")

        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP1XORWithoutBias231(mlp As clsMLPGeneric,
        Optional nbIterations% = 60000,
        Optional expectedLoss# = 0.11#,
        Optional learningRate! = 0.08!,
        Optional weightAdjustment! = 0.02!,
        Optional gain! = 1)

        InitXOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=False)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain, center:=0)

        mlp.InitializeWeights(1, {
            {0.73, 0.38},
            {0.07, 0.3},
            {0.99, 0.25}})
        mlp.InitializeWeights(2, {
            {1.0, 0.98, 0.61}})

        mlp.Train()

        'Dim expectedOutput = m_targetArrayXOR
        Dim expectedOutput = New Double(,) {
            {0.9},
            {0.1},
            {0.9},
            {0.1}}

        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

        Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")

        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    'Public Sub TestMLP1XORWithoutBias231b(mlp As clsMLPGeneric,
    '    Optional nbIterations% = 60000,
    '    Optional expectedLoss# = 0.03#,
    '    Optional learningRate! = 0.09!,
    '    Optional weightAdjustment! = 0.05!,
    '    Optional gain! = 1,
    '    Optional center! = 2.2!)

    '    InitXOR(mlp)
    '    mlp.Initialize(learningRate, weightAdjustment)
    '    mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=False)

    '    mlp.nbIterations = nbIterations
    '    mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain, center)

    '    mlp.InitializeWeights(1, {
    '        {0.66, 0.53},
    '        {0.65, 0.69},
    '        {0.82, 0.56}})
    '    mlp.InitializeWeights(2, {
    '        {0.62, 0.54, 0.5}})

    '    mlp.Train()

    '    Dim expectedOutput = m_targetArrayXOR
    '    Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

    '    Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")

    '    Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
    '    Assert.AreEqual(sExpectedOutput, sOutput)

    '    Dim loss! = mlp.ComputeAverageError()
    '    Dim lossRounded# = Math.Round(loss, 2)
    '    Assert.AreEqual(True, lossRounded <= expectedLoss)

    'End Sub

    Public Sub TestMLP1XORELU(mlp As clsMLPGeneric,
            Optional nbIterations% = 600,
            Optional expectedLoss# = 0.01#,
            Optional learningRate! = 0.05!,
            Optional weightAdjustment! = 0.0!,
            Optional gain! = 0.6!,
            Optional center! = 0.3!,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        InitXOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
        mlp.SetActivationFunction(
            enumActivationFunction.ELU, gain, center)

        mlp.InitializeWeights(1, {
            {0.98, 0.15, 0.5},
            {0.11, 0.11, 0.29}})
        mlp.InitializeWeights(2, {
            {0.83, 0.44, 0.06}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArrayXOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

#End Region

#Region "Vectorized tests"
#End Region

#Region "Universal tests"

    Public Sub TestMLP1XOR4Layers(mlp As clsMLPGeneric,
        Optional nbIterations% = 500,
        Optional expectedLoss# = 0.02,
        Optional learningRate! = 0.1!,
        Optional weightAdjustment! = 0.1,
        Optional gain! = 2,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        InitXOR(mlp)
        mlp.weightAdjustment = weightAdjustment
        mlp.learningRate = learningRate
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCountXOR4Layers, addBiasColumn:=True)
        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain, center:=0)

        mlp.InitializeWeights(1, {
            {-0.2, 0.97, 0.06},
            {0.73, -0.63, -0.21}})
        mlp.InitializeWeights(2, {
            {0.53, 0.58, -0.6},
            {-0.8, -0.59, 0.05}})
        mlp.InitializeWeights(3, {
            {-0.66, -0.55, 0.49}})

        mlp.Train()

        Dim expectedOutput = m_targetArrayXOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP1XOR5Layers(mlp As clsMLPGeneric,
        Optional nbIterations% = 900,
        Optional expectedLoss# = 0.02,
        Optional learningRate! = 0.1!,
        Optional weightAdjustment! = 0.1,
        Optional gain! = 2,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        InitXOR(mlp)
        mlp.weightAdjustment = weightAdjustment
        mlp.learningRate = learningRate
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCountXOR5Layers, addBiasColumn:=True)
        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain, center:=0)

        mlp.InitializeWeights(1, {
            {0.86, 0.09, -0.48},
            {0.84, -0.01, 0.52}})
        mlp.InitializeWeights(2, {
            {-0.86, 0.24, 0.42},
            {-0.68, 0.25, -0.68}})
        mlp.InitializeWeights(3, {
            {0.44, -0.42, 0.78},
            {0.38, 0.05, 0.91}})
        mlp.InitializeWeights(4, {
            {0.51, 0.11, 0.84}})

        mlp.Train()

        Dim expectedOutput = m_targetArrayXOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP1XORSigmoid(mlp As clsMLPGeneric,
            Optional nbIterations% = 2000,
            Optional expectedLoss# = 0.04#,
            Optional learningRate! = 0.1!,
            Optional weightAdjustment! = 0.1!,
            Optional gain! = 2.5,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        InitXOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
        mlp.SetActivationFunction(
            enumActivationFunction.Sigmoid, gain, center:=0)

        mlp.InitializeWeights(1, {
            {0.36, -0.84, 0.37},
            {0.19, -0.8, -0.56}})
        mlp.InitializeWeights(2, {
            {0.14, 0.3, 0.93}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArrayXOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP1XORHTangent(mlp As clsMLPGeneric,
            Optional nbIterations% = 2500,
            Optional expectedLoss# = 0.02#,
            Optional learningRate! = 0.05!,
            Optional weightAdjustment! = 0.1!,
            Optional gain! = 2.4,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        InitXOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations ' HTan: works
        mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
        mlp.SetActivationFunction(
            enumActivationFunction.HyperbolicTangent, gain, center:=0)

        mlp.InitializeWeights(1, {
            {-0.59, 0.78, -0.16},
            {0.25, 0.94, -0.17}})
        mlp.InitializeWeights(2, {
            {-0.45, -0.8, -0.37}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArrayXOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP1XORHTangent261(mlp As clsMLPGeneric,
            Optional nbIterations% = 400,
            Optional expectedLoss# = 0.02#,
            Optional learningRate! = 0.2!,
            Optional weightAdjustment! = 0,
            Optional gain! = 2,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        InitXOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations ' HTan: works
        mlp.InitializeStruct(m_neuronCountXOR261, addBiasColumn:=False)
        mlp.SetActivationFunction(
            enumActivationFunction.HyperbolicTangent, gain, center:=0)

        mlp.InitializeWeights(1, {
            {0.28, 0.43},
            {-0.47, -0.41},
            {0.02, -0.31},
            {0.06, 0.45},
            {0.22, 0.46},
            {-0.13, 0.08}})
        mlp.InitializeWeights(2, {
            {-0.05, 0.19, 0.34, -0.26, -0.38, -0.07}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArrayXOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP2XORSigmoid(mlp As clsMLPGeneric,
            Optional nbIterations% = 200,
            Optional expectedLoss# = 0.03#,
            Optional learningRate! = 0.09!,
            Optional weightAdjustment! = 0.1!,
            Optional gain! = 2.6!,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Init2XOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)
        mlp.SetActivationFunction(
            enumActivationFunction.HyperbolicTangent, gain, center:=0)

        mlp.InitializeWeights(1, {
            {0.6, 0.52, 0.12, -0.33, -0.48},
            {-0.5, -0.04, 0.39, 0.4, 0.64},
            {-0.66, 0.23, -0.19, 0.32, 0.6},
            {0.21, 0.6, -0.29, -0.7, 0.01}})
        mlp.InitializeWeights(2, {
            {-0.07, -0.69, -0.08, -0.52, 0.48},
            {0.25, -0.37, -0.34, 0.49, -0.65}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArray2XOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP2XORHTangent(mlp As clsMLPGeneric,
            Optional nbIterations% = 400,
            Optional expectedLoss# = 0.02#,
            Optional learningRate! = 0.1!,
            Optional weightAdjustment! = 0.15!,
            Optional gain! = 2,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Init2XOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations ' HTan: works
        mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)
        mlp.SetActivationFunction(
            enumActivationFunction.HyperbolicTangent, gain, center:=0)

        mlp.InitializeWeights(1, {
            {0.55, 0.15, 0.71, 0.29, 0.25},
            {0.29, 0.53, -0.1, 0.4, -0.67},
            {-0.54, 0.08, 0.27, 0.72, -0.3},
            {0.48, 0.47, -0.67, 0.17, 0.23}})
        mlp.InitializeWeights(2, {
            {-0.32, -0.57, -0.05, 0.55, -0.5},
            {0.28, -0.76, -0.42, 0.21, 0.31}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArray2XOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP2XORHTangent462(mlp As clsMLPGeneric,
        Optional nbIterations% = 700,
        Optional expectedLoss# = 0.01#,
        Optional learningRate! = 0.1!,
        Optional weightAdjustment! = 0,
        Optional gain! = 2,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Init2XOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations ' HTan: works
        mlp.InitializeStruct(m_neuronCount2XOR462, addBiasColumn:=False)
        mlp.SetActivationFunction(
            enumActivationFunction.HyperbolicTangent, gain, center:=0)

        mlp.InitializeWeights(1, {
            {0.1, -0.37, -0.04, -0.3},
            {-0.15, -0.02, 0.04, 0.18},
            {0.16, -0.45, 0.13, 0.12},
            {-0.28, 0.08, -0.45, -0.15},
            {0.15, 0.18, 0.48, -0.07},
            {0.2, -0.38, 0.24, -0.45}})
        mlp.InitializeWeights(2, {
            {0.5, 0.47, 0.37, 0.32, -0.3, 0.02},
            {0.34, 0.31, -0.12, -0.33, 0.01, -0.31}})

        'mlp.PrintWeights()
        'mlp.printOutput_ = True

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArray2XOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP3XORSigmoid(mlp As clsMLPGeneric,
            Optional nbIterations% = 600,
            Optional expectedLoss# = 0.02#,
            Optional learningRate! = 0.1!,
            Optional weightAdjustment! = 0.1!,
            Optional gain! = 2,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Init3XOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
        mlp.SetActivationFunction(
            enumActivationFunction.Sigmoid, gain, center:=0)

        mlp.InitializeWeights(1, {
            {-0.45, -0.45, -0.17, -0.29, -0.03, -0.19, 0.58},
            {0.01, 0.47, -0.55, -0.07, -0.46, -0.33, -0.21},
            {0.18, -0.21, -0.43, 0.49, -0.13, 0.6, -0.09},
            {0.43, 0.39, -0.25, 0.49, -0.07, 0.32, 0.38},
            {0.33, 0.55, -0.19, 0.38, 0.09, -0.19, -0.49},
            {-0.12, 0.58, 0.14, -0.02, 0.13, 0.56, -0.43}})

        mlp.InitializeWeights(2, {
            {-0.14, 0.31, -0.49, -0.46, 0.25, 0.32, -0.4},
            {-0.6, 0.43, -0.06, -0.03, -0.18, 0.5, -0.24},
            {-0.17, -0.53, -0.11, -0.55, 0.13, -0.4, 0.29}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArray3XOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP3XORHTangent(mlp As clsMLPGeneric,
            Optional nbIterations% = 1100,
            Optional expectedLoss# = 0.02#,
            Optional learningRate! = 0.1!,
            Optional weightAdjustment! = 0,
            Optional gain! = 1,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Init3XOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCount3XOR673, addBiasColumn:=True)
        mlp.SetActivationFunction(
            enumActivationFunction.HyperbolicTangent, gain, center:=0)

        mlp.InitializeWeights(1, {
            {-0.01, -0.12, -0.57, -0.02, -0.9, 0.61, 0.32},
            {0.5, 0.08, 0.86, -0.76, -0.91, -0.23, -0.81},
            {-0.13, 0.14, -0.46, 0.08, -0.25, 0.65, 0.45},
            {-0.48, 0.48, -0.47, 0.49, -0.34, -0.76, 0.71},
            {-0.86, 0.05, -0.18, 0.62, 0.35, 0.42, -0.61},
            {0.02, 0.91, -0.29, 0.37, 0.38, -0.34, 0.84},
            {0.2, 0.66, -0.58, -0.71, -0.51, -0.47, 0.39}})
        mlp.InitializeWeights(2, {
            {0.77, 0.49, -0.99, 0.03, -0.09, -0.61, -0.47, 0.09},
            {-0.57, 0.32, 0.26, -0.47, -0.22, -0.18, 0.17, -0.11},
            {0.68, -0.33, 0.79, 0.65, -0.01, 0.7, -0.18, -0.71}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArray3XOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

#End Region

End Module