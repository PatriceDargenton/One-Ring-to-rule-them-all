
Imports Perceptron
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Module modMLPTests

#Region "Standard tests"

    Public Sub TestMLP1XOR(mlp As clsMLPGeneric,
        Optional nbIterations% = 5000,
        Optional expectedLoss# = 0.04#,
        Optional learningRate! = 0.05!,
        Optional weightAdjustment! = 0.1!,
        Optional gain! = 2)

        mlp.inputArray = {
            {1, 0},
            {0, 0},
            {0, 1},
            {1, 1}}
        mlp.targetArray = {
            {1},
            {0},
            {1},
            {0}}
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.InitializeStruct({2, 2, 1}, addBiasColumn:=True)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)

        mlp.InitializeWeights(1, {
            {-0.75, 0.64, -0.09},
            {0.12, 0.75, -0.63}})
        mlp.InitializeWeights(2, {
            {-0.79, -0.13, 0.58}})

        mlp.Train()

        Dim expectedOutput = mlp.targetArray
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

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
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)

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

        Dim loss# = mlp.averageError
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
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)

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

        Dim loss# = mlp.averageError
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
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)

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

        Dim loss# = mlp.averageError
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
        mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=False)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)

        mlp.InitializeWeights(1, {
            {0.73, 0.38},
            {0.07, 0.3},
            {0.99, 0.25}})
        mlp.InitializeWeights(2, {
            {1.0, 0.98, 0.61}})

        mlp.Train()

        Dim expectedOutput = m_targetArrayXOR

        Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

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

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

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
        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain)

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

        Dim loss# = mlp.averageError
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
        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain)

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

        Dim loss# = mlp.averageError
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
            enumActivationFunction.Sigmoid, gain)

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

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        ' XOR at 90%: works
        mlp.TestAllSamples(m_inputArrayXOR90PC)
        Dim sOutput90PC$ = mlp.output.ToStringWithFormat(dec:="0.00")

        ' XOR at 80%: works
        mlp.TestAllSamples(m_inputArrayXOR80PC)
        Dim sOutput80PC$ = mlp.output.ToStringWithFormat(dec:="0.00")

        ' XOR at 70%: does not works anymore
        mlp.TestAllSamples(m_inputArrayXOR70PC)
        Dim sOutput70PC$ = mlp.output.ToStringWithFormat(dec:="0.00")

    End Sub

    Public Sub TestMLP1XORSigmoidRProp(mlp As clsMLPGeneric,
            Optional nbIterations% = 140,
            Optional expectedLoss# = 0.03#,
            Optional learningRate! = 0.1!,
            Optional weightAdjustment! = 0.1!,
            Optional gain! = 1,
            Optional learningMode As enumLearningMode = enumLearningMode.Vectorial,
            Optional trainingAlgorithm As enumTrainingAlgorithm =
                enumTrainingAlgorithm.Default)

        InitXOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
        mlp.trainingAlgorithm = trainingAlgorithm
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)

        mlp.InitializeWeights(1, {
            {-0.11, -0.17, 0.05},
            {0.47, 0.05, 0.08},
            {-0.26, 0.09, 0.1}})
        mlp.InitializeWeights(2, {
            {-0.38, 0.08, -0.2, -0.01}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArrayXOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        ' XOR at 90%: works for Accord
        mlp.TestAllSamples(m_inputArrayXOR90PC)
        Dim sOutput90PC$ = mlp.output.ToStringWithFormat(dec:="0.00")

        ' XOR at 80%: works for Accord
        mlp.TestAllSamples(m_inputArrayXOR80PC)
        Dim sOutput80PC$ = mlp.output.ToStringWithFormat(dec:="0.00")

        ' XOR at 70%: works for Accord
        mlp.TestAllSamples(m_inputArrayXOR70PC)
        Dim sOutput70PC$ = mlp.output.ToStringWithFormat(dec:="0.00")

    End Sub

    Public Sub TestMLP1XORTanhRProp(mlp As clsMLPGeneric,
            Optional nbIterations% = 180,
            Optional expectedLoss# = 0.03#,
            Optional learningRate! = 0.1!,
            Optional weightAdjustment! = 0.1!,
            Optional gain! = 2,
            Optional learningMode As enumLearningMode = enumLearningMode.Vectorial,
            Optional trainingAlgorithm As enumTrainingAlgorithm =
                enumTrainingAlgorithm.Default)

        InitXOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
        mlp.trainingAlgorithm = trainingAlgorithm
        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain)

        mlp.InitializeWeights(1, {
            {0.38, 0.47, -0.12},
            {0.41, -0.04, -0.03},
            {0.14, -0.12, -0.31}})
        mlp.InitializeWeights(2, {
            {-0.41, 0.27, -0.41, 0.46}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArrayXOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        ' XOR at 90%: does not work
        mlp.TestAllSamples(m_inputArrayXOR90PC)
        Dim sOutput90PC$ = mlp.output.ToStringWithFormat(dec:="0.00")

        ' XOR at 80%: does not work
        mlp.TestAllSamples(m_inputArrayXOR80PC)
        Dim sOutput80PC$ = mlp.output.ToStringWithFormat(dec:="0.00")

        ' XOR at 70%: does not work
        mlp.TestAllSamples(m_inputArrayXOR70PC)
        Dim sOutput70PC$ = mlp.output.ToStringWithFormat(dec:="0.00")

    End Sub

    Public Sub TestMLP1XORTanh(mlp As clsMLPGeneric,
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
            enumActivationFunction.HyperbolicTangent, gain)

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

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP1XORTanh261(mlp As clsMLPGeneric,
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
            enumActivationFunction.HyperbolicTangent, gain)

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

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP2XORSigmoid(mlp As clsMLPGeneric,
            Optional nbIterations% = 5000,
            Optional expectedLoss# = 0.03#,
            Optional learningRate! = 0.1!,
            Optional weightAdjustment! = 0.1!,
            Optional gain! = 1.0!,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut,
            Optional trainingAlgorithm As enumTrainingAlgorithm =
                enumTrainingAlgorithm.Default)

        Init2XOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)
        mlp.trainingAlgorithm = trainingAlgorithm
        mlp.SetActivationFunction(
            enumActivationFunction.Sigmoid, gain)

        mlp.InitializeWeights(1, {
            {-0.05, -0.28, -0.14, -0.42, -0.47},
            {0.41, 0.15, 0.05, 0.04, -0.46},
            {-0.23, 0.29, -0.19, 0.31, -0.37},
            {-0.4, -0.44, -0.29, -0.03, 0.05}})
        mlp.InitializeWeights(2, {
            {-0.2, 0.17, -0.35, 0.46, 0.26},
            {0.16, 0.35, 0.4, -0.11, -0.42}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArray2XOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP2XORTanh2(mlp As clsMLPGeneric,
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
            enumActivationFunction.HyperbolicTangent, gain)

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

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP2XORTanh(mlp As clsMLPGeneric,
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
            enumActivationFunction.HyperbolicTangent, gain)

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

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP2XORTanh462(mlp As clsMLPGeneric,
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
            enumActivationFunction.HyperbolicTangent, gain)

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

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArray2XOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss# = mlp.averageError
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
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)

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

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP3XORSigmoid2(mlp As clsMLPGeneric,
            Optional nbIterations% = 600,
            Optional expectedLoss# = 0.03#,
            Optional learningRate! = 0.2!,
            Optional weightAdjustment! = 0.2!,
            Optional gain! = 1,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Init3XOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)

        mlp.InitializeWeights(1, {
            {0.19, 0.16, 0.11, -0.16, 0.2, 0.25, -0.06},
            {0.03, -0.04, -0.45, -0.28, -0.26, -0.47, -0.47},
            {-0.15, 0.14, -0.08, 0.3, 0.07, 0.33, -0.08},
            {0.29, -0.15, -0.28, 0.08, 0.42, 0.13, -0.42},
            {-0.22, -0.28, -0.04, -0.23, 0.14, 0.36, -0.03},
            {-0.46, -0.02, 0.12, -0.09, 0.36, 0.04, -0.19}})
        mlp.InitializeWeights(2, {
            {0.45, -0.1, 0.29, 0.4, 0.44, 0.36, -0.5},
            {0.5, -0.25, -0.3, 0.03, 0.3, -0.32, 0.27},
            {-0.05, -0.32, -0.23, 0.04, 0.47, -0.31, 0.09}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArray3XOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP3XORTanh(mlp As clsMLPGeneric,
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
            enumActivationFunction.HyperbolicTangent, gain)

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

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP3XORTanh2(mlp As clsMLPGeneric,
            Optional nbIterations% = 1700,
            Optional expectedLoss# = 0.02#,
            Optional learningRate! = 0.2!,
            Optional weightAdjustment! = 0.05!,
            Optional gain! = 0.2!,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Init3XOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
        mlp.SetActivationFunction(
            enumActivationFunction.HyperbolicTangent, gain)

        mlp.InitializeWeights(1, {
            {0.89, 4.58, -4.45, 2.06, 2.27, 9.88, -3.54},
            {7.13, -2.0, -4.49, 1.85, 1.15, 7.36, -1.34},
            {-5.21, -6.99, -8.11, -6.19, 8.36, 4.09, -0.82},
            {7.52, -5.66, -3.38, -7.76, -4.6, 0.26, 5.3},
            {-4.54, -1.19, 3.66, 1.97, -7.85, 9.82, -3.4},
            {-0.86, 9.62, -8.06, -9.22, 9.43, 3.04, -6.54}})
        mlp.InitializeWeights(2, {
            {-5.2, 9.2, -4.82, 5.79, 0.12, -4.73, -3.79},
            {-0.91, 3.61, -0.6, 4.63, -2.67, -8.18, -4.84},
            {-6.79, -4.89, -4.11, -1.08, 9.92, 9.2, -4.15}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArray3XOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP3XORGaussian(mlp As clsMLPGeneric,
            Optional nbIterations% = 400,
            Optional expectedLoss# = 0.0#,
            Optional learningRate! = 0.15!,
            Optional weightAdjustment! = 0.25!,
            Optional gain! = 0.5!,
            Optional center! = 0.3!,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Init3XOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
        mlp.SetActivationFunction(enumActivationFunction.Gaussian, gain, center)

        mlp.InitializeWeights(1, {
            {0.39, 0.1, 0.21, 0.56, 0.84, 0.37, 0.06},
            {0.88, 0.81, 0.94, 0.82, 0.97, 0.24, 0.13},
            {0.96, 0.88, 0.49, 0.84, 0.74, 0.64, 0.98},
            {0.04, 0.39, 0.68, 0.06, 0.32, 0.38, 0.73},
            {0.44, 0.35, 0.89, 0.62, 0.51, 0.69, 0.67},
            {0.17, 0.29, 0.08, 0.83, 0.65, 0.49, 0.47}})
        mlp.InitializeWeights(2, {
            {0.8, 0.63, 0.21, 0.62, 0.58, 0.7, 0.33},
            {0.15, 0.13, 0.79, 0.39, 0.94, 0.35, 0.71},
            {0.53, 0.55, 0.77, 0.04, 0.68, 0.15, 0.36}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArray3XOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP3XORSinus(mlp As clsMLPGeneric,
            Optional nbIterations% = 200,
            Optional expectedLoss# = 0.0#,
            Optional learningRate! = 0.1!,
            Optional weightAdjustment! = 0.1!,
            Optional gain! = 1,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Init3XOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
        mlp.SetActivationFunction(enumActivationFunction.Sinus, gain)

        mlp.InitializeWeights(1, {
            {0.41, 0.31, 0.33, 0.61, 0.57, 0.59, 0.73},
            {0.65, 0.67, 0.25, 0.64, 0.65, 0.56, 0.73},
            {0.4, 0.78, 0.24, 0.16, 0.79, 0.39, 0.64},
            {0.89, 0.48, 0.61, 0.83, 0.46, 0.93, 0.75},
            {0.46, 0.13, 0.26, 0.27, 0.14, 0.59, 0.26},
            {0.23, 0.54, 0.45, 0.4, 0.93, 0.9, 0.98}})
        mlp.InitializeWeights(2, {
            {0.93, 0.49, 0.22, 0.1, 0.84, 0.48, 0.33},
            {0.49, 0.39, 0.93, 0.59, 0.22, 0.76, 0.41},
            {0.85, 1.0, 0.74, 0.13, 0.8, 0.9, 0.21}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArray3XOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP3XORELU(mlp As clsMLPGeneric,
            Optional nbIterations% = 250,
            Optional expectedLoss# = 0.01#,
            Optional learningRate! = 0.01!,
            Optional weightAdjustment! = 0,
            Optional gain! = 0.3!,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Init3XOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
        mlp.SetActivationFunction(enumActivationFunction.ELU, gain)

        mlp.InitializeWeights(1, {
            {-0.49, 0.5, 0.06, -0.35, 0.25, -0.16, -0.03},
            {-0.03, -0.26, 0.32, 0.4, 0.38, 0.49, -0.37},
            {0.3, 0.44, -0.09, -0.16, -0.27, 0.26, 0.39},
            {-0.09, 0.37, 0.36, -0.09, -0.41, -0.48, -0.03},
            {0.35, 0.37, -0.41, -0.02, 0.09, -0.46, -0.13},
            {0.06, 0.06, 0.18, 0.27, -0.21, -0.47, 0.26}})
        mlp.InitializeWeights(2, {
            {0.05, -0.14, 0.27, -0.1, 0.21, 0.16, -0.27},
            {-0.19, -0.44, 0.34, 0.23, 0.41, 0.23, -0.49},
            {-0.12, -0.36, 0.11, -0.47, -0.35, 0.09, 0.36}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArray3XOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP3XORMish(mlp As clsMLPGeneric,
            Optional nbIterations% = 1000,
            Optional expectedLoss# = 0.0#,
            Optional learningRate! = 0.01!,
            Optional weightAdjustment! = 0.005!,
            Optional gain! = 1.0!,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Init3XOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
        mlp.SetActivationFunction(enumActivationFunction.Mish, gain)

        mlp.InitializeWeights(1, {
            {0.14, 0.23, 0.04, 0.27, 0.28, 0.04, 0.27},
            {0.05, 0.29, 0.06, 0.06, 0.22, 0.18, 0.06},
            {0.09, 0.01, 0.27, 0.28, 0.02, 0.17, 0.16},
            {0.03, 0.04, 0.07, 0.14, 0.01, 0.27, 0.24},
            {0.17, 0.26, 0.03, 0.21, 0.17, 0.08, 0.17},
            {0.02, 0.27, 0.11, 0.04, 0.15, 0.23, 0.06}})
        mlp.InitializeWeights(2, {
            {0.08, 0.27, 0.08, 0.05, 0.07, 0.11, 0.12},
            {0.21, 0.2, 0.29, 0.09, 0.04, 0.18, 0.12},
            {0.2, 0.27, 0.14, 0.28, 0.07, 0.04, 0.18}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArray3XOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

#End Region

#Region "Iris flower standard tests"

    Public Sub TestMLPIrisFlowerAnalogTanh(mlp As clsMLPGeneric,
        Optional nbIterations% = 200,
        Optional expectedSuccess# = 0.967#,
        Optional expectedSuccessPrediction# = 0.967#,
        Optional expectedLoss# = 0.031#,
        Optional learningRate! = 0.1!,
        Optional weightAdjustment! = 0.1!,
        Optional gain! = 2,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        mlp.Initialize(learningRate, weightAdjustment)
        InitIrisFlowerAnalog4Layers(mlp)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain)

        mlp.InitializeWeights(1, {
            {0.1, 0.31, 0.24, -0.45, -0.2},
            {-0.28, -0.38, -0.26, -0.18, 0.32},
            {0.39, -0.05, 0.27, -0.03, -0.09},
            {-0.45, -0.15, 0.46, -0.33, 0.15},
            {0.28, -0.23, -0.5, -0.03, -0.09},
            {-0.03, 0.12, -0.44, 0.09, 0.27},
            {-0.19, 0.21, 0.4, -0.46, 0.41},
            {0.23, -0.3, 0.27, 0.07, -0.2},
            {0.25, 0.26, -0.36, 0.1, 0.32}})
        mlp.InitializeWeights(2, {
            {-0.31, 0.2, -0.1, 0.28, -0.02, -0.11, -0.31, -0.23, 0.32, 0.04},
            {0.24, 0.08, 0.39, 0.25, -0.32, 0.22, -0.45, 0.08, 0.23, 0.37},
            {-0.07, -0.17, -0.34, -0.32, 0.1, -0.34, -0.3, -0.48, 0.01, 0.37},
            {0.37, 0.09, -0.22, 0.41, 0.16, 0.32, -0.21, 0.43, 0.02, 0.14},
            {-0.28, -0.14, 0.49, 0.12, 0.38, 0.33, 0.48, -0.38, -0.47, 0.21},
            {0.06, 0.37, 0.45, 0.26, -0.3, -0.42, 0.12, -0.21, 0.29, -0.47},
            {-0.44, 0.3, -0.47, 0.37, 0.21, -0.2, -0.03, 0.1, 0.02, 0.4},
            {0.25, 0.15, -0.29, -0.26, -0.29, -0.36, -0.17, 0.04, -0.19, 0.27},
            {-0.32, -0.42, 0.49, 0.3, -0.2, 0.5, 0.15, 0.01, 0.21, 0.08}})
        mlp.InitializeWeights(3, {
            {-0.3, -0.5, -0.36, 0.07, -0.08, -0.17, -0.38, 0.4, 0.08, -0.14}})

        mlp.minimalSuccessTreshold = 0.2
        mlp.Train(learningMode)

        Dim success! = mlp.successPC
        Dim successRounded# = Math.Round(success, 3)
        Assert.AreEqual(True, successRounded >= expectedSuccess)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 3)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        If mlp.successPC = 1 Then
            Dim expectedOutput = m_targetArrayIrisFlowerAnalogTest
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)
        End If

        mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
            m_targetArrayIrisFlowerAnalogTest, nbOutputs:=1)
        Dim successPrediction! = mlp.successPC
        Dim successPredictionRounded# = Math.Round(successPrediction, 3)
        Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

    End Sub

    Public Sub TestMLPIrisFlowerAnalogTanh2(mlp As clsMLPGeneric,
        Optional nbIterations% = 50,
        Optional expectedSuccess# = 0.983#,
        Optional expectedSuccessPrediction# = 0.967#,
        Optional expectedLoss# = 0.06#,
        Optional learningRate! = 0.1!,
        Optional weightAdjustment! = 0.1!,
        Optional gain! = 2,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        mlp.Initialize(learningRate, weightAdjustment)
        InitIrisFlowerAnalog4Layers(mlp)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain)

        mlp.InitializeWeights(1, {
            {-0.79, -0.65, 0.35, 0.73, -0.75},
            {-0.6, 0.12, 0.27, -0.99, -0.08},
            {0.39, -0.48, 0.76, -0.42, -0.83},
            {-0.71, -0.64, -0.42, -0.55, 0.49},
            {0.19, 0.56, 0.24, -0.74, 0.93},
            {0.08, 0.56, -0.55, -0.81, 0.63},
            {0.47, 0.35, -0.18, -0.51, -0.3},
            {0.75, -0.97, 0.6, 0.43, 0.54},
            {0.4, 0.3, 0.47, -0.41, 0.75}})
        mlp.InitializeWeights(2, {
            {-0.48, 0.53, -0.18, 0.14, 0.97, 0.43, -0.11, 0.55, -0.3, 0.66},
            {-0.37, 0.63, 0.91, 0.86, 0.66, 0.31, -0.8, 0.41, 0.73, -0.44},
            {0.49, 0.15, -0.14, 0.34, 0.04, 0.45, -0.24, 0.75, 0.55, 0.99},
            {-0.93, -0.06, -0.67, -0.88, 0.09, -0.43, 0.36, 0.45, -0.49, 0.15},
            {-0.03, 0.47, 0.87, 0.88, -0.65, 0.88, 0.09, -0.76, 0.02, -0.78},
            {0.49, 0.59, -0.87, 0.71, -0.65, -0.81, 0.87, 0.18, -0.09, -0.99},
            {-0.91, 0.62, -0.02, -0.3, 0.31, 0.9, -0.62, -0.38, -0.66, -0.92},
            {0.91, 0.74, 0.1, -0.7, -0.04, 0.88, -0.04, 0.64, 0.54, 0.81},
            {-0.11, 0.73, 0.11, -0.03, 0.8, 0.63, -0.95, -0.92, 0.79, -0.9}})
        mlp.InitializeWeights(3, {
            {-0.78, -0.38, -0.43, 0.16, -0.28, 0.4, -0.7, 0.76, 0.56, -0.95}})

        mlp.minimalSuccessTreshold = 0.2
        mlp.Train(learningMode)

        Dim success! = mlp.successPC
        Dim successRounded# = Math.Round(success, 3)
        Assert.AreEqual(True, successRounded >= expectedSuccess)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 3)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        If mlp.successPC = 1 Then
            Dim expectedOutput = m_targetArrayIrisFlowerAnalogTest
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)
        End If

        mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
            m_targetArrayIrisFlowerAnalogTest, nbOutputs:=1)
        Dim successPrediction! = mlp.successPC
        Dim successPredictionRounded# = Math.Round(successPrediction, 3)
        Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

    End Sub

    Public Sub TestMLPIrisFlowerAnalogSigmoid(mlp As clsMLPGeneric,
        Optional nbIterations% = 150,
        Optional expectedSuccess# = 0.958#,
        Optional expectedSuccessPrediction# = 0.967#,
        Optional expectedLoss# = 0.072#,
        Optional learningRate! = 0.2!,
        Optional weightAdjustment! = 0.0!,
        Optional gain! = 1.1!,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        mlp.Initialize(learningRate, weightAdjustment)
        InitIrisFlowerAnalog4Layers(mlp)
        mlp.InitializeStruct({4, 12, 8, 1}, addBiasColumn:=True)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)

        mlp.InitializeWeights(1, {
            {0.35, -0.16, -0.09, 0.22, -0.47},
            {-0.46, 0.33, 0.21, 0.43, 0.1},
            {0.47, 0.19, -0.21, -0.42, 0.22},
            {0.18, 0.16, 0.47, 0.45, -0.38},
            {0.02, 0.36, 0.31, 0.46, -0.45},
            {-0.23, 0.17, -0.19, -0.49, -0.42},
            {0.36, 0.16, -0.2, 0.09, 0.4},
            {-0.06, 0.46, 0.23, 0.42, 0.45},
            {-0.04, -0.25, -0.24, -0.03, 0.17},
            {0.03, 0.41, -0.16, -0.13, 0.13},
            {0.27, 0.05, -0.09, 0.08, -0.17},
            {-0.02, 0.09, 0.26, 0.48, 0.27}})
        mlp.InitializeWeights(2, {
            {-0.35, -0.1, 0.43, 0.02, -0.39, -0.47, 0.49, 0.32, -0.26, 0.2, -0.27, 0.09, 0.5},
            {0.16, -0.23, 0.1, -0.15, -0.21, 0.02, -0.13, -0.17, 0.44, -0.12, -0.19, -0.19, -0.25},
            {0.22, -0.24, -0.08, 0.35, -0.3, 0.25, -0.35, 0.29, -0.44, -0.18, 0.24, -0.14, 0.15},
            {0.04, -0.09, 0.02, -0.37, 0.43, 0.04, 0.47, -0.09, 0.42, -0.44, 0.38, -0.25, -0.03},
            {-0.04, -0.09, 0.32, -0.04, 0.05, -0.29, 0.3, 0.27, -0.23, -0.26, -0.1, -0.11, -0.02},
            {-0.06, -0.3, 0.1, 0.45, 0.37, 0.49, 0.14, 0.48, 0.45, 0.32, 0.3, -0.06, 0.28},
            {-0.16, 0.34, 0.3, -0.3, 0.05, 0.1, 0.23, -0.21, 0.24, 0.47, 0.09, 0.27, 0.03},
            {0.06, 0.38, -0.23, -0.33, -0.38, 0.24, -0.06, -0.45, 0.03, -0.29, -0.19, -0.39, -0.49}})
        mlp.InitializeWeights(3, {
            {-0.25, 0.33, -0.02, -0.05, -0.49, 0.16, -0.39, -0.07, -0.04}})

        mlp.minimalSuccessTreshold = 0.2

        mlp.Train(learningMode)

        Dim success! = mlp.successPC
        Dim successRounded# = Math.Round(success, 3)
        Assert.AreEqual(True, successRounded >= expectedSuccess)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 3)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        If mlp.successPC = 1 Then
            Dim expectedOutput = m_targetArrayIrisFlowerAnalogTest
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)
        End If

        mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
            m_targetArrayIrisFlowerAnalogTest, nbOutputs:=1)
        Dim successPrediction! = mlp.successPC
        Dim successPredictionRounded# = Math.Round(successPrediction, 3)
        Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

    End Sub

    Public Sub TestMLPIrisFlowerAnalogGaussian(mlp As clsMLPGeneric,
        Optional nbIterations% = 100,
        Optional expectedSuccess# = 0.967#,
        Optional expectedSuccessPrediction# = 0.933#,
        Optional expectedLoss# = 0.07#,
        Optional learningRate! = 0.2!,
        Optional weightAdjustment! = 0.2!,
        Optional gain! = 0.2,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        mlp.Initialize(learningRate, weightAdjustment)
        InitIrisFlowerAnalog4Layers(mlp)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Gaussian, gain)

        mlp.InitializeWeights(1, {
            {-0.49, -0.39, 0.24, 0.25, 0.07},
            {-0.21, -0.42, -0.37, -0.09, 0.2},
            {-0.03, -0.27, 0.26, 0.23, 0.06},
            {0.25, -0.31, -0.05, -0.1, 0.17},
            {-0.19, -0.42, 0.37, 0.18, 0.38},
            {-0.12, -0.22, 0.35, -0.42, 0.06},
            {-0.26, -0.48, 0.41, -0.19, 0.21},
            {0.49, -0.12, -0.24, -0.48, -0.15},
            {0.27, 0.13, 0.2, 0.4, -0.04}})
        mlp.InitializeWeights(2, {
            {-0.48, 0.11, 0.18, -0.08, -0.4, -0.46, -0.45, 0.18, 0.23, 0.19},
            {-0.47, 0.37, -0.43, 0.19, 0.41, 0.3, -0.22, -0.17, -0.35, -0.24},
            {-0.05, -0.17, -0.06, -0.48, 0.07, -0.13, 0.43, -0.08, -0.46, 0.39},
            {0.18, -0.12, 0.47, -0.28, 0.36, 0.26, 0.1, -0.08, 0.48, 0.02},
            {-0.31, -0.15, -0.32, 0.12, 0.18, -0.39, -0.19, 0.06, -0.39, 0.05},
            {-0.2, 0.06, 0.14, -0.48, 0.19, -0.33, 0.17, 0.35, 0.23, -0.4},
            {0.13, 0.13, 0.33, -0.49, 0.15, 0.41, 0.35, 0.33, -0.35, -0.3},
            {0.36, -0.15, 0.12, -0.43, -0.4, -0.35, 0.32, -0.16, -0.04, -0.43},
            {-0.13, -0.19, -0.1, -0.12, 0.32, -0.17, -0.3, -0.34, 0.09, 0.43}})
        mlp.InitializeWeights(3, {
            {0.07, -0.25, -0.08, 0.06, 0.22, -0.16, 0.11, -0.48, -0.29, -0.05}})

        mlp.minimalSuccessTreshold = 0.2

        mlp.Train(learningMode)

        Dim success! = mlp.successPC
        Dim successRounded# = Math.Round(success, 3)
        Assert.AreEqual(True, successRounded >= expectedSuccess)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 3)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        If mlp.successPC = 1 Then
            Dim expectedOutput = m_targetArrayIrisFlowerAnalogTest
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)
        End If

        mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
            m_targetArrayIrisFlowerAnalogTest, nbOutputs:=1)
        Dim successPrediction! = mlp.successPC
        Dim successPredictionRounded# = Math.Round(successPrediction, 3)
        Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

    End Sub

    Public Sub TestMLPIrisFlowerLogicalTanh(mlp As clsMLPGeneric,
        Optional nbIterations% = 1500,
        Optional expectedSuccess# = 0.994#,
        Optional expectedSuccessPrediction# = 0.978#,
        Optional expectedLoss# = 0.025#,
        Optional learningRate! = 0.1!,
        Optional weightAdjustment! = 0.1!,
        Optional gain! = 2,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        ' 97.8% prediction, 99.4% learning with 1500 iterations

        mlp.Initialize(learningRate, weightAdjustment)
        InitIrisFlowerLogical4Layers(mlp)

        mlp.minimalSuccessTreshold = 0.3
        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain)

        mlp.InitializeWeights(1, {
            {0.74, -0.46, -0.07, -0.78, -0.76},
            {-0.51, 0.88, 0.34, 0.85, -0.28},
            {0.17, 1.04, -0.68, -0.62, 0.07},
            {0.69, -0.86, -0.69, 0.05, 0.52},
            {-0.63, 0.84, 0.63, 0.05, -0.69},
            {-0.65, -0.58, -0.78, -0.27, 0.72},
            {-0.3, -0.15, 0.2, -1.04, 0.85},
            {-0.11, 0.68, 0.54, -0.51, -0.97},
            {-0.04, 0.58, -0.79, 0.58, 0.82},
            {1.12, 0.19, 0.17, -0.32, -0.73},
            {0.53, 0.42, -0.68, -0.65, 0.79},
            {0.6, 0.06, 0.71, -0.73, -0.75},
            {-0.76, 0.49, -0.32, -0.66, 0.78},
            {-0.75, -0.35, -0.65, -0.26, -0.89},
            {-0.92, -0.47, 0.76, -0.31, -0.47},
            {-0.28, 0.78, 1.11, 0.16, -0.08}})
        mlp.InitializeWeights(2, {
            {0.18, 0.22, 0.22, -0.56, -0.52, 0.15, -0.22, -0.05, -0.45, 0.57, 0.48, 0.34, 0.18, -0.24, -0.42, 0.19, 0.03},
            {0.59, 0.41, 0.4, -0.14, -0.18, -0.42, -0.16, -0.35, -0.03, 0.55, 0.3, 0.4, 0.07, -0.56, 0.09, -0.15, 0.09},
            {-0.16, -0.21, -0.47, -0.26, -0.56, -0.17, 0.4, 0.47, -0.51, -0.15, -0.43, -0.19, 0.2, 0.01, -0.31, 0.19, 0.44},
            {0.47, -0.48, 0.32, -0.55, -0.26, 0.04, -0.24, 0.39, 0.09, 0.25, 0.09, 0.18, 0.06, 0.59, -0.2, -0.11, -0.55},
            {0.5, 0.37, 0.06, -0.31, 0.52, -0.03, 0.59, 0.01, -0.27, 0.34, -0.33, 0.44, 0.38, -0.07, 0.0, -0.4, 0.24},
            {-0.45, -0.1, 0.2, -0.02, -0.64, 0.43, 0.43, -0.01, -0.04, 0.24, -0.35, -0.37, 0.05, -0.1, 0.44, 0.63, 0.1},
            {0.1, -0.07, -0.45, 0.13, 0.48, 0.19, 0.28, 0.09, -0.53, 0.28, -0.45, 0.49, 0.29, -0.28, -0.5, -0.04, 0.39},
            {0.3, 0.5, 0.33, 0.12, 0.43, -0.58, 0.3, 0.3, -0.05, 0.36, 0.28, -0.28, 0.28, -0.18, -0.09, 0.52, -0.31}})
        mlp.InitializeWeights(3, {
            {0.43, -0.53, -0.53, -0.5, -0.28, 0.32, 0.47, 0.5, -0.57},
            {-0.86, 0.38, 0.12, 0.42, -0.16, -0.58, 0.53, 0.36, 0.34},
            {0.59, -0.53, 0.72, 0.35, -0.11, -0.22, 0.55, -0.05, 0.56}})

        mlp.Train()

        Dim success! = mlp.successPC
        Dim successRounded# = Math.Round(success, 3)
        Assert.AreEqual(True, successRounded >= expectedSuccess)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 3)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        If mlp.successPC = 1 AndAlso mlp.minimalSuccessTreshold <= 0.05! Then
            Dim expectedOutput = m_targetArrayIrisFlowerLogicalTrain
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)
        End If

        mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
            m_targetArrayIrisFlowerLogicalTest, nbOutputs:=3)
        Dim successPrediction! = mlp.successPC
        Dim successPredictionRounded# = Math.Round(successPrediction, 3)
        Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

    End Sub

    Public Sub TestMLPIrisFlowerLogicalSigmoid(mlp As clsMLPGeneric,
        Optional nbIterations% = 1000,
        Optional expectedSuccess# = 0.986#,
        Optional expectedSuccessPrediction# = 0.978#,
        Optional expectedLoss# = 0.058#,
        Optional learningRate! = 0.1!,
        Optional weightAdjustment! = 0.1!,
        Optional gain! = 1,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        ' 97.8% prediction, 98.6% learning with 200 iterations

        mlp.Initialize(learningRate, weightAdjustment)
        InitIrisFlowerLogical4Layers(mlp)

        mlp.minimalSuccessTreshold = 0.3
        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)

        mlp.InitializeWeights(1, {
            {0.86, -0.47, 0.71, -0.61, 0.37},
            {-0.88, -0.19, -0.68, 0.74, 0.36},
            {-0.03, -0.78, -0.69, 0.32, -0.87},
            {-0.23, -0.1, -0.26, -1.09, 0.8},
            {0.59, 0.82, 0.81, 0.52, 0.14},
            {-0.71, -0.56, 0.99, 0.04, -0.4},
            {-0.57, -0.44, 0.75, -0.82, 0.46},
            {-0.19, -0.41, 0.7, -0.86, 0.73},
            {-0.78, -0.33, -0.52, -0.4, -0.9},
            {-0.41, 0.03, -0.91, 0.41, 0.9},
            {-0.54, -0.49, 0.61, -0.62, 0.82},
            {0.89, -0.29, -0.98, 0.33, 0.04},
            {1.22, -0.36, 0.49, 0.3, 0.09},
            {-0.5, -0.69, -0.74, 0.77, 0.29},
            {-1.0, 0.2, -0.34, -0.85, 0.3},
            {-0.24, 0.66, 0.42, 0.61, -0.96}})
        mlp.InitializeWeights(2, {
            {0.15, -0.38, -0.24, 0.08, -0.57, 0.41, -0.01, 0.22, -0.36, 0.4, -0.54, -0.29, -0.31, 0.01, -0.49, 0.33, 0.27},
            {0.26, 0.39, 0.49, -0.44, -0.13, -0.44, -0.55, 0.39, 0.27, 0.53, 0.06, 0.19, -0.15, -0.28, 0.33, -0.14, -0.15},
            {0.43, 0.17, -0.43, -0.48, 0.08, 0.19, 0.28, -0.05, 0.47, -0.49, 0.13, 0.43, -0.17, 0.4, -0.42, 0.35, 0.23},
            {0.44, 0.28, 0.24, -0.37, -0.27, -0.48, -0.27, 0.25, -0.21, 0.43, 0.45, -0.46, 0.32, 0.07, -0.23, 0.36, 0.35},
            {0.15, -0.11, 0.23, 0.22, -0.46, -0.46, -0.45, 0.24, -0.05, -0.52, -0.08, 0.54, -0.16, -0.56, -0.09, 0.42, 0.21},
            {-0.21, 0.63, 0.19, -0.33, 0.2, 0.17, 0.49, -0.24, -0.29, 0.32, 0.4, -0.11, -0.5, -0.02, -0.2, -0.61, -0.01},
            {-0.28, 0.23, -0.4, -0.52, 0.34, 0.16, 0.11, -0.06, 0.31, 0.48, 0.33, 0.37, 0.48, 0.41, 0.02, -0.33, -0.4},
            {-0.33, -0.26, -0.35, 0.13, 0.56, 0.34, 0.36, -0.18, 0.44, -0.12, -0.19, -0.27, -0.19, 0.14, -0.5, 0.47, 0.46}})
        mlp.InitializeWeights(3, {
            {0.54, 0.39, 0.32, 0.22, 0.06, -0.27, -0.77, 0.52, 0.66},
            {0.32, -0.66, -0.02, -0.23, 0.6, -0.33, -0.46, 0.74, -0.39},
            {-0.62, 0.45, 0.54, -0.3, 0.6, -0.22, 0.26, 0.62, -0.37}})

        mlp.Train(learningMode)

        Dim success! = mlp.successPC
        Dim successRounded# = Math.Round(success, 3)
        Assert.AreEqual(True, successRounded >= expectedSuccess)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 3)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        If mlp.successPC = 1 AndAlso mlp.minimalSuccessTreshold <= 0.05! Then
            Dim expectedOutput = m_targetArrayIrisFlowerLogicalTrain
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)
        End If

        mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
            m_targetArrayIrisFlowerLogicalTest, nbOutputs:=3)
        Dim successPrediction! = mlp.successPC
        Dim successPredictionRounded# = Math.Round(successPrediction, 3)
        Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

    End Sub

    Public Sub TestMLPIrisFlowerLogicalGaussian(mlp As clsMLPGeneric,
        Optional nbIterations% = 100,
        Optional expectedSuccess# = 0.953#,
        Optional expectedSuccessPrediction# = 0.967#,
        Optional expectedLoss# = 0.06#,
        Optional learningRate! = 0.2!,
        Optional weightAdjustment! = 0.2!,
        Optional gain! = 0.1,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        ' 96.7% prediction, 95.3% learning with 100 iterations

        mlp.Initialize(learningRate, weightAdjustment)
        InitIrisFlowerLogical4Layers(mlp)
        mlp.InitializeStruct({4, 9, 9, 3}, addBiasColumn:=True)

        mlp.minimalSuccessTreshold = 0.3
        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Gaussian, gain)

        mlp.InitializeWeights(1, {
            {-0.15, -0.26, 0.3, -0.18, -0.5},
            {0.1, 0.29, -0.41, -0.16, -0.39},
            {0.17, -0.38, 0.1, -0.2, -0.13},
            {0.4, 0.36, 0.35, 0.22, 0.38},
            {-0.16, 0.23, 0.2, -0.11, -0.09},
            {-0.32, 0.21, -0.43, -0.4, 0.47},
            {-0.31, 0.48, 0.16, 0.22, 0.3},
            {0.21, 0.11, -0.15, -0.02, 0.38},
            {0.41, 0.44, 0.24, -0.12, 0.03}})
        mlp.InitializeWeights(2, {
            {0.34, 0.48, -0.17, -0.06, 0.44, 0.11, 0.33, -0.03, 0.41, -0.43},
            {0.12, 0.11, -0.41, -0.07, -0.43, -0.48, -0.27, -0.36, -0.01, -0.36},
            {0.44, 0.08, 0.21, 0.16, -0.21, -0.16, -0.26, 0.5, -0.07, 0.3},
            {0.47, -0.48, 0.38, -0.16, 0.23, 0.24, 0.16, -0.47, 0.08, -0.35},
            {-0.32, 0.31, 0.22, -0.41, -0.5, -0.23, -0.42, -0.09, 0.36, 0.18},
            {0.29, -0.25, -0.26, 0.09, -0.24, -0.23, 0.16, -0.35, -0.07, -0.24},
            {0.09, -0.02, -0.02, -0.23, 0.15, 0.09, -0.29, -0.41, -0.16, -0.22},
            {0.07, -0.39, 0.41, 0.49, 0.26, 0.27, 0.49, 0.07, -0.21, -0.26},
            {-0.23, 0.33, -0.36, 0.25, -0.5, 0.21, 0.28, -0.21, -0.42, -0.04}})
        mlp.InitializeWeights(3, {
            {-0.42, 0.1, -0.18, 0.06, 0.21, -0.16, -0.05, -0.06, -0.09, -0.44},
            {0.49, -0.44, 0.08, -0.39, 0.18, -0.16, -0.26, 0.33, -0.01, -0.21},
            {-0.41, 0.36, -0.31, 0.5, 0.26, -0.16, 0.22, -0.23, 0.44, 0.31}})

        mlp.Train(learningMode)

        Dim success! = mlp.successPC
        Dim successRounded# = Math.Round(success, 3)
        Assert.AreEqual(True, successRounded >= expectedSuccess)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 3)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        If mlp.successPC = 1 AndAlso mlp.minimalSuccessTreshold <= 0.05! Then
            Dim expectedOutput = m_targetArrayIrisFlowerLogicalTrain
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)
        End If

        mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
            m_targetArrayIrisFlowerLogicalTest, nbOutputs:=3)
        Dim successPrediction! = mlp.successPC
        Dim successPredictionRounded# = Math.Round(successPrediction, 3)
        Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

    End Sub

    Public Sub TestMLPIrisFlowerLogicalSinus(mlp As clsMLPGeneric,
        Optional nbIterations% = 200,
        Optional expectedSuccess# = 0.933#,
        Optional expectedSuccessPrediction# = 0.978#,
        Optional expectedLoss# = 0.1#,
        Optional learningRate! = 0.05!,
        Optional weightAdjustment! = 0.05!,
        Optional gain! = 0.9,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        ' 97.8% prediction, 93.3% learning with 200 iterations

        mlp.Initialize(learningRate, weightAdjustment)
        InitIrisFlowerLogical4Layers(mlp)
        mlp.InitializeStruct({4, 9, 9, 3}, addBiasColumn:=True)

        mlp.minimalSuccessTreshold = 0.3
        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Sinus, gain)

        mlp.InitializeWeights(1, {
            {0.07, -0.14, 0.17, -0.35, -0.03},
            {-0.32, -0.28, -0.27, -0.19, -0.29},
            {-0.02, -0.28, -0.45, 0.32, -0.44},
            {0.07, 0.39, -0.36, -0.28, -0.27},
            {-0.21, 0.4, -0.49, -0.17, 0.24},
            {-0.3, -0.16, -0.45, -0.23, -0.12},
            {0.12, -0.12, -0.38, -0.25, 0.42},
            {0.49, -0.5, -0.12, -0.02, -0.11},
            {0.05, -0.24, -0.46, 0.09, -0.47}})
        mlp.InitializeWeights(2, {
            {-0.16, 0.23, -0.33, -0.37, 0.3, -0.26, 0.12, -0.22, -0.05, 0.16},
            {0.32, 0.23, 0.18, -0.48, 0.4, -0.24, 0.5, 0.44, -0.39, 0.1},
            {0.18, 0.19, 0.31, 0.14, 0.39, -0.36, 0.34, 0.26, 0.39, -0.15},
            {0.25, -0.21, -0.35, -0.03, 0.36, 0.1, 0.38, -0.14, -0.04, -0.37},
            {0.5, -0.41, 0.17, 0.09, -0.39, -0.24, -0.18, -0.14, 0.08, -0.36},
            {0.05, -0.18, 0.43, 0.49, -0.15, 0.15, 0.42, 0.03, -0.26, 0.16},
            {-0.09, 0.36, -0.11, -0.4, 0.41, 0.03, 0.08, -0.29, -0.34, -0.21},
            {-0.13, 0.13, -0.02, 0.48, 0.1, 0.08, -0.48, -0.28, 0.04, 0.13},
            {0.32, -0.02, -0.32, 0.26, 0.31, -0.08, -0.14, -0.35, -0.38, -0.29}})
        mlp.InitializeWeights(3, {
            {0.18, -0.14, -0.37, 0.3, 0.22, -0.36, 0.2, 0.07, 0.18, 0.08},
            {0.18, -0.38, -0.3, -0.22, 0.27, 0.42, 0.35, 0.45, -0.11, 0.27},
            {0.12, 0.2, 0.49, 0.1, 0.35, 0.43, 0.18, 0.13, -0.21, -0.01}})

        mlp.Train(learningMode)

        Dim success! = mlp.successPC
        Dim successRounded# = Math.Round(success, 3)
        Assert.AreEqual(True, successRounded >= expectedSuccess)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 3)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        If mlp.successPC = 1 AndAlso mlp.minimalSuccessTreshold <= 0.05! Then
            Dim expectedOutput = m_targetArrayIrisFlowerLogicalTrain
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)
        End If

        mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
            m_targetArrayIrisFlowerLogicalTest, nbOutputs:=3)
        Dim successPrediction! = mlp.successPC
        Dim successPredictionRounded# = Math.Round(successPrediction, 3)
        Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

    End Sub

#End Region

#Region "Sunspot tests"

    Public Sub TestMLPSunspot1Sigmoid(mlp As clsMLPGeneric,
        Optional nbIterations% = 200,
        Optional expectedSuccess# = 0.7#,
        Optional expectedSuccessPrediction# = 0.9#,
        Optional expectedLoss# = 0.08#,
        Optional learningRate! = 0.1!,
        Optional weightAdjustment! = 0.1!,
        Optional gain! = 1,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        InitSunspot1(mlp)
        mlp.nbLinesToLearn = 49 ' With 49 samples, works only using ThreadCount = 1: multithread does not work yet!
        mlp.InitializeStruct({7, 13, 1}, addBiasColumn:=True)
        mlp.Initialize(learningRate, weightAdjustment)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)

        mlp.InitializeWeights(1, {
            {-0.44, 0.16, -0.28, -0.05, -0.05, 0.11, 0.45, 0.32, 0.24, -0.14, -0.27},
            {-0.28, 0.2, -0.02, -0.3, -0.29, 0.09, 0.29, 0.28, 0.46, 0.3, 0.3},
            {-0.25, 0.29, 0.35, 0.42, 0.2, -0.37, -0.19, 0.24, -0.03, 0.37, 0.43},
            {0.42, 0.38, 0.29, -0.33, 0.14, 0.07, 0.21, -0.17, -0.5, -0.36, -0.29},
            {0.05, -0.41, -0.45, -0.13, -0.09, 0.03, -0.02, -0.47, -0.45, 0.41, -0.24},
            {-0.09, -0.07, 0.09, 0.03, 0.41, 0.32, 0.01, 0.5, 0.39, -0.14, -0.21},
            {0.29, 0.09, -0.09, -0.46, -0.4, -0.35, -0.29, -0.26, -0.03, 0.3, -0.39},
            {0.08, -0.19, 0.33, 0.15, 0.25, 0.4, -0.29, 0.48, 0.34, 0.38, -0.5},
            {0.13, -0.12, 0.24, 0.4, -0.39, 0.16, 0.39, 0.32, -0.5, -0.25, 0.35},
            {-0.24, -0.19, -0.04, 0.46, -0.13, -0.07, -0.17, 0.32, 0.31, -0.06, -0.05},
            {-0.2, 0.36, -0.21, 0.2, -0.24, -0.43, 0.12, 0.28, 0.41, 0.02, -0.09},
            {0.28, 0.47, -0.47, -0.19, -0.3, -0.46, 0.05, -0.15, 0.15, 0.3, 0.35},
            {0.23, -0.45, 0.02, -0.31, 0.29, 0.02, 0.28, 0.15, -0.48, -0.43, 0.06}})
        mlp.InitializeWeights(2, {
            {-0.33, -0.42, 0.38, 0.11, -0.09, -0.1, 0.32, -0.3, -0.28, -0.16, -0.17, 0.35, 0.02, 0.44}})

        mlp.minimalSuccessTreshold = 0.1

        mlp.Train(learningMode)

        Dim success! = mlp.successPC
        Dim successRounded# = Math.Round(success, 3)
        Assert.AreEqual(True, successRounded >= expectedSuccess)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 3)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        mlp.TestAllSamples(mlp.inputArrayTest, mlp.targetArrayTest, nbOutputs:=1)
        Dim successPrediction! = mlp.successPC
        Dim successPredictionRounded# = Math.Round(successPrediction, 3)
        Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

    End Sub

    Public Sub TestMLPSunspot1Tanh(mlp As clsMLPGeneric,
        Optional nbIterations% = 200,
        Optional expectedSuccess# = 0.7#,
        Optional expectedSuccessPrediction# = 1.0#,
        Optional expectedLoss# = 0.08#,
        Optional learningRate! = 0.1!,
        Optional weightAdjustment! = 0.1!,
        Optional gain! = 1,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        InitSunspot1(mlp)
        mlp.InitializeStruct({7, 13, 1}, addBiasColumn:=True)
        mlp.Initialize(learningRate, weightAdjustment)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain)

        mlp.InitializeWeights(1, {
            {0.12, 0.09, 0.27, -0.48, 0.18, -0.39, -0.35, 0.49},
            {0.03, 0.36, -0.41, -0.35, 0.26, 0.38, 0.41, 0.02},
            {0.39, 0.3, 0.32, -0.23, -0.3, -0.18, -0.39, -0.43},
            {0.1, -0.41, -0.03, -0.17, -0.27, -0.43, -0.29, -0.02},
            {-0.23, -0.49, -0.25, 0.22, 0.07, 0.27, -0.13, 0.23},
            {0.37, 0.06, -0.04, -0.06, 0.03, 0.47, 0.2, 0.45},
            {-0.07, -0.41, 0.49, 0.15, -0.08, -0.23, 0.16, 0.45},
            {-0.43, -0.48, 0.18, 0.17, -0.2, -0.07, 0.15, 0.11},
            {0.38, 0.24, 0.13, -0.31, 0.46, -0.38, -0.07, -0.42},
            {-0.1, 0.14, 0.36, 0.17, 0.03, 0.14, -0.11, 0.46},
            {-0.12, 0.23, 0.14, -0.38, 0.12, 0.09, 0.48, -0.44},
            {0.12, 0.2, -0.15, -0.45, -0.3, -0.25, -0.47, 0.29},
            {0.42, -0.41, 0.46, 0.38, 0.26, 0.04, -0.19, 0.16}})
        mlp.InitializeWeights(2, {
            {0.41, -0.25, 0.28, -0.09, 0.48, -0.09, -0.06, 0.13, 0.22, -0.21, 0.07, 0.29, 0.03, -0.38}})

        mlp.minimalSuccessTreshold = 0.1

        mlp.Train(learningMode)

        Dim success! = mlp.successPC
        Dim successRounded# = Math.Round(success, 3)
        Assert.AreEqual(True, successRounded >= expectedSuccess)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 3)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        mlp.TestAllSamples(mlp.inputArrayTest, mlp.targetArrayTest, nbOutputs:=1)
        Dim successPrediction! = mlp.successPC
        Dim successPredictionRounded# = Math.Round(successPrediction, 3)
        Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

    End Sub

    Public Sub TestMLPSunspotTanh2(mlp As clsMLPGeneric,
        Optional nbIterations% = 100,
        Optional expectedLearningAccuracy# = 0.931#,
        Optional expectedPredictionAccuracy# = 0.934#,
        Optional expectedSuccess# = 0.737#,
        Optional expectedSuccessPrediction# = 0.8#,
        Optional expectedLoss# = 0.0758#,
        Optional learningRate! = 0.1!,
        Optional weightAdjustment! = 0.1!,
        Optional gain! = 2,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        InitSunspot2(mlp)
        mlp.InitializeStruct({3, 18, 1}, addBiasColumn:=True)
        mlp.Initialize(learningRate, weightAdjustment)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain)

        mlp.InitializeWeights(1, {
            {0.03, -0.47, 0.31, 0.24},
            {0.42, -0.35, 0.3, 0.09},
            {0.11, -0.44, -0.27, 0.15},
            {-0.1, 0.16, 0.24, 0.05},
            {0.2, -0.42, -0.21, 0.44},
            {-0.41, 0.41, 0.06, 0.09},
            {-0.08, 0.09, -0.27, -0.41},
            {0.05, 0.5, -0.08, 0.05},
            {0.03, -0.49, 0.46, -0.47},
            {-0.19, -0.13, 0.23, -0.27},
            {0.32, 0.27, 0.07, -0.16},
            {-0.36, -0.4, -0.25, 0.06},
            {-0.2, 0.31, 0.12, 0.31},
            {-0.35, 0.26, 0.03, 0.12},
            {-0.02, -0.28, -0.19, -0.17},
            {0.42, 0.22, -0.46, 0.11},
            {0.14, 0.18, -0.38, -0.1},
            {0.2, 0.21, -0.26, -0.17}})
        mlp.InitializeWeights(2, {
            {-0.15, -0.44, -0.39, -0.18, -0.16, -0.29, -0.05, -0.18, -0.15, 0.17, 0.29, 0.25, -0.13, 0.11, -0.09, 0.28, -0.02, -0.16, 0.05}})

        mlp.minimalSuccessTreshold = 0.1

        mlp.Train(learningMode)

        Dim success! = mlp.successPC
        Dim successRounded# = Math.Round(success, 3)
        Assert.AreEqual(True, successRounded >= expectedSuccess)

        Dim loss# = mlp.averageError
        Dim learningAccuracy = 1 - loss
        Dim learningAccuracyR = Math.Round(learningAccuracy, 3)
        Assert.AreEqual(True, learningAccuracyR >= expectedLearningAccuracy)
        Dim lossRounded# = Math.Round(loss, 3)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        mlp.TestAllSamples(mlp.inputArrayTest, mlp.targetArrayTest, nbOutputs:=1)
        Dim predictionLoss# = mlp.averageError
        Dim predictionAccuracy = 1 - predictionLoss
        Dim predictionAccuracyR = Math.Round(predictionAccuracy, 3)
        Assert.AreEqual(True, predictionAccuracyR >= expectedPredictionAccuracy)

        Dim successPrediction! = mlp.successPC
        Dim successPredictionRounded# = Math.Round(successPrediction, 3)
        Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

    End Sub

#End Region

End Module
