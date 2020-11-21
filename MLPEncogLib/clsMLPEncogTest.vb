
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Imports Microsoft.VisualStudio.TestTools.UnitTesting

Module modMLPEncogTest

    Sub Main()
        Console.WriteLine("Encog MLP with the classical XOR test.")
        EncogMLPXorTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()
    End Sub

    Public Sub EncogMLPXorTest(Optional nbXor% = 1)

        Dim mlp As New clsMLPEncog

        mlp.ShowMessage("Encog MLP Xor test")
        mlp.ShowMessage("------------------")

        mlp.inputArray = m_inputArrayXOR
        mlp.targetArray = m_targetArrayXOR

        'mlp.nbIterations = 500 ' Sigmoid: works
        mlp.nbIterations = 2000 ' Hyperbolic tangent: works fine
        'mlp.nbIterations = 20000 ' Stochastic

        mlp.Initialize(learningRate:=0)

        mlp.printOutput_ = True
        mlp.printOutputMatrix = False

        If nbXor = 1 Then
            mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR4Layers2331, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR5Layers23331, addBiasColumn:=True)
            mlp.printOutputMatrix = True
            mlp.inputArray = m_inputArrayXOR
            mlp.targetArray = m_targetArrayXOR
        ElseIf nbXor = 2 Then
            mlp.inputArray = m_inputArray2XOR
            mlp.targetArray = m_targetArray2XOR
            mlp.InitializeStruct(m_neuronCount2XOR462, addBiasColumn:=True)
        ElseIf nbXor = 3 Then
            'mlp.nbIterations = 20000
            mlp.inputArray = m_inputArray3XOR
            mlp.targetArray = m_targetArray3XOR
            mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
        End If

        'mlp.SetActivationFunction(enumActivationFunction.Sigmoid)
        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

        mlp.Randomize()

        mlp.PrintWeights()

        WaitForKeyToStart()

        mlp.TrainVector() ' Works fine

        mlp.ShowMessage("Encog MLP Xor test: Done.")

        If nbXor > 1 Then Exit Sub

        WaitForKeyToContinue("Press a key to print MLP weights")
        mlp.PrintWeights()

    End Sub

End Module

Namespace EncogMLP

    <TestClass()>
    Public Class clsMLPEncogTest

        Private m_mlp As New clsMLPEncog

        'Private m_mlp As New clsMLPClassic ' 1/13 success
        'Private m_mlp As New NetworkOOP.MultilayerPerceptron ' 2/12 success
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

        Private Sub InitIrisFlowerAnalog()
            m_mlp.Initialize(learningRate:=0)
            m_mlp.inputArray = m_inputArrayIrisFlower
            m_mlp.targetArray = m_targetArrayIrisFlowerAnalog
            m_mlp.InitializeStruct(m_neuronCountIrisFlowerAnalog, addBiasColumn:=True)
        End Sub

        Private Sub InitIrisFlowerLogical()
            m_mlp.Initialize(learningRate:=0)
            m_mlp.inputArray = m_inputArrayIrisFlower
            m_mlp.targetArray = m_targetArrayIrisFlowerLogical
            m_mlp.InitializeStruct(m_neuronCountIrisFlowerLogical, addBiasColumn:=True)
        End Sub

        '<TestMethod()>
        'Public Sub EncogMLP1XORSemiStochastic()

        '    InitXOR()
        '    m_mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)

        '    m_mlp.nbIterations = 300
        '    m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

        '    m_mlp.InitializeWeights(1, {
        '        {-0.96, 0.9, 0.41},
        '        {-0.31, -0.78, -0.64},
        '        {0.26, 0.8, 0.15}})
        '    m_mlp.InitializeWeights(2, {
        '        {-0.04, -0.14, 0.9, -0.27}})

        '    ' There is no TrainOneSample function!
        '    m_mlp.Train(learningMode:=enumLearningMode.SemiStochastic)

        '    Dim expectedOutput = m_targetArrayXOR
        '    Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

        '    Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

        '    Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        '    Assert.AreEqual(sExpectedOutput, sOutput)

        '    Const expectedLoss# = 0.01
        '    Dim loss! = m_mlp.averageError
        '    Dim lossRounded# = Math.Round(loss, 2)
        '    Assert.AreEqual(True, lossRounded <= expectedLoss)

        'End Sub

        '<TestMethod()>
        'Public Sub EncogMLP1XORStochastic()

        '    InitXOR()
        '    m_mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)

        '    m_mlp.nbIterations = 300
        '    m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

        '    m_mlp.InitializeWeights(1, {
        '        {-0.96, 0.9, 0.41},
        '        {-0.31, -0.78, -0.64},
        '        {0.26, 0.8, 0.15}})
        '    m_mlp.InitializeWeights(2, {
        '        {-0.04, -0.14, 0.9, -0.27}})

        '    ' There is no TrainOneSample function!
        '    m_mlp.Train(learningMode:=enumLearningMode.Stochastic)

        '    Dim expectedOutput = m_targetArrayXOR
        '    Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

        '    Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

        '    Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        '    Assert.AreEqual(sExpectedOutput, sOutput)

        '    Const expectedLoss# = 0.01
        '    Dim loss! = m_mlp.averageError
        '    Dim lossRounded# = Math.Round(loss, 2)
        '    Assert.AreEqual(True, lossRounded <= expectedLoss)

        'End Sub

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
            Dim loss! = m_mlp.averageError
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
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        '<TestMethod()>
        'Public Sub EncogMLP1XOR4Layers()

        '    TestMLP1XOR4Layers(m_mlp, weightAdjustment:=0,
        '        learningMode:=enumLearningMode.Systematic)

        'End Sub

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
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        '<TestMethod()>
        'Public Sub EncogMLP1XORSigmoid()

        '    TestMLP1XORSigmoid231(m_mlp)

        'End Sub

        <TestMethod()>
        Public Sub EncogMLP1XORSigmoid()

            TestMLP1XORSigmoid231(m_mlp, learningMode:=enumLearningMode.Vectorial)

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

            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        '<TestMethod()>
        'Public Sub EncogMLP1XORHTangent()

        '    TestMLP1XORHTangent(m_mlp, weightAdjustment:=0, gain:=2,
        '        learningMode:=enumLearningMode.Systematic)

        'End Sub

        <TestMethod()>
        Public Sub EncogMLP1XORHTangent()

            TestMLP1XORHTangent(m_mlp, weightAdjustment:=0, gain:=2,
                learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub EncogMLP1XORHTangent261()

            TestMLP1XORHTangent261(m_mlp, nbIterations:=200, gain:=2,
                learningMode:=enumLearningMode.Vectorial)

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
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        '<TestMethod()>
        'Public Sub EncogMLP2XORHTangent()

        '    TestMLP2XORHTangent(m_mlp, weightAdjustment:=0, gain:=2,
        '        learningMode:=enumLearningMode.Systematic)

        'End Sub

        <TestMethod()>
        Public Sub EncogMLP2XORHTangent()

            TestMLP2XORHTangent(m_mlp, weightAdjustment:=0, gain:=2,
                learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub EncogMLP2XORHTangent462()

            TestMLP2XORHTangent462(m_mlp, gain:=2,
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
            Dim loss! = m_mlp.averageError
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
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        '<TestMethod()>
        'Public Sub EncogMLP3XORHTangent()

        '    TestMLP3XORHTangent(m_mlp, gain:=2) ' works slowly

        'End Sub

        <TestMethod()>
        Public Sub EncogMLP3XORHTangent()

            TestMLP3XORHTangent(m_mlp, gain:=2,
                learningMode:=enumLearningMode.Vectorial)

        End Sub

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
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub EncogMLPIrisFlowerAnalog()

            InitIrisFlowerAnalog()

            m_mlp.nbIterations = 300
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {0.9, 0.67, -0.21, -0.11, -0.29},
                {-0.52, -0.27, -0.05, -0.19, 0.8},
                {-0.3, 0.58, 0.5, 0.92, -0.49},
                {-0.09, 0.22, -0.08, -0.84, 0.77},
                {-0.65, -0.93, -0.47, 0.41, -0.66},
                {0.54, -0.52, -0.54, -0.07, 0.82},
                {0.62, -0.17, 0.05, -0.64, 0.08},
                {0.86, -0.62, -0.67, 0.41, 0.21},
                {-0.92, 0.01, -0.23, 0.65, 0.68},
                {0.53, -0.38, -0.4, -0.68, 0.25},
                {-0.95, -0.73, -0.74, 0.9, -0.55},
                {0.31, -0.68, -0.42, 0.73, 0.47},
                {-0.27, -0.96, -0.18, 0.27, -0.77},
                {0.84, -0.23, 0.47, 0.3, -0.76},
                {-0.26, -0.17, 0.88, 0.83, 0.57},
                {0.61, -0.75, -0.6, 0.31, 0.53}})
            m_mlp.InitializeWeights(2, {
                {-0.25, -0.22, -0.25, -0.81, -0.1, -0.94, 0.27, 0.85, 0.98, -0.38, 0.47, 0.54, -0.01, -0.51, -0.08, -0.94, 0.5},
                {0.87, -0.05, -0.63, -0.02, 0.86, -0.87, 0.48, -0.38, 0.78, 0.61, -0.39, -0.13, -0.16, 0.56, 0.82, 0.32, 0.54},
                {0.43, 0.88, -0.02, -0.3, 0.46, 0.97, 0.34, -0.92, 0.04, 0.64, -0.44, 0.39, -0.8, -0.39, -0.73, 0.63, -0.14},
                {0.22, 0.02, 0.4, 0.27, -0.81, 0.5, -0.3, -0.94, 0.44, -0.5, -0.3, -0.54, -0.48, 0.18, -0.07, -0.62, -0.96},
                {-0.15, -0.04, 0.86, -0.63, -0.65, -0.81, 0.4, -0.3, -0.52, -0.75, 0.22, -0.54, 0.95, -0.39, 0.64, -0.28, -0.94},
                {-0.34, -0.9, -0.39, -0.45, 0.66, 0.91, -0.54, 0.89, 0.81, 0.8, -0.71, -0.45, 0.02, -1.0, 0.34, -0.63, -0.33},
                {-0.16, -0.94, -0.54, -0.43, 0.6, 0.04, -0.99, 0.51, 0.47, 0.58, 0.51, 0.93, -0.69, -0.19, -0.63, 0.51, -0.5},
                {0.86, 0.07, -0.96, -0.36, -0.83, -0.79, -0.47, -0.01, 0.61, 0.38, -0.96, -0.83, -0.67, -0.08, -0.48, -0.01, 0.15},
                {-0.41, 0.73, -0.04, 0.8, -0.51, 0.79, 0.38, -0.18, -0.88, 0.98, 0.15, -0.79, -0.9, -0.32, -0.43, -0.97, 0.19},
                {-0.91, -0.29, -0.35, -0.63, 0.83, -0.66, -0.47, 0.26, 0.45, -0.65, -0.53, 0.66, 0.61, -0.61, 0.06, 0.27, -0.91},
                {0.27, -0.52, 0.18, 0.49, -0.94, 0.6, -0.22, -0.6, -0.88, 0.88, 0.94, 0.46, 0.36, -0.5, -0.65, 0.84, -0.39},
                {-0.78, -0.84, -0.75, -0.86, -0.69, 0.5, -0.47, -0.99, 0.67, -0.86, 0.26, -0.87, 0.78, 0.56, -0.14, 0.56, 0.55},
                {-0.44, -0.52, -0.44, -0.82, -0.08, 0.12, -0.52, -0.93, 0.13, -0.95, 0.77, 0.94, 0.86, -0.17, 0.19, -0.48, 0.65},
                {0.33, -0.11, -0.14, -0.32, 0.82, -0.68, -0.74, -0.89, -0.49, -1.0, 0.84, 0.33, -0.68, 0.38, -0.85, -0.25, 0.96},
                {0.06, 0.19, -0.35, 0.38, -0.49, 0.86, 0.72, 0.05, 0.72, -0.54, 0.86, 0.55, 0.61, -0.9, -0.69, 0.5, 0.73},
                {0.47, 0.41, -0.65, 0.7, -0.13, -0.72, 0.98, -0.83, 0.53, 0.27, -0.68, -0.88, -0.86, 0.82, 0.89, -0.28, -0.47}})
            m_mlp.InitializeWeights(3, {
                {-0.92, -0.39, 0.5, 0.51, 0.75, 0.72, 0.1, -0.45, -0.69, 0.28, -0.1, -0.63, 0.5, -0.19, 1.0, 0.82, 0.56}})

            m_mlp.minimalSuccessTreshold = 0.2
            m_mlp.Train()

            Const expectedLoss# = 0.03
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            Dim expectedSuccess# = 0.98
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 2)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            If m_mlp.successPC = 1 AndAlso m_mlp.minimalSuccessTreshold <= 0.05! Then
                Dim expectedOutput = m_targetArrayIrisFlowerAnalogUnnormalized
                Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
                Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
                Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
                Assert.AreEqual(sExpectedOutput, sOutput)
            End If

        End Sub

        <TestMethod()>
        Public Sub EncogMLPIrisFlowerAnalog4Layers()

            InitIrisFlowerAnalog()
            m_mlp.InitializeStruct({4, 50, 20, 1}, addBiasColumn:=True)

            m_mlp.nbIterations = 200
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {0.97, -0.54, 0.85, -0.08, 0.03},
                {0.58, -0.4, 0.15, 0.43, -0.63},
                {0.32, -0.38, 0.31, 0.97, 1.0},
                {0.99, 0.73, -0.7, 0.33, 0.82},
                {-0.1, -0.8, 0.94, 0.04, 0.96},
                {-0.76, 0.44, -0.2, 0.77, 0.59},
                {0.28, 0.8, -0.01, 0.28, 0.21},
                {0.42, -0.26, 0.76, 0.43, -0.84},
                {-0.74, 0.16, -0.72, -0.26, -0.69},
                {0.13, -0.86, -0.45, -0.73, -0.83},
                {-0.1, -0.38, 0.68, -0.87, -0.4},
                {-0.81, -0.46, 0.9, -0.68, -0.49},
                {0.36, 0.11, -0.41, 0.19, -0.12},
                {0.37, -0.13, 0.44, -0.47, -0.6},
                {0.62, 0.69, -0.96, -0.96, 0.31},
                {-0.02, -0.44, 0.21, 0.82, 0.03},
                {0.82, 0.26, -0.22, -0.81, -0.44},
                {-0.3, 0.52, 0.7, -0.67, 0.7},
                {0.78, -0.09, 0.94, 0.56, 0.71},
                {0.7, -0.31, -0.14, 0.09, -0.04},
                {0.89, -0.05, -0.16, 0.12, 0.93},
                {-0.47, 0.66, -0.92, 0.33, 0.81},
                {0.78, 0.61, 0.87, -0.78, 0.11},
                {-0.42, -0.13, 0.89, 0.81, 0.46},
                {-0.17, 0.45, 0.56, 0.02, 0.42},
                {-0.24, 0.65, 0.66, -0.13, -0.54},
                {0.62, 0.76, -0.34, 0.17, -0.39},
                {-0.25, 0.9, -0.33, -0.23, 0.02},
                {-0.69, -0.86, -0.43, -0.46, 0.81},
                {0.39, 0.72, 0.75, -0.66, 0.49},
                {0.43, -0.9, -0.49, -0.03, -0.91},
                {-0.66, 0.4, 0.86, 0.1, -0.21},
                {-0.08, 0.75, 0.71, 0.76, 0.71},
                {0.85, 0.41, -0.74, 0.48, 0.02},
                {-0.54, -0.07, 0.87, -0.17, 0.31},
                {-0.07, -0.17, -0.05, 0.08, -0.4},
                {-0.98, 0.72, -0.91, 0.89, 0.1},
                {-0.84, 0.19, 0.45, -0.61, 0.94},
                {0.09, 0.65, 0.48, 0.22, 0.52},
                {-0.09, 0.02, -0.87, -0.89, 0.23},
                {-0.76, 0.13, 0.85, 0.67, 0.87},
                {-0.75, 0.29, 0.97, -0.33, 0.74},
                {-0.23, 0.0, -0.09, -0.28, -0.92},
                {0.88, -0.05, 0.1, 0.2, 0.05},
                {-0.56, -0.7, 0.34, 1.0, 0.5},
                {0.11, 0.71, 0.22, 0.53, 0.15},
                {-0.33, 0.13, -0.3, -0.22, 0.78},
                {-0.78, 0.9, 0.56, -0.62, -0.68},
                {-0.01, -0.81, -0.09, 0.06, 0.89},
                {-0.11, 0.88, 0.08, 0.52, -0.04}})
            m_mlp.InitializeWeights(2, {
                {-0.04, 0.93, 0.34, 0.49, -0.2, -0.37, 0.62, 0.26, -0.43, 0.83, 0.63, 0.15, -0.87, -0.41, 0.86, 0.07, 0.73, 0.84, -0.08, 0.17, -0.43, -0.78, -0.14, -0.84, -0.02, -0.79, -0.12, -0.11, 0.04, -0.11, -0.53, 0.29, 0.82, -0.03, 0.32, 0.91, 0.75, -0.36, -0.3, 0.47, 0.36, 0.23, 0.16, 0.3, -0.37, 0.22, 0.25, -0.96, -0.05, 0.58, -0.04},
                {0.08, -0.37, -0.44, 0.19, -0.52, 0.9, 0.11, -0.16, 0.09, 0.63, -0.4, 0.85, -0.3, 0.57, 0.66, 0.15, -0.47, -0.25, 0.2, 0.49, -0.11, 0.31, 0.77, -0.71, 0.79, 0.63, -0.65, -0.63, -0.03, 0.47, -0.74, 0.92, -0.64, -0.52, -0.15, -0.11, -0.94, 0.43, -0.03, -0.07, 0.92, -0.44, 0.64, -0.87, -0.76, -0.17, 0.18, 0.87, 0.4, 0.94, 0.31},
                {-0.33, -0.82, 0.54, -0.41, 0.1, -0.04, 0.69, -0.87, 0.81, -0.02, -0.87, -0.36, 0.63, -0.53, -0.61, 0.54, -0.82, 0.57, -0.49, -0.49, -0.01, 0.07, 0.99, -0.8, -0.45, 0.89, -0.9, 0.35, -0.97, 0.81, -0.28, 0.48, 0.29, -0.35, -0.65, 0.32, -0.97, 0.57, 0.0, 0.32, 0.12, -0.42, 0.43, 0.43, -0.89, -0.41, -0.62, 0.87, 0.43, -0.45, 0.73},
                {-0.9, -0.53, -0.99, -0.88, -0.74, 0.89, -0.89, -0.87, -0.4, -0.16, 0.72, 0.59, 0.53, 0.6, -0.96, -0.52, 0.21, 0.92, -0.01, 0.64, 0.49, -0.55, -0.57, 0.32, -0.84, -0.14, -0.33, 0.72, 0.5, -0.93, 0.4, 0.23, 0.12, 0.63, -0.44, 0.1, -0.48, 0.41, 0.95, -0.41, 0.0, 0.19, -0.6, 0.12, 0.89, 0.13, -0.94, -0.41, 0.96, -0.39, 0.83},
                {0.59, 0.08, -0.74, -0.16, 0.91, -1.0, -0.04, -0.07, -0.71, 0.15, -0.93, 0.01, -0.28, 0.4, -0.41, 0.8, 0.88, -0.38, 0.51, -0.52, 0.65, -0.08, 0.79, -0.83, 0.3, -0.9, 0.42, 0.67, 0.71, -0.18, 0.37, -0.52, -0.05, 0.06, 0.22, 0.63, 0.73, 0.2, -0.09, 0.98, -0.64, -0.73, -0.3, 0.27, -0.03, -0.18, -0.14, -0.54, -0.93, -0.91, 0.9},
                {0.22, -0.81, 0.68, -0.32, -0.25, 0.17, 0.77, 0.38, 0.6, 0.73, 0.19, -0.16, -0.97, 0.09, 0.7, 0.22, -0.53, -0.56, 0.37, -0.56, -0.89, 0.02, 0.05, 0.19, 0.38, -0.74, -0.03, 0.28, 0.3, 0.36, -0.23, 0.79, -0.32, 0.72, 0.77, 0.99, 0.53, 0.46, -0.06, 0.62, -0.69, 0.87, 0.33, -0.53, 0.88, -0.31, -0.95, 0.07, -0.15, -0.29, 0.98},
                {0.26, 0.73, 0.39, 0.06, 0.06, 0.49, -0.4, 0.94, 0.75, -0.55, 0.6, -0.4, -0.02, -0.04, 0.36, 0.86, -0.7, 0.31, 0.86, 0.16, 0.59, 0.95, -0.96, -0.15, -0.69, -0.51, 0.23, -0.92, -0.19, -1.0, 0.54, 0.8, -0.53, -0.23, 0.97, 0.13, 0.29, -0.93, -0.9, 0.72, -0.55, 0.02, 0.25, -0.89, 0.22, 0.84, 0.74, -0.53, -0.89, -0.34, -0.81},
                {-0.46, -0.93, 0.26, 0.5, 0.95, -0.77, -0.59, -0.03, 0.34, 0.9, 0.75, -0.97, -0.07, 0.84, -0.58, -0.39, -0.34, -0.19, -0.58, -0.45, 0.39, 0.29, -0.83, -0.59, -0.5, 0.94, 0.52, 0.75, 0.82, -0.11, 0.65, 0.05, -0.05, 0.58, -0.29, -0.45, -0.84, -0.65, -0.85, 0.54, -0.92, -0.68, -0.23, -0.58, -0.07, 0.36, -0.57, 0.8, 0.15, 0.3, -0.72},
                {-0.61, 0.95, -0.04, -0.71, 0.81, 0.82, -0.58, 0.42, -0.15, -0.3, -0.74, 0.28, 0.64, 0.39, 0.65, -0.14, -0.9, -0.84, 0.82, 0.35, -0.44, 0.18, -0.26, -0.72, -0.69, 0.14, 0.91, 0.71, 0.04, -0.66, 0.6, 0.0, -0.3, 0.63, 0.55, 0.14, 0.55, -0.16, 0.24, 0.57, -0.89, -0.35, -0.79, -0.82, -0.97, 0.06, 0.93, 0.89, -0.93, 0.31, -0.6},
                {0.51, -0.73, 0.69, 0.07, -0.81, 0.68, 0.22, 0.57, 0.7, 0.26, 0.16, -0.34, 0.24, 0.16, -0.82, 0.73, 0.86, -0.55, -0.33, 0.37, -0.09, 0.23, 0.03, 0.88, 0.61, -0.41, 0.64, 0.85, -0.7, 0.81, 0.22, 0.97, -0.56, 0.6, 0.11, -0.86, 0.19, -0.04, 0.84, -0.95, 0.16, -0.01, -0.24, 0.08, -0.88, 0.44, 0.1, 0.31, -0.94, 0.68, 0.58},
                {-0.02, -0.33, -0.64, 0.69, -0.89, 0.26, 0.38, -0.94, 0.5, -0.31, -0.46, 0.76, -0.93, 0.44, 0.28, 0.56, -0.77, 0.38, -0.2, 0.75, 0.58, -0.25, 0.94, -0.07, -0.54, 0.94, -0.12, -0.18, -0.53, 0.82, -0.14, -0.84, -0.25, -0.19, 0.87, 0.92, 0.23, 0.2, -0.4, 0.57, 0.35, 0.89, -0.98, 0.95, -0.78, 0.67, 0.37, 0.01, 0.41, 0.35, -0.27},
                {-0.81, -0.71, 0.92, 0.19, 0.92, -0.27, 0.66, 0.66, -0.12, -0.34, -0.34, 0.45, -0.94, -0.15, -0.03, -0.16, -0.82, 0.79, 0.27, 0.79, -0.3, -0.61, 0.37, 0.68, 0.96, 0.18, -0.35, 0.52, -0.95, -0.32, 0.14, 0.14, -0.61, -0.24, 0.27, -0.59, -0.03, 0.61, -0.11, -0.27, -0.76, -0.12, -0.99, 0.33, 0.9, 0.15, -0.18, -0.33, 0.04, 0.12, -0.34},
                {-0.66, 0.13, 0.23, -0.57, -0.41, 0.98, -0.01, 0.21, -0.94, -0.72, 0.68, 0.71, -0.91, -0.8, -0.46, 0.66, -0.02, 0.03, -0.8, 0.32, 0.75, 0.85, -0.93, -0.12, -0.55, 0.88, -0.63, -0.36, 0.99, -0.75, -0.79, -0.63, 0.17, 0.26, 0.48, 0.2, -0.7, 0.25, 0.16, 0.87, 0.48, -0.87, 0.65, -0.82, 0.59, 0.43, 0.05, -0.84, -0.53, -0.3, -0.07},
                {-0.18, -0.15, 0.18, 0.01, -0.97, 0.62, -0.36, 0.09, 0.46, -0.97, 0.35, 0.84, 0.28, 0.48, 0.54, 0.73, 0.24, -0.57, 0.64, -0.09, 0.15, 0.34, 0.35, -0.87, 0.36, -0.3, -0.94, 0.49, -0.01, -0.07, 0.3, -0.37, 0.66, -0.37, 0.11, 0.13, 0.05, -0.24, 0.18, -0.16, 0.1, 0.06, 0.2, -0.7, 0.75, 0.79, -0.73, 0.75, 0.88, -0.69, -0.51},
                {-0.92, -0.47, 0.04, 0.37, -0.41, -0.4, -0.23, 0.95, -0.17, -0.82, -0.6, 0.96, -0.81, -0.57, 0.38, 0.38, 0.91, 0.9, 0.87, -0.55, -0.84, -0.13, 0.03, -0.79, -0.78, 0.65, 0.66, 0.72, 0.89, 1.0, -0.77, 0.67, 0.08, -0.47, 0.08, -0.73, 0.89, 0.48, -0.34, -0.47, -0.37, -0.08, 0.03, -0.87, -0.63, 0.86, 0.72, -0.03, 0.34, 0.33, 0.18},
                {0.81, 0.01, -0.66, 0.36, -0.85, 0.02, -0.1, 0.52, -0.21, -0.83, -0.06, 0.23, 0.78, -0.55, -0.44, -0.08, -0.29, -0.46, -0.02, -0.37, 0.43, -0.72, 0.26, -0.55, -0.44, -0.63, 0.35, 0.86, -0.5, 0.92, 0.56, -0.36, -0.45, -0.07, 0.34, -0.88, -0.57, 0.84, -0.68, 0.67, 0.77, 0.55, -0.21, 0.59, 0.03, 0.06, 0.8, -0.96, 0.39, -0.36, 0.01},
                {-0.2, -0.88, -0.1, 0.08, 0.98, -0.91, 0.52, -0.79, -0.02, 0.03, -0.9, 0.5, 0.95, -0.78, -0.78, 0.18, -0.88, -0.04, -0.04, -0.77, -0.3, -0.5, 0.82, 0.32, 0.57, 0.74, 0.9, 0.53, 0.0, -0.65, 0.17, -0.24, -0.23, -0.37, 0.7, -0.87, 0.53, -0.14, -0.51, -0.57, 0.65, 0.82, 0.33, 0.13, -0.79, -0.98, 0.65, -0.55, 0.31, -0.56, -0.14},
                {0.86, 0.61, -0.8, -0.83, 0.42, -0.68, 0.21, -0.15, 0.55, 0.53, -0.03, 0.26, -0.47, 0.11, -0.97, -0.56, 0.91, -0.67, -0.74, -0.42, -0.13, 0.02, -0.46, -0.36, -0.33, -0.95, 0.69, 0.22, -0.62, 0.85, 0.96, -0.6, 0.49, -0.64, 0.66, -0.91, 0.25, -0.2, -0.38, 0.62, -0.59, -0.68, 0.92, -0.82, -0.69, 0.54, -0.02, -0.47, 0.79, 0.91, -0.18},
                {0.09, 0.43, 0.02, -0.78, 0.18, -0.79, -0.74, -0.07, 0.18, 0.84, 0.44, -0.23, -0.29, -0.13, -0.87, 0.01, -0.76, -0.41, -0.41, -0.59, -0.49, 0.99, -0.43, -0.34, -0.73, -0.97, -0.94, 0.44, 0.0, -0.99, -0.32, -0.31, -0.08, -0.72, -0.94, -0.59, -0.07, 0.37, 0.32, -0.51, 0.97, -0.77, -0.27, -0.5, 0.13, -0.85, 0.07, -0.72, 0.6, -0.48, 0.87},
                {-0.29, -0.92, 0.4, 0.4, 0.57, -0.91, -0.25, -0.77, 0.58, 0.33, -0.44, 0.97, 0.23, -0.02, 0.63, 0.7, 0.37, 0.33, 0.44, 0.25, 0.06, -0.63, -0.37, 0.87, -0.4, -0.22, 0.03, 0.13, -0.86, -0.63, -0.65, -0.95, 0.59, 0.8, 0.23, 0.3, 0.23, -0.9, -0.73, -0.74, 0.3, 0.67, -0.51, -0.06, -0.78, 0.39, -0.72, -0.24, 0.79, 0.61, 0.36}})
            m_mlp.InitializeWeights(3, {
                {-0.81, -0.19, 0.85, -0.61, -0.29, -0.71, -0.66, -0.83, -0.45, -0.3, -0.05, 0.4, 0.27, 0.99, -0.17, -0.73, -0.78, -0.28, -0.29, -0.89, -0.49}})

            m_mlp.minimalSuccessTreshold = 0.2
            m_mlp.Train()

            Const expectedLoss# = 0.03
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            Dim expectedSuccess# = 0.967
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            If m_mlp.successPC = 1 AndAlso m_mlp.minimalSuccessTreshold <= 0.05! Then
                Dim expectedOutput = m_targetArrayIrisFlowerLogical
                Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
                Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
                Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
                Assert.AreEqual(sExpectedOutput, sOutput)
            End If

        End Sub

        <TestMethod()>
        Public Sub EncogMLPIrisFlowerLogical()

            InitIrisFlowerLogical()

            m_mlp.nbIterations = 300
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {0.3, 0.24, -0.25, -0.76, -0.17},
                {-0.94, 0.63, 0.48, 0.96, -0.49},
                {0.43, -0.77, -0.43, -0.04, -0.99},
                {0.73, 0.08, -0.15, -0.32, 0.49},
                {-0.2, -0.76, 0.88, 0.53, 0.79},
                {0.97, -0.34, 0.74, 0.74, -0.58},
                {0.67, 0.65, -0.66, -0.84, 0.77},
                {0.23, 0.0, -0.05, -0.36, 0.5},
                {-0.6, 0.19, -0.28, -0.59, 0.29},
                {0.72, 0.43, 0.97, -0.58, 0.27},
                {-0.85, -0.99, 0.93, -0.46, -0.64},
                {-0.24, -0.16, 0.13, 0.37, -0.73},
                {-0.04, 0.35, -0.62, 0.96, -0.27},
                {-0.64, 0.17, -0.96, -0.18, -0.48},
                {0.27, -0.11, 0.91, 0.88, -0.38},
                {0.64, 0.64, -0.15, 0.23, 0.15}})
            m_mlp.InitializeWeights(2, {
                {0.52, -0.92, 0.8, -0.72, -0.97, 0.15, 0.37, -0.05, 0.89, 0.89, -0.68, -0.15, 0.29, 0.02, 0.21, -0.13, 0.32},
                {0.16, -0.3, 0.39, -0.54, -0.44, 0.06, -0.26, 0.32, -0.09, 0.8, -0.04, 0.36, 0.6, -0.23, -0.11, -0.89, -0.11},
                {-0.4, -0.7, 0.85, -0.81, 0.65, 0.97, 0.83, 0.6, 0.66, -0.51, -0.35, -0.52, -0.22, 0.21, -0.3, -0.32, -0.47},
                {0.39, 0.87, -0.2, -0.96, 0.81, -0.08, 0.42, 0.76, 0.5, 0.98, -0.61, 0.69, -0.46, 0.58, -0.04, 0.12, 0.23},
                {-0.79, -0.97, -0.78, -0.78, 0.12, 0.7, -0.33, 0.88, 0.37, -0.33, -0.58, 0.75, -0.52, -0.65, 0.85, 0.93, 0.99},
                {-0.79, -0.58, -0.5, -0.24, -0.81, 0.87, -0.51, -0.3, -0.58, -0.97, 0.36, -0.69, 0.37, -0.1, -0.88, -0.86, 0.15},
                {-0.89, 0.06, 0.03, 0.12, 0.68, 0.67, 0.21, 0.73, 0.21, -0.32, 0.53, -0.28, 0.76, -0.2, 0.28, -0.34, -0.48},
                {0.01, 0.82, -0.38, 0.1, 0.59, 0.05, -0.91, -0.84, -0.33, -0.33, -0.47, 0.86, -0.47, -0.99, 0.48, -0.61, -0.1},
                {0.32, -0.81, -0.59, -0.9, 0.67, -0.73, -0.22, -0.11, 0.85, 0.01, -0.89, -0.77, -0.74, -0.76, 0.6, -0.2, 0.49},
                {0.93, 0.46, 0.0, -0.22, 0.68, -0.25, -0.4, -0.4, 0.89, 0.67, -0.07, -0.01, 0.96, -0.21, 0.95, -0.23, 0.73},
                {-0.56, 0.39, -0.83, 0.2, 0.46, -0.3, -0.64, 0.55, -0.86, -0.47, 0.91, -0.51, 0.87, -0.84, 0.43, -0.7, 0.99},
                {0.45, -0.74, 0.86, -0.25, 0.03, 0.44, -0.46, -0.63, 0.1, -0.94, 0.99, 0.69, 0.74, 0.43, 0.79, 0.69, -0.6},
                {0.2, 0.38, 0.98, -0.18, -0.76, 0.3, 0.29, 0.49, 0.5, -0.06, -0.44, 0.89, -0.88, -0.13, -0.56, 0.21, 0.47},
                {-0.03, 0.89, -0.68, 0.85, 0.2, -0.73, -0.92, -0.76, 0.13, -0.16, -0.36, 0.14, -0.31, 0.84, -0.02, -0.67, 0.01},
                {-0.01, 0.9, -0.7, 0.62, 0.84, 0.79, -0.14, -0.71, 0.75, -0.4, -0.34, 0.72, -0.89, -0.36, 0.73, 0.66, 0.57},
                {0.9, 0.89, -0.67, 0.43, 0.87, -0.45, -0.1, 0.4, -0.81, -0.45, 0.76, 0.02, -0.62, -0.03, 0.54, -0.58, 0.26}})
            m_mlp.InitializeWeights(3, {
                {0.06, -0.24, 0.04, -0.69, 0.82, 0.26, -0.53, -0.5, 0.59, -0.8, -0.08, 0.04, 0.77, 0.79, 0.0, 0.44, 0.45},
                {-0.38, 0.17, 0.07, -0.83, 0.59, -0.7, 0.55, 0.56, 0.78, -0.84, 0.78, -0.37, -0.1, 0.85, 0.67, 0.07, -0.28},
                {-0.44, 0.53, -0.77, 0.08, -0.72, 0.01, -0.55, 0.72, 0.63, -0.74, 0.2, -0.41, 0.26, 0.47, -0.01, 0.9, -0.9}})

            m_mlp.minimalSuccessTreshold = 0.3
            m_mlp.Train()

            Const expectedLoss# = 0.009
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            Dim expectedSuccess# = 0.996
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            If m_mlp.successPC = 1 AndAlso m_mlp.minimalSuccessTreshold <= 0.05! Then
                Dim expectedOutput = m_targetArrayIrisFlowerLogical
                Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
                Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
                Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
                Assert.AreEqual(sExpectedOutput, sOutput)
            End If

        End Sub

        <TestMethod()>
        Public Sub EncogMLPIrisFlowerAnalogStdr()

            TestMLPIrisFlowerAnalog(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub EncogMLPIrisFlowerAnalogPredictionTanh()

            ' 96.7% prediction, 98.3% learning with 50 iterations in 9 msec.

            TestMLPIrisFlowerAnalogPrediction(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub EncogMLPIrisFlowerLogicalStdr()

            TestMLPIrisFlowerLogical(m_mlp, nbIterations:=1000, expectedSuccess:=0.884#)

        End Sub

        <TestMethod()>
        Public Sub EncogMLPIrisFlowerLogicalPredictionStdr()

            ' 97.8% prediction, 99.4% learning with 500 iterations in 3.7 sec.

            TestMLPIrisFlowerLogicalPredictionTanh(m_mlp, nbIterations:=500)

        End Sub

        <TestMethod()>
        Public Sub EncogMLPIrisFlowerLogicalPrediction3L()

            ' 95.6% prediction, 96.4% learning with 1500 iterations in 74 msec.

            InitIrisFlowerLogicalPrediction(m_mlp)
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
            Dim loss! = m_mlp.averageError
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
        Public Sub EncogMLPIrisFlowerLogicalPrediction4L()

            ' 97.8% prediction, 100% learning with 1500 iterations in 280 msec.

            InitIrisFlowerLogical()
            'm_mlp.inputArray = m_inputArrayIrisFlowerTrain
            m_mlp.inputArray = m_inputArrayIrisFlowerTrainOriginal
            m_mlp.targetArray = m_targetArrayIrisFlowerLogicalTrain
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
            Dim loss! = m_mlp.averageError
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
        Public Sub EncogMLPIrisFlowerLogicalPrediction4L4883()

            ' 97.8% prediction, 100% learning with 2500 iterations in 309 msec. (original)
            ' 96.7% prediction, 100% learning with 2500 iterations in 309 msec. (corrected)

            InitIrisFlowerLogical()
            'm_mlp.inputArray = m_inputArrayIrisFlowerTrain
            m_mlp.inputArray = m_inputArrayIrisFlowerTrainOriginal
            m_mlp.targetArray = m_targetArrayIrisFlowerLogicalTrain
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
            Dim loss! = m_mlp.averageError
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
        Public Sub EncogMLPIrisFlowerLogicalPrediction4L4773()

            ' 97.8% prediction, 100% learning with 3500 iterations in 383 msec.

            InitIrisFlowerLogical()
            'm_mlp.inputArray = m_inputArrayIrisFlowerTrain
            m_mlp.inputArray = m_inputArrayIrisFlowerTrainOriginal
            m_mlp.targetArray = m_targetArrayIrisFlowerLogicalTrain
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
            Dim loss! = m_mlp.averageError
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
        Public Sub EncogMLPIrisFlowerLogicalPrediction4L4663()

            ' 97.8% prediction, 100% learning with 3500 iterations in 350 msec.

            InitIrisFlowerLogical()
            'm_mlp.inputArray = m_inputArrayIrisFlowerTrain
            m_mlp.inputArray = m_inputArrayIrisFlowerTrainOriginal
            m_mlp.targetArray = m_targetArrayIrisFlowerLogicalTrain
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
            Dim loss! = m_mlp.averageError
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
        Public Sub EncogMLPIrisFlowerLogicalPrediction4LSigmoid()

            ' 94.4% prediction, 99.4% learning with 400 iterations in 74 msec.

            InitIrisFlowerLogicalPrediction(m_mlp)

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
            Dim loss! = m_mlp.averageError
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
        Public Sub EncogMLPIrisFlowerLogicalPrediction4LSigmoid3()

            ' 97.8% prediction, 93.3% learning with 300 iterations in 57 msec.

            InitIrisFlowerLogicalPrediction(m_mlp)

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
            Dim loss! = m_mlp.averageError
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
        Public Sub EncogMLPIrisFlowerLogicalPrediction4LSigmoid2()

            ' 97.8% prediction, 97.2% learning with 200 iterations in 103 msec.
            TestMLPIrisFlowerLogicalPredictionSigmoid(m_mlp,
                nbIterations:=500, expectedSuccess:=0.972)

        End Sub

        <TestMethod()>
        Public Sub EncogMLPIrisFlowerAnalog2()

            ' 96.7% prediction, 95.8% learning with 250 iterations in x msec.

            InitIrisFlowerAnalogPrediction(m_mlp)
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
            Dim loss! = m_mlp.averageError
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

    End Class

End Namespace