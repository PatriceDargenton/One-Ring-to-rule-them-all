
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Imports Microsoft.VisualStudio.TestTools.UnitTesting

Module modMLPEncogTest

    Sub Main()
        Console.WriteLine("Encog MLP with the classical XOR test.")
        EncogMLPTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()
    End Sub

    Public Sub EncogMLPTest()

        Dim mlp As New clsMLPEncog

        mlp.ShowMessage("Encog MLP test")
        mlp.ShowMessage("--------------")

        mlp.inputArray = m_inputArrayXOR
        mlp.targetArray = m_targetArrayXOR

        mlp.nbIterations = 500 ' Sigmoid: works
        mlp.nbIterations = 500 ' Hyperbolic tangent: works fine
        'mlp.nbIterations = 20000 ' Stochastic

        mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0)

        'mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
        mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
        'mlp.InitializeStruct(m_neuronCountXOR4Layers2331, addBiasColumn:=True)
        'mlp.InitializeStruct(m_neuronCountXOR5Layers23331, addBiasColumn:=True)

        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=1, center:=0)
        'mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=1, center:=0)

        mlp.Randomize()

        mlp.PrintWeights()

        WaitForKeyToStart()

        mlp.printOutput_ = True
        mlp.TrainVector() ' Works fine

        mlp.ShowMessage("Encog MLP test: Done.")

    End Sub

End Module

Namespace EncogMLP

    <TestClass()>
    Public Class clsMLPEncogTest

        Private m_mlp As New clsMLPEncog

        'Private m_mlp As New clsMLPClassic ' 1/12 success
        'Private m_mlp As New NetworkOOP.MultilayerPerceptron ' 2/12 success
        'Private m_mlp As New clsMLPAccord ' 1/12 success

        <TestInitialize()>
        Public Sub Init()
        End Sub

        Private Sub InitXOR()
            m_mlp.inputArray = m_inputArrayXOR
            m_mlp.targetArray = m_targetArrayXOR
            m_mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
        End Sub

        Private Sub Init2XOR()
            m_mlp.inputArray = m_inputArray2XOR
            m_mlp.targetArray = m_targetArray2XOR
            m_mlp.InitializeStruct(m_neuronCount2XOR452, addBiasColumn:=True)
        End Sub

        Private Sub Init3XOR()
            m_mlp.inputArray = m_inputArray3XOR
            m_mlp.targetArray = m_targetArray3XOR
            m_mlp.InitializeStruct(m_neuronCount3XOR673, addBiasColumn:=True)
        End Sub

        '<TestMethod()>
        'Public Sub EncogMLP1XORSemiStochastic()

        '    m_mlp.Initialize(learningRate:=0.01!, weightAdjustment:=0)
        '    InitXOR()
        '    m_mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)

        '    m_mlp.nbIterations = 300
        '    m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=1, center:=0)

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

        '    Dim expectedLoss# = 0.01
        '    Dim loss! = m_mlp.ComputeAverageError()
        '    Dim lossRounded# = Math.Round(loss, 2)
        '    Assert.AreEqual(True, lossRounded <= expectedLoss)

        'End Sub

        '<TestMethod()>
        'Public Sub EncogMLP1XORStochastic()

        '    m_mlp.Initialize(learningRate:=0.01!, weightAdjustment:=0)
        '    InitXOR()
        '    m_mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)

        '    m_mlp.nbIterations = 300
        '    m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=1, center:=0)

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

        '    Dim expectedLoss# = 0.01
        '    Dim loss! = m_mlp.ComputeAverageError()
        '    Dim lossRounded# = Math.Round(loss, 2)
        '    Assert.AreEqual(True, lossRounded <= expectedLoss)

        'End Sub

        <TestMethod()>
        Public Sub EncogMLP1XORSigmoidWithoutBias()

            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0)
            InitXOR()
            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=False)

            m_mlp.nbIterations = 2000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=1, center:=0)

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

            Dim expectedLoss# = 0.03
            Dim loss! = m_mlp.ComputeAverageError()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub EncogMLP1XORSigmoidWithoutBias231()

            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0)
            InitXOR()
            m_mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=False)

            m_mlp.nbIterations = 200
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=1, center:=0)

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

            Dim expectedLoss# = 0.01
            Dim loss! = m_mlp.ComputeAverageError()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        '<TestMethod()>
        'Public Sub EncogMLP1XOR4Layers()

        '    TestMLP1XOR4Layers(m_mlp, weightAdjustment:=0, gain:=1,
        '        learningMode:=enumLearningMode.Systematic)

        'End Sub

        <TestMethod()>
        Public Sub EncogMLP1XOR4Layers()

            TestMLP1XOR4Layers(m_mlp, weightAdjustment:=0, gain:=1,
                learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub EncogMLP1XOR5Layers()

            InitXOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0)
            m_mlp.InitializeStruct(m_neuronCountXOR5Layers23331, addBiasColumn:=True)

            m_mlp.nbIterations = 300
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=1, center:=0)

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

            Dim expectedLoss# = 0
            Dim loss! = m_mlp.ComputeAverageError()
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
            mlp.SetActivationFunction(
            enumActivationFunction.Sigmoid, gain, center:=0)

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

            Dim loss! = mlp.ComputeAverageError()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        '<TestMethod()>
        'Public Sub EncogMLP1XORHTangent()

        '    TestMLP1XORHTangent(m_mlp, weightAdjustment:=0, gain:=1,
        '        learningMode:=enumLearningMode.Systematic)

        'End Sub

        <TestMethod()>
        Public Sub EncogMLP1XORHTangent()

            TestMLP1XORHTangent(m_mlp, weightAdjustment:=0, gain:=1,
                learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub EncogMLP2XORSigmoid()

            m_mlp.Initialize(learningRate:=0.05!, weightAdjustment:=0)
            Init2XOR()

            m_mlp.nbIterations = 200
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=1, center:=0)

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

            Dim expectedLoss# = 0.01
            Dim loss! = m_mlp.ComputeAverageError()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        '<TestMethod()>
        'Public Sub EncogMLP2XORHTangent()

        '    TestMLP2XORHTangent(m_mlp, weightAdjustment:=0, gain:=1,
        '        learningMode:=enumLearningMode.Systematic)

        'End Sub

        <TestMethod()>
        Public Sub EncogMLP2XORHTangent()

            TestMLP2XORHTangent(m_mlp, weightAdjustment:=0, gain:=1,
                learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub EncogMLP2XORSinus()

            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0)
            Init2XOR()

            m_mlp.nbIterations = 200
            m_mlp.SetActivationFunction(enumActivationFunction.Sinus, gain:=1, center:=0)

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

            Dim expectedLoss# = 0.01
            Dim loss! = m_mlp.ComputeAverageError()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub EncogMLP3XORSigmoid()

            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0)
            Init3XOR()

            m_mlp.nbIterations = 300
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=1, center:=0)

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

            Dim expectedLoss# = 0.01
            Dim loss! = m_mlp.ComputeAverageError()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        '<TestMethod()>
        'Public Sub EncogMLP3XORHTangent()

        '    TestMLP3XORHTangent(m_mlp) ' works slowly

        'End Sub

        <TestMethod()>
        Public Sub EncogMLP3XORHTangent()

            TestMLP3XORHTangent(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub EncogMLP3XORSinus()

            m_mlp.Initialize(learningRate:=0.01!, weightAdjustment:=0)
            Init3XOR()

            m_mlp.nbIterations = 3000
            m_mlp.SetActivationFunction(enumActivationFunction.Sinus, gain:=1, center:=0)

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

            Dim expectedLoss# = 0.01
            Dim loss! = m_mlp.ComputeAverageError()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

    End Class

End Namespace