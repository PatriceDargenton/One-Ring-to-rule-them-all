
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Imports Microsoft.VisualStudio.TestTools.UnitTesting

' Tests are very slow!?
#Const Implementation = 1 ' 0: Off, 1: On

Module modMLPKerasTest

    Sub Main()
        Console.WriteLine("Keras MLP with the classical XOR test.")
        KerasMLPTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()
    End Sub

    Public Sub KerasMLPTest(Optional nbXor% = 1)

        Dim mlp As New clsMLPKeras

        mlp.ShowMessage("Keras MLP test")
        mlp.ShowMessage("--------------")

        mlp.inputArray = m_inputArrayXOR
        mlp.targetArray = m_targetArrayXOR

        mlp.nbIterations = 1500 ' Sigmoid: works
        'mlp.nbIterations = 2500 ' Sigmoid: works
        'mlp.nbIterations = 2000 ' Hyperbolic tangent: works fine

        'mlp.Initialize(learningRate:=0.001!, weightAdjustment:=0)
        'mlp.Initialize(learningRate:=0.01!, weightAdjustment:=0)
        mlp.Initialize(learningRate:=0.02!, weightAdjustment:=0)

        mlp.printOutput_ = True
        mlp.printOutputMatrix = False

        If nbXor = 1 Then
            'mlp.nbIterations = 1000
            'mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR281, addBiasColumn:=False)
            'mlp.InitializeStruct(m_neuronCountXOR291, addBiasColumn:=False)
            'mlp.InitializeStruct(m_neuronCountXOR2_10_1, addBiasColumn:=False)
            mlp.InitializeStruct(m_neuronCountXOR2_16_1, addBiasColumn:=False)
            'mlp.InitializeStruct(m_neuronCountXOR4Layers2331, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR4Layers2661, addBiasColumn:=False)
            'mlp.InitializeStruct(m_neuronCountXOR5Layers23331, addBiasColumn:=True)
            mlp.printOutputMatrix = True
            mlp.inputArray = m_inputArrayXOR
            mlp.targetArray = m_targetArrayXOR
        ElseIf nbXor = 2 Then
            mlp.nbIterations = 1500
            mlp.inputArray = m_inputArray2XOR
            mlp.targetArray = m_targetArray2XOR
            'mlp.InitializeStruct(m_neuronCount2XOR482, addBiasColumn:=False)
            mlp.InitializeStruct(m_neuronCount2XOR4_10_2, addBiasColumn:=False)
        ElseIf nbXor = 3 Then
            mlp.nbIterations = 1500
            mlp.inputArray = m_inputArray3XOR
            mlp.targetArray = m_targetArray3XOR
            'mlp.InitializeStruct(m_neuronCount3XOR673, addBiasColumn:=False)
            'mlp.InitializeStruct(m_neuronCount3XOR683, addBiasColumn:=False)
            mlp.InitializeStruct(m_neuronCount3XOR6_10_3, addBiasColumn:=False)
        End If

        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=1, center:=0)
        'mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=-2, center:=0)

        mlp.Randomize()

        mlp.PrintWeights()

        WaitForKeyToStart()

        mlp.Train(learningMode:=enumLearningMode.VectorialBatch) ' Works fine

        mlp.ShowMessage("Keras MLP test: Done.")

        If nbXor > 1 Then Exit Sub

        WaitForKeyToContinue("Press a key to print MLP weights")
        mlp.PrintWeights()

    End Sub

End Module

#If Implementation Then

Namespace KerasMLP

    <TestClass()>
    Public Class clsMLPKerasTest

        Private m_mlp As New clsMLPKeras

        Private Sub InitXOR()
            m_mlp.inputArray = m_inputArrayXOR
            m_mlp.targetArray = m_targetArrayXOR
        End Sub

        Private Sub Init2XOR()
            m_mlp.inputArray = m_inputArray2XOR
            m_mlp.targetArray = m_targetArray2XOR
        End Sub

        Private Sub Init3XOR()
            m_mlp.inputArray = m_inputArray3XOR
            m_mlp.targetArray = m_targetArray3XOR
        End Sub

        <TestMethod()>
        Public Sub KerasMLP1XOR4Layers()

            InitXOR()
            m_mlp.learningRate = 0.02
            m_mlp.nbIterations = 700
            m_mlp.InitializeStruct(m_neuronCountXOR4Layers2661, addBiasColumn:=False)
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=1, center:=0)

            m_mlp.InitializeWeights(1, {
                {0.13, 0.23, 0.49, 0.15, 0.09, 0.49},
                {0.44, -0.02, 0.58, 0.72, 0.59, 0.63}})
            m_mlp.InitializeWeights(2, {
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}})
            m_mlp.InitializeWeights(3, {
                {0.03, 0.34, -0.61, 0.57, 0.45, 0.63},
                {-0.55, 0.44, -0.64, 0.52, -0.67, 0.38},
                {0.42, -0.04, -0.61, 0.16, -0.24, 0.69},
                {-0.64, -0.45, -0.39, 0.46, 0.59, 0.68},
                {0.34, 0.13, -0.22, -0.58, 0.41, -0.12},
                {0.13, -0.49, 0.00, 0.59, -0.54, 0.64}})
            m_mlp.InitializeWeights(4, {
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}})
            m_mlp.InitializeWeights(5, {
                {-0.51},
                {-0.55},
                {-0.65},
                {0.69},
                {-0.66},
                {0.75}})
            m_mlp.InitializeWeights(6, {{0.0}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArrayXOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Dim expectedLoss# = 0.04
            Dim loss! = m_mlp.ComputeAverageError()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub KerasMLP1XOR5Layers()

            InitXOR()
            m_mlp.learningRate = 0.02
            m_mlp.nbIterations = 900
            m_mlp.InitializeStruct(m_neuronCountXOR5Layers27771, addBiasColumn:=False)
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=1, center:=0)

            m_mlp.InitializeWeights(1, {
                {0.34, 0.43, -0.09, -0.01, 0.8, 0.21, -0.17},
                {-0.37, -0.58, -0.28, -0.54, 0.64, -0.39, -0.79}})
            m_mlp.InitializeWeights(2, {
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}})
            m_mlp.InitializeWeights(3, {
                {0.49, 0.03, 0.33, 0.4, 0.03, 0.22, 0.5},
                {-0.62, 0.15, -0.3, -0.3, -0.16, -0.31, -0.13},
                {-0.27, -0.45, -0.16, 0.00, 0.24, -0.09, -0.37},
                {0.32, -0.5, -0.4, 0.62, 0.63, 0.13, 0.13},
                {-0.29, 0.61, -0.23, 0.42, 0.63, 0.03, -0.22},
                {-0.38, 0.06, 0.3, 0.3, 0.54, 0.55, 0.49},
                {0.34, 0.44, 0.33, 0.1, 0.14, -0.4, -0.11}})
            m_mlp.InitializeWeights(4, {
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}})
            m_mlp.InitializeWeights(5, {
                {-0.64, -0.03, 0.49, -0.46, -0.15, 0.59, 0.46},
                {-0.34, -0.05, 0.56, 0.26, 0.09, -0.64, -0.48},
                {-0.35, -0.09, -0.2, -0.63, 0.64, 0.07, -0.37},
                {-0.28, -0.26, 0.3, 0.52, 0.58, 0.11, 0.31},
                {0.35, 0.1, 0.46, -0.6, -0.45, 0.11, -0.24},
                {0.3, 0.63, -0.55, -0.61, 0.08, -0.6, -0.2},
                {0.44, -0.38, 0.63, -0.34, -0.63, 0.08, -0.33}})
            m_mlp.InitializeWeights(6, {
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}})
            m_mlp.InitializeWeights(7, {
                {-0.47},
                {0.1},
                {-0.47},
                {0.26},
                {-0.28},
                {0.4},
                {0.4}})
            m_mlp.InitializeWeights(8, {{0.0}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArrayXOR

            Dim sOutput$ = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Dim expectedLoss# = 0.02
            Dim loss! = m_mlp.ComputeAverageError()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub KerasMLP1XORSigmoid()

            m_mlp.Initialize(learningRate:=0.02!, weightAdjustment:=0)
            InitXOR()

            m_mlp.InitializeStruct(m_neuronCountXOR271, addBiasColumn:=False)

            m_mlp.nbIterations = 4000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=1, center:=0)

            m_mlp.InitializeWeights(1, {
                {0.35, 0.23, -0.12, 0.19, -0.09, 0.58, -0.24},
                {-0.14, 0.52, 0.72, 0.8, -0.37, 0.14, -0.04}})
            m_mlp.InitializeWeights(2, {
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}})
            m_mlp.InitializeWeights(3, {
                {0.25},
                {-0.44},
                {-0.59},
                {-0.14},
                {-0.24},
                {0.67},
                {-0.22}})
            m_mlp.InitializeWeights(4, {{0.0}})

            m_mlp.Train(enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Dim expectedLoss# = 0.02
            Dim loss! = m_mlp.ComputeAverageError()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub KerasMLP1XORHTangent()

            m_mlp.Initialize(learningRate:=0.02!, weightAdjustment:=0)
            InitXOR()

            m_mlp.InitializeStruct(m_neuronCountXOR2_16_1, addBiasColumn:=False)

            m_mlp.nbIterations = 600
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=-2, center:=0)

            m_mlp.InitializeWeights(1, {
                {-0.21, 0.51, 0.41, 0.18, 0.49, -0.33, -0.17, -0.08, 0.28, -0.02, -0.5, 0.19, -0.33, 0.15, -0.3, -0.21},
                {0.46, -0.45, -0.53, -0.09, -0.47, 0.52, 0.49, 0.26, -0.14, -0.1, -0.19, 0.5, -0.39, -0.57, 0.19, 0.08}})
            m_mlp.InitializeWeights(2, {
                {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00}})
            m_mlp.InitializeWeights(3, {
                {0.24},
                {0.17},
                {-0.18},
                {0.37},
                {0.44},
                {0.5},
                {-0.14},
                {-0.18},
                {-0.01},
                {0.16},
                {0.2},
                {-0.59},
                {0.47},
                {-0.39},
                {0.01},
                {0.18}})
            m_mlp.InitializeWeights(4, {{0.0}})

            m_mlp.Train(enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Dim expectedLoss# = 0.02
            Dim loss! = m_mlp.ComputeAverageError()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub KerasMLP2XORSigmoid()

            m_mlp.Initialize(learningRate:=0.02!, weightAdjustment:=0)
            Init2XOR()

            m_mlp.InitializeStruct(m_neuronCount2XOR482, addBiasColumn:=False)

            m_mlp.nbIterations = 6000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=1, center:=0)

            m_mlp.InitializeWeights(1, {
                {-0.43, -0.32, 0.52, 0.24, 0.5, -0.27, 0.45, 0.56},
                {0.51, 0.03, -0.27, -0.23, -0.4, -0.56, 0.31, -0.32},
                {0.65, 0.22, 0.6, 0.48, 0.44, 0.12, -0.31, 0.04},
                {-0.37, 0.25, 0.41, -0.1, -0.29, 0.59, 0.08, 0.1}})
            m_mlp.InitializeWeights(2, {
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}})
            m_mlp.InitializeWeights(3, {
                {-0.59, -0.27},
                {-0.56, -0.33},
                {0.00, -0.43},
                {0.48, 0.7},
                {-0.2, -0.42},
                {0.01, -0.61},
                {0.54, 0.32},
                {0.01, 0.52}})
            m_mlp.InitializeWeights(4, {{0.0, 0.0}})

            m_mlp.Train(enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArray2XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Dim expectedLoss# = 0
            Dim loss! = m_mlp.ComputeAverageError()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub KerasMLP2XORHTangent()

            m_mlp.Initialize(learningRate:=0.02!, weightAdjustment:=0)
            Init2XOR()

            m_mlp.InitializeStruct(m_neuronCount2XOR4_32_2, addBiasColumn:=False)

            m_mlp.nbIterations = 900
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=1, center:=0)

            m_mlp.InitializeWeights(1, {
                {0.26, -0.15, 0.18, 0.15, 0.15, -0.23, 0.1, 0.22, -0.19, -0.16, -0.24, 0.33, 0.09, 0.06, -0.29, -0.02, 0.12, -0.14, 0.33, -0.4, 0.00, 0.23, 0.04, 0.11, -0.11, 0.19, 0.22, -0.35, 0.2, -0.38, -0.3, -0.14},
                {0.15, -0.29, 0.27, 0.08, -0.22, -0.3, 0.26, -0.27, -0.15, 0.13, 0.05, 0.29, -0.4, -0.04, 0.04, -0.4, 0.35, 0.13, -0.08, 0.09, -0.29, -0.05, 0.38, -0.29, -0.01, -0.29, 0.25, -0.01, 0.38, -0.2, -0.32, 0.34},
                {0.14, 0.13, 0.39, -0.15, 0.11, -0.06, -0.24, -0.26, 0.16, -0.18, 0.25, 0.15, -0.26, -0.19, -0.13, 0.37, 0.04, -0.02, -0.2, -0.31, -0.21, 0.16, -0.21, 0.3, 0.22, 0.3, 0.35, -0.1, 0.32, -0.4, 0.13, 0.06},
                {-0.07, 0.34, -0.29, 0.13, -0.18, 0.38, 0.35, 0.38, -0.19, -0.18, 0.12, -0.22, -0.12, -0.15, 0.39, 0.08, 0.28, -0.12, -0.08, -0.33, 0.29, 0.33, 0.17, -0.25, 0.00, -0.22, -0.09, 0.21, -0.13, 0.11, 0.16, -0.35}})
            m_mlp.InitializeWeights(2, {
                {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00}})
            m_mlp.InitializeWeights(3, {
                {0.13, 0.41},
                {0.15, -0.25},
                {-0.21, 0.02},
                {-0.07, 0.2},
                {-0.05, -0.02},
                {-0.04, -0.17},
                {-0.14, 0.17},
                {-0.03, 0.11},
                {0.09, 0.29},
                {-0.24, -0.25},
                {0.03, 0.23},
                {0.09, 0.23},
                {0.33, 0.18},
                {0.15, -0.26},
                {-0.16, -0.2},
                {-0.05, 0.05},
                {0.05, -0.14},
                {-0.15, 0.04},
                {-0.38, -0.36},
                {-0.4, -0.42},
                {-0.19, -0.34},
                {0.41, -0.29},
                {-0.07, -0.12},
                {0.1, -0.26},
                {0.35, 0.22},
                {-0.1, -0.1},
                {0.12, -0.1},
                {0.19, 0.25},
                {-0.33, -0.4},
                {-0.35, 0.04},
                {-0.19, -0.2},
                {0.01, -0.12}})
            m_mlp.InitializeWeights(4, {{0.0, 0.0}})

            m_mlp.Train(enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArray2XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Dim expectedLoss# = 0.02
            Dim loss! = m_mlp.ComputeAverageError()
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub KerasMLP3XORSigmoid()

            Init3XOR()
            m_mlp.Initialize(learningRate:=0.02!, weightAdjustment:=0)

            m_mlp.InitializeStruct(m_neuronCount3XOR673, addBiasColumn:=False)

            m_mlp.nbIterations = 700
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=1, center:=0)

            m_mlp.InitializeWeights(1, {
                {0.2, -0.18, 0.21, -0.51, 0.62, -0.3, -0.18},
                {-0.31, -0.14, -0.41, 0.14, 0.42, -0.41, 0.61},
                {0.46, 0.16, -0.41, 0.33, -0.41, 0.32, 0.16},
                {0.31, 0.21, 0.65, -0.21, 0.35, 0.61, 0.62},
                {0.49, 0.44, -0.46, 0.33, -0.59, 0.51, -0.51},
                {-0.07, -0.4, 0.08, 0.29, 0.16, 0.57, 0.38}})
            m_mlp.InitializeWeights(2, {
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}})
            m_mlp.InitializeWeights(3, {
                {-0.64, 0.44, -0.48},
                {-0.16, 0.17, 0.41},
                {0.59, 0.21, -0.67},
                {-0.43, -0.07, 0.01},
                {-0.66, -0.14, 0.35},
                {0.66, -0.11, -0.44},
                {-0.67, -0.18, -0.23}})
            m_mlp.InitializeWeights(4, {{0.0, 0.0, 0.0}})

            m_mlp.Train(enumLearningMode.VectorialBatch)

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

        <TestMethod()>
        Public Sub KerasMLP3XORHTangent()

            Init3XOR()
            m_mlp.Initialize(learningRate:=0.02!, weightAdjustment:=0)

            m_mlp.InitializeStruct(m_neuronCount3XOR6_32_3, addBiasColumn:=False)

            m_mlp.nbIterations = 600 '1000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=1, center:=0)

            m_mlp.InitializeWeights(1, {
                {-0.2, 0.15, 0.03, 0.3, 0.15, -0.05, -0.15, 0.37, -0.12, -0.19, -0.06, -0.23, 0.31, -0.31, 0.09, -0.19, -0.24, -0.38, 0.17, 0.00, 0.22, 0.18, -0.02, -0.28, 0.3, 0.24, 0.18, -0.26, -0.25, 0.03, -0.3, -0.07},
                {-0.04, -0.32, -0.2, 0.19, 0.35, 0.3, 0.3, -0.17, -0.38, 0.3, -0.17, 0.07, -0.04, -0.34, -0.14, 0.31, 0.24, 0.03, 0.38, 0.04, 0.02, 0.1, -0.32, -0.1, -0.14, 0.25, 0.12, 0.25, 0.03, 0.17, 0.17, 0.05},
                {0.22, 0.14, 0.34, -0.02, -0.17, 0.11, -0.28, -0.28, 0.05, 0.21, 0.02, 0.36, 0.22, 0.17, 0.04, 0.38, 0.02, -0.25, -0.28, -0.36, -0.11, 0.02, 0.32, 0.09, 0.24, -0.23, -0.36, -0.04, 0.3, 0.27, -0.33, -0.2},
                {-0.08, -0.01, -0.37, 0.11, -0.23, -0.35, -0.35, -0.11, 0.28, -0.32, -0.09, 0.19, 0.34, 0.2, 0.29, 0.3, -0.22, -0.2, -0.13, 0.15, 0.21, -0.38, -0.02, -0.21, 0.24, 0.00, -0.11, -0.29, -0.26, -0.24, 0.07, -0.16},
                {0.17, 0.06, -0.19, 0.09, 0.2, -0.39, -0.2, -0.26, 0.27, -0.35, -0.35, -0.37, 0.00, 0.27, 0.1, -0.05, -0.28, 0.23, -0.37, 0.25, 0.28, 0.05, -0.11, -0.01, -0.07, 0.12, 0.13, 0.37, 0.35, -0.29, 0.05, 0.37},
                {-0.02, 0.2, -0.17, 0.02, -0.36, -0.11, -0.07, -0.27, -0.08, 0.2, -0.31, 0.28, 0.17, -0.3, 0.05, -0.22, 0.14, 0.39, 0.39, -0.35, 0.1, -0.05, 0.03, -0.27, -0.29, -0.25, -0.32, -0.19, 0.14, -0.29, 0.31, 0.15}})
            m_mlp.InitializeWeights(2, {
                {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00}})
            m_mlp.InitializeWeights(3, {
                {-0.12, 0.13, -0.15},
                {0.05, 0.25, -0.03},
                {-0.37, -0.27, -0.24},
                {-0.36, -0.24, 0.37},
                {-0.09, -0.26, -0.13},
                {0.00, -0.1, 0.21},
                {0.36, -0.21, 0.27},
                {0.22, 0.32, -0.22},
                {0.2, -0.15, 0.34},
                {-0.17, 0.41, -0.15},
                {-0.01, 0.11, 0.24},
                {0.05, -0.4, 0.21},
                {0.16, 0.34, -0.38},
                {0.36, -0.27, -0.03},
                {0.08, 0.19, 0.08},
                {-0.27, -0.28, -0.16},
                {0.22, -0.39, 0.41},
                {-0.03, 0.26, -0.15},
                {-0.09, 0.06, 0.31},
                {-0.28, -0.11, 0.06},
                {-0.01, 0.12, 0.25},
                {0.18, 0.11, 0.34},
                {-0.28, 0.27, 0.3},
                {0.14, -0.2, 0.2},
                {-0.26, 0.33, -0.16},
                {0.26, 0.4, 0.04},
                {0.31, 0.34, -0.27},
                {-0.4, -0.24, -0.33},
                {-0.3, -0.31, -0.21},
                {0.31, -0.22, -0.34},
                {0.2, 0.39, 0.01},
                {-0.3, 0.13, 0.09}})
            m_mlp.InitializeWeights(4, {{0.0, 0.0, 0.0}})

            m_mlp.Train(enumLearningMode.VectorialBatch)

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

#End If