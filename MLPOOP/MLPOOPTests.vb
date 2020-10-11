
Imports Perceptron.NetworkOOP
Imports Perceptron.Activation
Imports Perceptron.Utilities
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Imports Microsoft.VisualStudio.TestTools.UnitTesting

Module Main

    Sub Main()

        OOPMLPXorTest()

    End Sub

    Public Sub OOPMLPXorTest(Optional nbXor% = 1)

        Dim standard As New Randoms.Standard(
            New Range(-1, 1), seed:=DateTime.Now.Millisecond)

        Dim mlp As New MultilayerPerceptron(
            learning_rate:=0.5,
            momentum:=0.8, randomizer:=standard,
            activation:=New HyperbolicTangent(alpha:=0.5#))

        'mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=0.5)
        'mlp.ActivationFunction = Nothing

        mlp.ShowMessage("Object-oriented programming MLP Xor test")
        mlp.ShowMessage("----------------------------------------")

        mlp.printOutput_ = True
        mlp.printOutputMatrix = False
        mlp.nbIterations = 2000

        If nbXor = 1 Then
            'num_input:=2, num_hidden:={5}, num_output:=1
            'mlp.InitializeStruct({2, 5, 1}, addBiasColumn:=True)
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
            mlp.inputArray = m_inputArray3XOR
            mlp.targetArray = m_targetArray3XOR
            mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
        End If

        mlp.Randomize()
        mlp.PrintWeights()

        WaitForKeyToStart()

        'Dim Training As New List(Of Training)
        'Training.Add(New Training({0, 1}, {1}))
        'Training.Add(New Training({0, 0}, {0}))
        'Training.Add(New Training({1, 0}, {1}))
        'Training.Add(New Training({1, 1}, {0}))

        'Training.Add(New Training({1, 0, 1, 0}, {1, 1}))
        'Training.Add(New Training({1, 0, 0, 0}, {1, 0}))
        'Training.Add(New Training({1, 0, 0, 1}, {1, 1}))
        'Training.Add(New Training({1, 0, 1, 1}, {1, 0}))

        'Dim result = False
        'While Not result
        '    mlp.TrainOrig(Training, 5, 0.1)
        '    Console.WriteLine(String.Format(
        '        "Total error on correctly predicting training set: {0}",
        '        mlp.TotalError))
        '    Console.ReadLine()
        'End While

        'Dim nbIterations% = 3000
        'For iteration = 0 To nbIterations - 1
        '    mlp.TrainOneIteration(Training)
        '    If (iteration < 10 OrElse
        '        ((iteration + 1) Mod 100 = 0 AndAlso iteration < 1000) OrElse
        '        ((iteration + 1) Mod 1000 = 0 AndAlso iteration < 10000) OrElse
        '        (iteration + 1) Mod 10000 = 0) Then
        '        Dim msg$ = vbLf & "Iteration n°" & iteration + 1 & "/" & nbIterations & vbLf &
        '            "Output: " & mlp.PrintOutputOOP() & vbLf &
        '            "Average error: " & mlp.TotalError.ToString(format6Dec)
        '        Console.WriteLine(msg)
        '    End If
        'Next
        'Console.WriteLine("Press a key to quit.")
        'Console.ReadLine()

        mlp.Train()
        'mlp.Train(enumLearningMode.SemiStochastic) ' Works
        'mlp.Train(enumLearningMode.Stochastic) ' Works

        mlp.ShowMessage("Object-oriented programming MLP Xor test: Done.")

        If nbXor > 1 Then Exit Sub

        WaitForKeyToContinue("Press a key to print MLP weights")
        mlp.PrintWeights()

    End Sub

End Module

Namespace OOPMLP

    <TestClass()>
    Public Class MultiLayerPerceptronOOPTest

        Private m_mlp As New MultilayerPerceptron

        ' Weights are quite the same as MLP Classic, but not exactly:
        'Private m_mlp As New clsMLPClassic ' 13 success, 4 fails
        'Private m_mlp As New clsMLPAccord ' 10 success, 7 fails
        'Private m_mlp As New clsMLPTensorFlow  ' 1 success, 16 fails

        ' Weights are not stored in the same way:
        'Private m_mlp As New MatrixMLP.MultiLayerPerceptron ' 15/15 fails
        'Private m_mlp As New VectorizedMatrixMLP.clsVectorizedMatrixMLP ' 15/15 fails
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

            'TestMLP1XORWithoutBias231b(m_mlp) Work only there

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

            'm_mlp.PrintWeights()

            m_mlp.Train()

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
        Public Sub MLPOOP1XORHTangent()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New HyperbolicTangent(alpha:=2.4#)

            TestMLP1XORHTangent(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP1XORHTangent261()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New HyperbolicTangent(alpha:=2.0#)

            TestMLP1XORHTangent261(m_mlp, nbIterations:=500)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP1XORELU()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New ELU(alpha:=0.6!, center:=0.3!)

            TestMLP1XORELU(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP2XORSigmoid()

            TestMLP2XORSigmoid(m_mlp, nbIterations:=600)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP2XORHTangent()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New HyperbolicTangent(alpha:=2.0#)

            TestMLP2XORHTangent(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP2XORHTangent462()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New HyperbolicTangent(alpha:=2.0#)

            TestMLP2XORHTangent462(m_mlp, nbIterations:=9000)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP3XORSigmoid()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New Sigmoid(alpha:=2)

            TestMLP3XORSigmoid(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP3XORHTangent()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New HyperbolicTangent(alpha:=1.0#)

            TestMLP3XORHTangent(m_mlp)

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
            Dim loss! = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub MLPOOPIrisFlowerAnalog()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New Sigmoid(alpha:=1.0#)

            TestMLPIrisFlowerAnalog(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPOOPIrisFlowerLogical()

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New Sigmoid(alpha:=1.0#)

            TestMLPIrisFlowerLogical(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub MLPOOPIrisFlowerLogicalPrediction()

            ' 97.8% prediction, 99.4% learning with 800 iterations in 3.7 sec.

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New HyperbolicTangent(alpha:=2.0#)

            TestMLPIrisFlowerLogicalPredictionTanh(m_mlp,
                nbIterations:=800, expectedSuccess:=0.989#)

        End Sub

        <TestMethod()>
        Public Sub MLPOOPIrisFlowerLogicalPredictionSigmoid()

            ' 97.8% prediction, 98.6% learning with 800 iterations in 2.3 sec.

            ' OOP activation function: before Initialize()
            'm_mlp.ActivationFunction = New Sigmoid(alpha:=1.0#)

            TestMLPIrisFlowerLogicalPredictionSigmoid(m_mlp, nbIterations:=800)

        End Sub

    End Class

End Namespace