
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Imports Microsoft.VisualStudio.TestTools.UnitTesting

Module modMLPTensorFlowTest

    Sub Main()
        Console.WriteLine("TensorFlow MLP with the classical XOR test.")
        TensorFlowMLPXORTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()
    End Sub

    Public Sub TensorFlowMLPXORTest(Optional nbXor% = 1)

        Dim mlp As New clsMLPTensorFlow

        mlp.ShowMessage("TensorFlow MLP Xor test")
        mlp.ShowMessage("-----------------------")

        mlp.nbIterations = 5000 ' Hyperbolic tangent: works
        mlp.printOutput_ = True
        mlp.printOutputMatrix = False

        If nbXor = 1 Then
            mlp.nbIterations = 500
            mlp.Initialize(learningRate:=0.2!)
            mlp.printOutputMatrix = True
            mlp.inputArray = m_inputArrayXOR
            mlp.targetArray = m_targetArrayXOR
            mlp.InitializeStruct(m_neuronCountXOR261, addBiasColumn:=False)
        ElseIf nbXor = 2 Then
            ' 75% success
            mlp.nbIterations = 5000
            mlp.Initialize(learningRate:=0.1!)
            mlp.inputArray = m_inputArray2XOR
            mlp.targetArray = m_targetArray2XOR
            mlp.InitializeStruct(m_neuronCount2XOR462, addBiasColumn:=False)
        ElseIf nbXor = 3 Then
            ' 190/192: 99% success
            mlp.nbIterations = 10000
            mlp.Initialize(learningRate:=0.05!)
            mlp.inputArray = m_inputArray3XOR
            mlp.targetArray = m_targetArray3XOR
            mlp.InitializeStruct(m_neuronCount3XOR673, addBiasColumn:=False)
        End If

        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

        mlp.Randomize()

        mlp.PrintWeights()

        WaitForKeyToStart()

        mlp.TrainVector() ' Works fine

        mlp.ShowMessage("TensorFlow MLP Xor test: Done.")

        If nbXor > 1 Then Exit Sub

        WaitForKeyToContinue("Press a key to print MLP weights")
        mlp.PrintWeights()

    End Sub

    Public Sub TensorFlowMLPIrisFlowerAnalogTest()

        Dim mlp As New clsMLPTensorFlow
        mlp.ShowMessage("TensorFlow.NET MLP Iris flower analog test")
        mlp.ShowMessage("------------------------------------------")

        mlp.nbIterations = 10000

        mlp.Initialize(learningRate:=0.01!, weightAdjustment:=0.01!)

        mlp.printOutput_ = True
        mlp.printOutputMatrix = False

        mlp.inputArray = m_inputArrayIrisFlower
        mlp.targetArray = m_targetArrayIrisFlowerAnalog
        mlp.InitializeStruct(m_neuronCountIrisFlowerAnalog4_20_1, addBiasColumn:=False)

        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

        mlp.Randomize()

        mlp.PrintParameters()

        WaitForKeyToStart()

        mlp.minimalSuccessTreshold = 0.2
        mlp.Train()

        mlp.ShowMessage("TensorFlow.NET MLP Iris flower analog test: Done.")

    End Sub

End Module

Namespace TensorFlowMLP

    <TestClass()>
    Public Class clsMLPTensorFlowTest

        Private m_mlp As New clsMLPTensorFlow

        ' Weights are not initialized:
        'Private m_mlp As New clsMLPClassic ' 0/2 success
        'Private m_mlp As New clsMLPOOP ' 0/2 success
        'Private m_mlp As New clsMLPAccord ' 0/2 success
        'Private m_mlp As New clsMLPEncog ' 0/2 success

        <TestInitialize()>
        Public Sub Init()
        End Sub

        Private Sub InitXOR()
            m_mlp.inputArray = m_inputArrayXOR
            m_mlp.targetArray = m_targetArrayXOR
            m_mlp.InitializeStruct(m_neuronCountXOR261, addBiasColumn:=False)
        End Sub

        Private Sub Init2XOR()
            m_mlp.inputArray = m_inputArray2XOR
            m_mlp.targetArray = m_targetArray2XOR
            m_mlp.InitializeStruct(m_neuronCount2XOR462, addBiasColumn:=False)
        End Sub

        Private Sub Init3XOR()
            m_mlp.inputArray = m_inputArray3XOR
            m_mlp.targetArray = m_targetArray3XOR
            m_mlp.InitializeStruct(m_neuronCount3XOR673, addBiasColumn:=False)
        End Sub

        <TestMethod()>
        Public Sub TensorFlowMLP1XORTanh261()

            TestMLP1XORTanh261(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub TensorFlowMLP1XORTanh()

            InitXOR()
            m_mlp.Initialize(learningRate:=0.2!)
            m_mlp.InitializeStruct(m_neuronCountXOR261, addBiasColumn:=False)

            m_mlp.nbIterations = 400
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            'm_mlp.InitializeWeights(1, {{}})
            'm_mlp.InitializeWeights(2, {{}})

            m_mlp.Train(enumLearningMode.Vectorial)

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

        ' Bias is not implemented
        '<TestMethod()>
        'Public Sub TensorFlowMLP2XORHSigmoid()

        '    TestMLP2XORSigmoid(m_mlp)

        'End Sub

        ' Does not work:

        '<TestMethod()>
        'Public Sub TensorFlowMLP2XORTanh()

        '    Init2XOR()
        '    m_mlp.Initialize(learningRate:=0.1!)

        '    m_mlp.nbIterations = 4000
        '    m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

        '    'm_mlp.InitializeWeights(1, {{}})
        '    'm_mlp.InitializeWeights(2, {{}})

        '    m_mlp.Train(enumLearningMode.Vectorial)

        '    Dim expectedOutput = m_targetArray2XOR
        '    Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

        '    Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

        '    Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        '    Assert.AreEqual(sExpectedOutput, sOutput)

        '    Const expectedLoss# = 0.01
        '    Dim loss# = m_mlp.averageError
        '    Dim lossRounded# = Math.Round(loss, 2)
        '    Assert.AreEqual(True, lossRounded <= expectedLoss)

        'End Sub

        '<TestMethod()>
        'Public Sub TensorFlowMLP2XORTanh462()

        '    TestMLP2XORTanh462(m_mlp, nbIterations:=10000)

        'End Sub

        '<TestMethod()>
        'Public Sub TensorFlowMLP3XORTanh()

        '    TestMLP3XORTanh(m_mlp) 

        'End Sub

        '<TestMethod()>
        'Public Sub TensorFlowMLP3XORTanh()

        '    TestMLP3XORTanh(m_mlp, learningMode:=enumLearningMode.Vectorial)

        'End Sub

    End Class

End Namespace