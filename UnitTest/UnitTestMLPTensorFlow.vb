
Imports Perceptron
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPClassic ' enumLearningMode
Imports Microsoft.VisualStudio.TestTools.UnitTesting

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