
Option Infer On

Imports Microsoft.VisualStudio.TestTools.UnitTesting

Namespace MatrixMLP

    <TestClass()>
    Public Class MultiLayerPerceptronTest

        Private m_mlp As New MultiLayerPerceptron

        ' Some functions are not implemented
        'Private m_mlp As New NetworkOOP.MultilayerPerceptron
        'Private m_mlp As New clsMLPClassic 
        'Private m_mlp As New VectorizedMatrixMLP.clsVectorizedMatrixMLP 

        <TestInitialize()>
        Public Sub Init()
        End Sub

        Private Sub InitXOR()
            m_mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=True)
            m_mlp.inputArray = m_inputArrayXOR
            m_mlp.targetArray = m_targetArrayXOR
        End Sub

        Private Sub Init2XOR()
            m_mlp.InitStruct(m_neuronCount2XOR, addBiasColumn:=True)
            m_mlp.inputArray = m_inputArray2XOR
            m_mlp.targetArray = m_targetArray2XOR
        End Sub

        Private Sub Init3XOR()
            m_mlp.InitStruct(m_neuronCount3XOR, addBiasColumn:=True)
            m_mlp.inputArray = m_inputArray3XOR
            m_mlp.targetArray = m_targetArray3XOR
        End Sub

        <TestMethod()>
        Public Sub MatrixMLPXORSigmoid()

            Dim nbIterations% = 10000
            m_mlp.SetActivationFunctionForMatrix(TActivationFunctionForMatrix.Sigmoid,
                gain:=1, center:=2.2)

            InitXOR()
            m_mlp.Init(learningRate:=0.9, weightAdjustment:=0.9)

            m_mlp.weights_ih = {
                {0.95, 0.82},
                {0.78, 0.27}}
            m_mlp.weights_ho = {
                {0.77, 0.27}}
            m_mlp.bias_h = {
                {0.91},
                {0.9}}
            m_mlp.bias_o = {
                {0.01}}

            m_mlp.nbIterations = nbIterations
            m_mlp.Train()

            Dim expectedOutput = New Single(,) {
                {0.99},
                {0.01},
                {0.99},
                {0.01}}

            Dim sOutput = m_mlp.output.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.01
            m_mlp.targetArray = m_targetArrayXOR
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            Assert.AreEqual(rExpectedLoss, rLossRounded)

        End Sub

        <TestMethod()>
        Public Sub MatrixMLPXORSigmoidWithoutBias()

            Dim nbIterations% = 100000
            m_mlp.SetActivationFunctionForMatrix(TActivationFunctionForMatrix.Sigmoid,
                gain:=1, center:=0)

            InitXOR()
            m_mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=False)
            m_mlp.Init(learningRate:=0.1, weightAdjustment:=0.1)

            m_mlp.weights_ih = {
                {0.76, 0.81},
                {0.09, 0.16}}
            m_mlp.weights_ho = {
                {0.17, 0.17}}

            m_mlp.nbIterations = nbIterations
            m_mlp.Train()

            Dim expectedOutput = New Single(,) {
                {0.92},
                {0.01},
                {0.92},
                {0.06}}

            Dim sOutput = m_mlp.output.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.06
            m_mlp.targetArray = m_targetArrayXOR
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            Assert.AreEqual(rExpectedLoss, rLossRounded)

        End Sub

        <TestMethod()>
        Public Sub MatrixMLPXORHyperbolicTangent()

            Dim nbIterations% = 5000
            m_mlp.SetActivationFunctionForMatrix(TActivationFunctionForMatrix.HyperbolicTangent,
                gain:=1, center:=0)

            InitXOR()
            m_mlp.Init(learningRate:=0.05, weightAdjustment:=0.05)

            m_mlp.weights_ih = {
                {0.79, 0.81},
                {0.41, 0.84}}
            m_mlp.weights_ho = {
                {0.87, 0.11}}
            m_mlp.bias_h = {
                {0.31},
                {0.81}}
            m_mlp.bias_o = {
                {0.6}}

            m_mlp.nbIterations = nbIterations
            m_mlp.Train()

            Dim expectedOutput = New Single(,) {
                {0.97},
                {0},
                {0.97},
                {0}}

            Dim sOutput = m_mlp.output.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.01
            m_mlp.targetArray = m_targetArrayXOR
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            Assert.AreEqual(rExpectedLoss, rLossRounded)

        End Sub

        <TestMethod()>
        Public Sub MatrixMLPXORELU()

            Dim nbIterations% = 300
            m_mlp.SetActivationFunctionForMatrix(TActivationFunctionForMatrix.ELU,
                gain:=1, center:=-1.8)

            InitXOR()
            m_mlp.Init(learningRate:=0.1, weightAdjustment:=0.1)

            m_mlp.weights_ih = {
                {0.57, 0.6},
                {0.34, 0.14}}
            m_mlp.weights_ho = {
                {0.8, 0.46}}
            m_mlp.bias_h = {
                {0.24},
                {0.29}}
            m_mlp.bias_o = {
                {0.27}}

            m_mlp.nbIterations = nbIterations
            m_mlp.Train()

            Dim expectedOutput = New Single(,) {
                {1},
                {0},
                {1},
                {0}}

            Dim sOutput = m_mlp.output.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0
            m_mlp.targetArray = m_targetArrayXOR
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            Assert.AreEqual(rExpectedLoss, rLossRounded)

        End Sub

        <TestMethod()>
        Public Sub MatrixMLP2XORHyperbolicTangent()

            Dim nbIterations% = 5000
            m_mlp.SetActivationFunctionForMatrix(
                TActivationFunctionForMatrix.HyperbolicTangent, gain:=1, center:=0.5)

            Init2XOR()
            m_mlp.Init(learningRate:=0.1, weightAdjustment:=0.1)

            m_mlp.weights_ih = {
                {0.39, 0.5, 0.3, 0.25},
                {0.31, 0.07, 0.53, 0.15},
                {0.84, 0.47, 0.91, 0.86},
                {0.88, 0.13, 0.34, 0.81}}
            m_mlp.weights_ho = {
                {0.58, 0.76, 0.12, 0.45},
                {0.8, 0.01, 0.67, 0.75}}
            m_mlp.bias_h = {
                {0.74},
                {0.8},
                {0.05},
                {0.37}}
            m_mlp.bias_o = {
                {0.66},
                {0.42}}

            m_mlp.nbIterations = nbIterations
            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.01
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            Assert.AreEqual(rExpectedLoss, rLossRounded)

        End Sub

    End Class

End Namespace