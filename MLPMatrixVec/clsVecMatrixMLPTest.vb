
Option Infer On

Imports Perceptron
Imports Perceptron.MLP
'Imports Perceptron.ActivationFunction
Imports Perceptron.MLP.ActivationFunction

Imports Microsoft.VisualStudio.TestTools.UnitTesting


Namespace VectorizedMatrixMLP

    <TestClass()> _
    Public Class clsVecMatrixMLPTest

        Private m_mlp As New clsVectorizedMatrixMLP

        <TestInitialize()>
        Public Sub Init()
        End Sub

        Private Sub InitXOR()
            m_mlp.Init(learningRate:=0.1!, weightAdjustment:=1)
            m_mlp.input = m_inputArrayXOR
            m_mlp.target = m_targetArrayXOR
            m_mlp.targetArray = m_targetArrayXOR
        End Sub

        Private Sub Init2XOR()
            m_mlp.Init(learningRate:=0.1!, weightAdjustment:=1)
            m_mlp.input = m_inputArray2XOR
            m_mlp.target = m_targetArray2XOR
            m_mlp.targetArray = m_targetArray2XOR
        End Sub

        Private Sub Init3XOR()
            m_mlp.Init(learningRate:=0.1!, weightAdjustment:=1)
            m_mlp.input = m_inputArray3XOR
            m_mlp.target = m_targetArray3XOR
            m_mlp.targetArray = m_targetArray3XOR
        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLPXORSigmoid()

            InitXOR()
            m_mlp.nbIterations = 30000 ' Sigmoid: works
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=0)

            m_mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.w(0) = {
                 {-0.5, -0.78, -0.07},
                 {0.54, 0.32, -0.13},
                 {-0.29, 0.89, -0.8}}
            m_mlp.w(1) = {
                 {0.28},
                 {-0.94},
                 {-0.5},
                 {-0.36}}

            m_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)

            Dim expectedOutput = New Double(,) {
                {0.97},
                {0.02},
                {0.97},
                {0.02}}

            Dim sOutput = m_mlp.outputMatrix.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.1
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            Assert.AreEqual(rExpectedLoss, rLossRounded)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLPXORSigmoidWithoutBias()

            InitXOR()
            m_mlp.nbIterations = 100000 ' Sigmoid: works
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=0)

            m_mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=False)

            m_mlp.w(0) = {
                {0.38, 0.55},
                {0.24, 0.58}}
            m_mlp.w(1) = {
                {0.2},
                {0.16}}

            m_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)

            Dim expectedOutput = New Double(,) {
                {0.93},
                {0.03},
                {0.93},
                {0.09}}

            Dim sOutput = m_mlp.outputMatrix.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.26
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            Assert.AreEqual(rExpectedLoss, rLossRounded)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLPXORHTangent()

            InitXOR()
            m_mlp.nbIterations = 5000 ' Hyperbolic tangent: works
            m_mlp.SetActivationFunction(TActivationFunction.HyperbolicTangent, gain:=1, center:=0)

            m_mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.w(0) = {
                 {-0.5, -0.78, -0.07},
                 {0.54, 0.32, -0.13},
                 {-0.29, 0.89, -0.8}}
            m_mlp.w(1) = {
                 {0.28},
                 {-0.94},
                 {-0.5},
                 {-0.36}}

            m_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)

            Dim expectedOutput = New Double(,) {
                {0.99},
                {0.0},
                {0.99},
                {0.0}}

            Dim sOutput = m_mlp.outputMatrix.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.02
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            Assert.AreEqual(rExpectedLoss, rLossRounded)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLPXORGaussian()

            InitXOR()
            m_mlp.nbIterations = 400 ' Gaussian: works
            m_mlp.SetActivationFunction(TActivationFunction.Gaussian, gain:=1, center:=0)

            m_mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.w(0) = {
                {-0.5, -0.78, -0.07},
                {0.54, 0.32, -0.13},
                {-0.29, 0.89, -0.8}}
            m_mlp.w(1) = {
                {0.28},
                {-0.94},
                {-0.5},
                {-0.36}}

            m_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)

            Dim expectedOutput = New Double(,) {
                {0.99},
                {0.03},
                {0.99},
                {0.03}}

            Dim sOutput = m_mlp.outputMatrix.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.08
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            Assert.AreEqual(rExpectedLoss, rLossRounded)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLPXORSinus()

            InitXOR()
            m_mlp.nbIterations = 200 ' Sinus: works
            m_mlp.SetActivationFunction(TActivationFunction.Sinus, gain:=1, center:=0)

            m_mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=False)

            m_mlp.w(0) = {
                {0.7, 0.8},
                {0.04, 0.59}}
            m_mlp.w(1) = {
                {0.03},
                {0.66}}

            m_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)

            Dim expectedOutput = New Double(,) {
                {0.99},
                {0.0},
                {0.99},
                {0.0}}

            Dim sOutput = m_mlp.outputMatrix.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.02
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            Assert.AreEqual(rExpectedLoss, rLossRounded)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLPXORArcTangent()

            InitXOR()
            m_mlp.nbIterations = 500 ' Arc Tangent: works
            m_mlp.SetActivationFunction(TActivationFunction.ArcTangent, gain:=1, center:=0)

            m_mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.w(0) = {
                {-0.5, -0.78, -0.07},
                {0.54, 0.32, -0.13},
                {-0.29, 0.89, -0.8}}
            m_mlp.w(1) = {
                {0.28},
                {-0.94},
                {-0.5},
                {-0.36}}

            m_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)

            Dim expectedOutput = New Double(,) {
                {0.99},
                {0},
                {0.99},
                {0}}

            Dim sOutput = m_mlp.outputMatrix.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.03
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            Assert.AreEqual(rExpectedLoss, rLossRounded)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLPXORELU()

            InitXOR()
            m_mlp.nbIterations = 400 ' ELU: works
            m_mlp.SetActivationFunction(TActivationFunction.ELU, gain:=0.1, center:=0.4)
            m_mlp.Init(learningRate:=0.07!, weightAdjustment:=1)

            m_mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.w(0) = {
                {0.37, 0.08, 0.74},
                {0.59, 0.54, 0.32},
                {0.2, 0.78, 0.01}}
            m_mlp.w(1) = {
                {0.5},
                {0.27},
                {0.4},
                {0.11}}

            m_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)

            Dim expectedOutput = New Double(,) {
                {0.99},
                {0.02},
                {0.99},
                {0.01}}

            Dim sOutput = m_mlp.outputMatrix.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.04
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            Assert.AreEqual(rExpectedLoss, rLossRounded)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLPXORReLUSigmoid()

            InitXOR()
            m_mlp.nbIterations = 5000 ' ReLUSigmoid: works fine
            m_mlp.SetActivationFunction(TActivationFunction.ReLuSigmoid, gain:=1, center:=0)

            m_mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=True)

            m_mlp.w(0) = {
                {-0.5, -0.78, -0.07},
                {0.54, 0.32, -0.13},
                {-0.29, 0.89, -0.8}}
            m_mlp.w(1) = {
                {0.28},
                {-0.94},
                {-0.5},
                {-0.36}}

            m_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)

            Dim expectedOutput = m_targetArrayXOR

            Dim sOutput = m_mlp.outputMatrix.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.01
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            Assert.AreEqual(rExpectedLoss, rLossRounded)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLP2XORSinus()

            Init2XOR()
            m_mlp.nbIterations = 5000 ' Sinus: works
            m_mlp.SetActivationFunction(TActivationFunction.Sinus, gain:=1, center:=0)

            m_mlp.InitStruct(m_neuronCount2XOR, addBiasColumn:=True)

            m_mlp.w(0) = {
                {0.85, 0.86, 0.13, 0.16, 0.57},
                {0.93, 0.8, 0.45, 0.62, 0.96},
                {0.27, 0.54, 0.46, 0.41, 0.03},
                {0.82, 0.73, 0.7, 0.89, 0.38},
                {0.28, 0.95, 0.12, 0.28, 0.28}}
            m_mlp.w(1) = {
                {0.48, 0.65},
                {0.86, 0.63},
                {0.97, 0.3},
                {0.3, 0.7},
                {0.06, 0.66},
                {0.05, 0.69}}

            m_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)

            Dim expectedOutput = m_targetArray2XOR

            Dim sOutput = m_mlp.outputMatrix.ToString()
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString()
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.02
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            Assert.AreEqual(rExpectedLoss, rLossRounded)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLP2XORGaussian()

            Init2XOR()
            m_mlp.nbIterations = 1000 ' Gaussian: works
            m_mlp.SetActivationFunction(TActivationFunction.Gaussian, gain:=1, center:=0)

            m_mlp.InitStruct(m_neuronCount2XOR, addBiasColumn:=True)

            m_mlp.w(0) = {
                {0.26, 0.34, 0.51, 0.79, 0.37},
                {0.38, 0.84, 0.97, 0.65, 0.93},
                {0.2, 0.22, 0.69, 0.25, 0.68},
                {0.45, 0.58, 0.77, 0.59, 0.46},
                {0.58, 0.46, 0.93, 0.47, 0.28}}
            m_mlp.w(1) = {
                {0.23, 0.29},
                {0.75, 0.21},
                {0.52, 0.38},
                {0.63, 0.68},
                {0.96, 0.53},
                {0.31, 0.91}}

            m_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)

            Dim expectedOutput = m_targetArray2XOR

            Dim sOutput = m_mlp.outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.14
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            Assert.AreEqual(rExpectedLoss, rLossRounded)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLP2XORHTan()

            Init2XOR()
            m_mlp.nbIterations = 2000 ' HTan: works
            m_mlp.SetActivationFunction(TActivationFunction.HyperbolicTangent, gain:=1, center:=0.5)

            m_mlp.InitStruct(m_neuronCount2XOR, addBiasColumn:=True)

            m_mlp.w(0) = {
                {0.21, 0.81, 0.81, 0.57, 0.47},
                {0.5, 0.48, 0.22, 0.17, 0.17},
                {0.73, 0.29, 0.55, 0.41, 0.86},
                {0.91, 0.74, 0.87, 0.32, 0.85},
                {0.99, 0.31, 0.83, 0.95, 0.03}}
            m_mlp.w(1) = {
                {0.26, 0.09},
                {0.1, 0.16},
                {0.04, 0.87},
                {0.88, 0.8},
                {0.34, 0.99},
                {0.9, 0.28}}

            m_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)

            Dim expectedOutput = m_targetArray2XOR

            Dim sOutput = m_mlp.outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.25
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            Assert.AreEqual(rExpectedLoss, rLossRounded)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLP2XORSigmoid()

            Init2XOR()
            m_mlp.nbIterations = 5000 ' Sigmoid: works
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=0)

            m_mlp.InitStruct(m_neuronCount2XOR, addBiasColumn:=True)

            m_mlp.w(0) = {
                {0.42, 0.38, 0.53, 0.72, 0.19},
                {0.79, 0.54, 0.55, 0.31, 0.09},
                {0.12, 0.04, 0.12, 0.62, 0.58},
                {0.72, 0.91, 0.06, 0.69, 0.11},
                {0.38, 0.91, 0.72, 0.18, 0.68}}
            m_mlp.w(1) = {
                {0.51, 0.43},
                {0.18, 0.43},
                {0.19, 0.86},
                {0.26, 0.86},
                {0.73, 0.63},
                {0.23, 0.95}}

            m_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)

            Dim expectedOutput = m_targetArray2XOR

            Dim sOutput = m_mlp.outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.47
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            Assert.AreEqual(rExpectedLoss, rLossRounded)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLP2XORArcTan()

            Init2XOR()
            m_mlp.nbIterations = 20000 ' Arc tangent: works bad
            m_mlp.SetActivationFunction(TActivationFunction.ArcTangent, gain:=1, center:=0)

            m_mlp.InitStruct(m_neuronCount2XOR, addBiasColumn:=True)

            m_mlp.w(0) = {
                {0.28, 0.32, 0.65, 0.33, 0.18},
                {0.42, 0.2, 0.09, 0.53, 0.49},
                {0.43, 0.9, 0.41, 0.48, 0.2},
                {0.83, 0.59, 0.24, 0.83, 0.64},
                {0.15, 0.07, 0.96, 0.98, 0.16}}
            m_mlp.w(1) = {
                {0.05, 0.91},
                {0.66, 0.52},
                {0.43, 0.03},
                {0.65, 0.93},
                {0.67, 0.48},
                {0.51, 0.09}}

            m_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)

            Dim expectedOutput = m_targetArray2XOR

            Dim sOutput = m_mlp.outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.23
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            Assert.AreEqual(rExpectedLoss, rLossRounded)

        End Sub

        <TestMethod()> _
        Public Sub VectorizedMatrixMLP3XORGaussian()

            Init3XOR()
            m_mlp.nbIterations = 2000 ' Gaussian: works
            m_mlp.SetActivationFunction(TActivationFunction.Gaussian, gain:=1, center:=0)

            m_mlp.InitStruct(m_neuronCount3XOR, addBiasColumn:=True)

            m_mlp.w(0) = {
                {0.67, 0.05, 0.56, 0.02, 0.2, 0.87, 1.0},
                {0.14, 0.59, 0.89, 0.43, 0.33, 0.11, 0.76},
                {0.04, 0.71, 0.65, 0.17, 0.52, 0.85, 0.37},
                {1.0, 0.94, 0.86, 0.9, 0.72, 0.12, 0.3},
                {0.96, 0.95, 0.58, 0.95, 0.13, 0.96, 0.2},
                {0.69, 0.6, 0.32, 0.32, 0.53, 0.73, 0.43},
                {0.02, 0.04, 0.16, 0.73, 0.08, 0.06, 0.07}}
            m_mlp.w(1) = {
                {0.4, 0.65, 0.13},
                {0.65, 0.01, 0.04},
                {0.68, 0.11, 0.7},
                {0.12, 0.48, 0.75},
                {0.7, 0.18, 0.65},
                {0.31, 0.47, 0.2},
                {0.14, 0.56, 0.36},
                {0.11, 0.32, 0.85}}

            m_mlp.TrainVector()
            'm_mlp.Train(clsMLPGeneric.enumLearningMode.Vectoriel)

            Dim expectedOutput = m_targetArray3XOR

            Dim sOutput = m_mlp.outputMatrix.ToStringWithFormat(dec:="0.0")
            Dim expectedMatrix As Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.13
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            Assert.AreEqual(rExpectedLoss, rLossRounded)

        End Sub

    End Class

End Namespace