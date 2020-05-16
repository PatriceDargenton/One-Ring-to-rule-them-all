
Option Infer On

Imports Microsoft.VisualStudio.TestTools.UnitTesting
Imports Perceptron.clsMLPClassic ' enumLearningMode

Module modMLPClassicTest

    Sub Main()
        Console.WriteLine("MultiLayerPerceptron with the classical XOR test.")
        ClassicMLPTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()
    End Sub

    Public Sub ClassicMLPTest()

        Dim mlp As New clsMLPClassic

        mlp.ShowMessage("Classic MLP test")
        mlp.ShowMessage("----------------")

        mlp.Init(learningRate:=0.1!, weightAdjustment:=0.1!)

        Dim nbIterations%

        'mlp.SetActivationFunction(TActivationFunction.SigmoideStandard, gain:=1, center:=0)
        'nbIterations = 10000 ' Sigmoid: works

        mlp.SetActivationFunction(TActivationFunction.HyperbolicTangent, gain:=1, center:=0)
        nbIterations = 1000 ' Hyperbolic tangent: works fine

        'mlp.SetActivationFunction(TActivationFunction.Gaussienne, gain:=1, center:=0)
        'nbIterations = 1000 ' Gaussian: works fine

        'mlp.SetActivationFunction(TActivationFunction.Sinus, gain:=1, center:=0)
        'nbIterations = 1000 ' Sinus: works fine

        'mlp.SetActivationFunction(TActivationFunction.ArcTangente, gain:=1, center:=0)
        'nbIterations = 1000 ' ArcTangent: works fine

        'mlp.SetActivationFunction(TActivationFunction.ELU, gain:=1, center:=0)
        'nbIterations = 10000 ' ELU: Does not work

        'mlp.SetActivationFunction(TActivationFunction.ReLu, gain:=0.9, center:=0)
        'nbIterations = 1000 ' ReLU: works fine

        'mlp.SetActivationFunction(TActivationFunction.ReLuSigmoid, gain:=1, center:=0)
        'nbIterations = 10000 ' ReLUSigmoid: works?

        'mlp.SetActivationFunction(TActivationFunction.DoubleThreshold, gain:=1, center:=0)
        'nbIterations = 10000 ' DoubleThreshold: works fine

        mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=True)

        mlp.Randomize()
        mlp.PrintWeights()

        Console.WriteLine()
        Console.WriteLine("Press a key to start.")
        Console.ReadKey()
        Console.WriteLine()

        'Dim arVal1#(,) = {
        '     {0.28, 0.28, 0.76},
        '     {0.25, 0.88, 0.62}}
        'Dim arVal2#(,) = {
        '     {0.56, 0.92, 0.19}}
        'mlp.InitPoids(1, arVal1)
        'mlp.InitPoids(2, arVal2)

        mlp.targetArray = m_targetArrayXOR

        mlp.printOutput_ = True
        mlp.nbIterations = nbIterations
        mlp.inputArray = m_inputArrayXOR
        mlp.Train()
        'mlp.Train(enumLearningMode.SemiStochastique)
        'mlp.Train(enumLearningMode.Stochastique)

        mlp.ShowMessage("Classic MLP test: Done.")

    End Sub

End Module

Namespace ClassicMLP

    <TestClass()>
    Public Class MultiLayerPerceptronTest

        Private m_mlp As New clsMLPClassic

        ' Weights are quite the same as MLP Classic, but not exactly:
        'Private m_mlp As New NetworkOOP.MultilayerPerceptron ' 10/16 fails

        ' Weights are not stored in the same way:
        'Private m_mlp As New MatrixMLP.MultiLayerPerceptron ' 13/13 fails
        'Private m_mlp As New VectorizedMatrixMLP.clsVectorizedMatrixMLP ' 13/13 fails

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
        Public Sub MLPXORHTangent()

            m_mlp.Init(learningRate:=0.25!, weightAdjustment:=0.9!)
            InitXOR()

            m_mlp.nbIterations = 1000
            m_mlp.SetActivationFunction(TActivationFunction.HyperbolicTangent, gain:=1, center:=0)

            m_mlp.WeightInit(1, {
                {0.28, 0.28, 0.76},
                {0.25, 0.88, 0.62}})
            m_mlp.WeightInit(2, {
                {0.56, 0.92, 0.19}})

            m_mlp.Train()

            Dim expectedOutput1K = New Double(,) {
                {0.99},
                {0.02},
                {0.99},
                {0}}

            'Dim expectedOutput10K = New Double(,) {
            '    {1},
            '    {0},
            '    {1},
            '    {0}}

            Dim outputMaxtrix As MatrixMLP.Matrix = m_mlp.outputArraySingle
            Dim sOutput = outputMaxtrix.ToStringWithFormat(dec:="0.0")

            Dim expectedMatrix As MatrixMLP.Matrix = expectedOutput1K ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss1K# = 0.01
            Dim rLoss# = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            Assert.AreEqual(rExpectedLoss1K, rLossRounded)

        End Sub

        <TestMethod()>
        Public Sub MLPXORSigmoid()

            m_mlp.Init(learningRate:=1.1!, weightAdjustment:=0.5!)
            InitXOR()

            m_mlp.nbIterations = 1100
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=1)

            m_mlp.WeightInit(1, {
                {0.42, 0.82, 0.15},
                {0.2, 0.8, 0.28}})
            m_mlp.WeightInit(2, {
                {0.71, 0.25, 0.35}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As MatrixMLP.Matrix = expectedOutput ' Double(,) -> Matrix

            Dim outputMaxtrix As MatrixMLP.Matrix = m_mlp.outputArraySingle
            Dim sOutput = outputMaxtrix.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.04
            Dim rLoss# = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub MLPXORSigmoidWithoutBias()

            m_mlp.Init(learningRate:=0.1!, weightAdjustment:=0.02!)
            InitXOR()
            m_mlp.InitStruct(m_neuronCountXOR, addBiasColumn:=False)

            m_mlp.nbIterations = 30000
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=0.2!)

            m_mlp.WeightInit(1, {
                {0.42, 0.79},
                {0.55, 0.02}})
            m_mlp.WeightInit(2, {
               {0.51, 0.31}})

            m_mlp.Train()

            'Dim expectedOutput = m_targetArrayXOR
            Dim expectedOutput = New Double(,) {
                {0.9},
                {0.1},
                {0.9},
                {0.2}}

            Dim expectedMatrix As MatrixMLP.Matrix = expectedOutput ' Double(,) -> Matrix

            Dim outputMaxtrix As MatrixMLP.Matrix = m_mlp.outputArraySingle
            Dim sOutput = outputMaxtrix.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.12
            Dim rLoss# = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub MLPXORSigmoidWithoutBias231()

            m_mlp.Init(learningRate:=0.08!, weightAdjustment:=0.02!)
            InitXOR()
            m_mlp.InitStruct(m_neuronCountXOR231, addBiasColumn:=False)

            m_mlp.nbIterations = 12000 '10000
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=0.0!)

            m_mlp.WeightInit(1, {
                {0.73, 0.38},
                {0.07, 0.3},
                {0.99, 0.25}})
            m_mlp.WeightInit(2, {
               {1.0, 0.98, 0.61}})

            m_mlp.Train()

            'Dim expectedOutput = m_targetArrayXOR
            Dim expectedOutput = New Double(,) {
                {0.9},
                {0.1},
                {0.9},
                {0.1}}

            Dim expectedMatrix As MatrixMLP.Matrix = expectedOutput ' Double(,) -> Matrix

            Dim outputMaxtrix As MatrixMLP.Matrix = m_mlp.outputArraySingle
            Dim sOutput = outputMaxtrix.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.1
            Dim rLoss# = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub MLPXORHReLU()

            m_mlp.Init(learningRate:=0.1!, weightAdjustment:=0.1!)
            InitXOR()

            m_mlp.nbIterations = 400
            m_mlp.SetActivationFunction(TActivationFunction.ReLu, gain:=0.9, center:=0)

            m_mlp.WeightInit(1, {
                {0.07, 0.79, 0.94},
                {0.33, 0.33, 0.93}})
            m_mlp.WeightInit(2, {
                {0.63, 0.79, 0.69}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As MatrixMLP.Matrix = expectedOutput ' Double(,) -> Matrix

            Dim outputMaxtrix As MatrixMLP.Matrix = m_mlp.outputArraySingle
            Dim sOutput = outputMaxtrix.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.04
            Dim rLoss# = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()> _
        Public Sub MLPXORHReLUSigmoid()

            m_mlp.Init(learningRate:=0.1!, weightAdjustment:=0.1!)
            InitXOR()

            m_mlp.nbIterations = 9000
            m_mlp.SetActivationFunction(TActivationFunction.ReLuSigmoid, gain:=1, center:=0)

            m_mlp.WeightInit(1, {
                {0.58, 0.23, 0.25},
                {0.88, 0.78, 0.18}})
            m_mlp.WeightInit(2, {
                {0.29, 0.34, 0.81}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As MatrixMLP.Matrix = expectedOutput ' Double(,) -> Matrix

            Dim outputMaxtrix As MatrixMLP.Matrix = m_mlp.outputArraySingle
            Dim sOutput = outputMaxtrix.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0
            Dim rLoss# = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub MLPXORHDbleThreshold()

            m_mlp.Init(learningRate:=0.1!, weightAdjustment:=0.1!)
            InitXOR()

            m_mlp.nbIterations = 5000
            m_mlp.SetActivationFunction(TActivationFunction.DoubleThreshold, gain:=1, center:=0)

            m_mlp.WeightInit(1, {
                {0.86, 0.37, 0.8},
                {1.0, 0.62, 0.17}})
            m_mlp.WeightInit(2, {
                {0.54, 0.83, 0.41}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As MatrixMLP.Matrix = expectedOutput ' Double(,) -> Matrix

            Dim outputMaxtrix As MatrixMLP.Matrix = m_mlp.outputArraySingle
            Dim sOutput = outputMaxtrix.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0
            Dim rLoss# = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub MLP2XORSigmoid()

            m_mlp.Init(learningRate:=1.8!, weightAdjustment:=0.04!)
            Init2XOR()

            m_mlp.nbIterations = 500
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=1)

            m_mlp.WeightInit(1, {
                {0.8, 0.72, 0.13, 0.33, 0.61},
                {0.41, 0.82, 0.97, 0.68, 0.22},
                {0.1, 0.65, 0.14, 0.36, 0.91},
                {0.93, 0.36, 0.06, 0.35, 0.9}})
            m_mlp.WeightInit(2, {
                {0.81, 0.09, 0.29, 0.53, 0.98},
                {0.54, 0.26, 0.25, 0.89, 0.35}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR
            Dim expectedMatrix As MatrixMLP.Matrix = expectedOutput ' Double(,) -> Matrix

            Dim outputMaxtrix As MatrixMLP.Matrix = m_mlp.outputArraySingle
            Dim sOutput = outputMaxtrix.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.02
            Dim rLoss# = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub MLP2XORHTangent()

            m_mlp.Init(learningRate:=0.1!, weightAdjustment:=0.1!)
            Init2XOR()

            m_mlp.nbIterations = 500
            m_mlp.SetActivationFunction(TActivationFunction.HyperbolicTangent, gain:=1, center:=0)

            m_mlp.WeightInit(1, {
                {0.37, 0.11, 0.32, 0.3, 0.2},
                {0.44, 0.42, 0.87, 0.24, 0.84},
                {0.86, 0.68, 0.29, 0.17, 0.68},
                {0.09, 0.57, 0.98, 0.48, 0.29}})
            m_mlp.WeightInit(2, {
                {0.5, 0.83, 0.36, 0.23, 0.45},
                {0.05, 0.37, 0.94, 0.7, 0.82}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR
            Dim expectedMatrix As MatrixMLP.Matrix = expectedOutput ' Double(,) -> Matrix

            Dim outputMaxtrix As MatrixMLP.Matrix = m_mlp.outputArraySingle
            Dim sOutput = outputMaxtrix.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.01
            Dim rLoss# = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub MLP2XORDbleThreshold()

            m_mlp.Init(learningRate:=0.1!, weightAdjustment:=0.1!)
            Init2XOR()

            m_mlp.nbIterations = 3000
            m_mlp.SetActivationFunction(TActivationFunction.DoubleThreshold, gain:=1, center:=0)

            m_mlp.WeightInit(1, {
                {0.43, 0.21, 0.16, 0.85, 0.2},
                {0.25, 0.5, 0.87, 0.5, 0.36},
                {0.21, 0.78, 0.68, 0.47, 0.35},
                {0.42, 0.09, 0.25, 0.87, 0.17}})
            m_mlp.WeightInit(2, {
                {0.88, 0.37, 0.12, 0.17, 0.79},
                {0.71, 0.88, 0.7, 0.83, 0.02}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR
            Dim expectedMatrix As MatrixMLP.Matrix = expectedOutput ' Double(,) -> Matrix

            Dim outputMaxtrix As MatrixMLP.Matrix = m_mlp.outputArraySingle
            Dim sOutput = outputMaxtrix.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0
            Dim rLoss# = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub MLP2XORReLU()

            m_mlp.Init(learningRate:=0.1!, weightAdjustment:=0.1!)
            Init2XOR()

            m_mlp.nbIterations = 200
            m_mlp.SetActivationFunction(TActivationFunction.ReLu, gain:=0.6, center:=0.4)

            m_mlp.WeightInit(1, {
                {0.97, 0.65, 0.44, 0.61, 0.73},
                {0.08, 0.65, 0.37, 0.13, 0.18},
                {0.23, 0.2, 0.64, 0.44, 0.27},
                {0.33, 0.46, 0.99, 0.49, 0.09}})
            m_mlp.WeightInit(2, {
                {0.02, 0.36, 0.11, 0.94, 0.82},
                {0.34, 0.92, 0.18, 1.0, 0.73}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR
            Dim expectedMatrix As MatrixMLP.Matrix = expectedOutput ' Double(,) -> Matrix

            Dim outputMaxtrix As MatrixMLP.Matrix = m_mlp.outputArraySingle
            Dim sOutput = outputMaxtrix.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0
            Dim rLoss# = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub MLP3XORGaussian()

            m_mlp.Init(learningRate:=0.15!, weightAdjustment:=0.25!)
            Init3XOR()

            m_mlp.nbIterations = 400
            m_mlp.SetActivationFunction(TActivationFunction.Gaussian, gain:=0.5!, center:=0.3!)

            m_mlp.WeightInit(1, {
                {0.39, 0.1, 0.21, 0.56, 0.84, 0.37, 0.06},
                {0.88, 0.81, 0.94, 0.82, 0.97, 0.24, 0.13},
                {0.96, 0.88, 0.49, 0.84, 0.74, 0.64, 0.98},
                {0.04, 0.39, 0.68, 0.06, 0.32, 0.38, 0.73},
                {0.44, 0.35, 0.89, 0.62, 0.51, 0.69, 0.67},
                {0.17, 0.29, 0.08, 0.83, 0.65, 0.49, 0.47}})
            m_mlp.WeightInit(2, {
                {0.8, 0.63, 0.21, 0.62, 0.58, 0.7, 0.33},
                {0.15, 0.13, 0.79, 0.39, 0.94, 0.35, 0.71},
                {0.53, 0.55, 0.77, 0.04, 0.68, 0.15, 0.36}})
            m_mlp.Train()

            Dim outputMaxtrix As MatrixMLP.Matrix = m_mlp.outputArraySingle
            Dim sOutput = outputMaxtrix.ToStringWithFormat(dec:="0.0")

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As MatrixMLP.Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.01
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub MLP3XORSinus()

            m_mlp.Init(learningRate:=0.1!, weightAdjustment:=0.1!)
            Init3XOR()

            m_mlp.nbIterations = 200
            m_mlp.SetActivationFunction(TActivationFunction.Sinus, gain:=1, center:=0)

            m_mlp.WeightInit(1, {
                {0.41, 0.31, 0.33, 0.61, 0.57, 0.59, 0.73},
                {0.65, 0.67, 0.25, 0.64, 0.65, 0.56, 0.73},
                {0.4, 0.78, 0.24, 0.16, 0.79, 0.39, 0.64},
                {0.89, 0.48, 0.61, 0.83, 0.46, 0.93, 0.75},
                {0.46, 0.13, 0.26, 0.27, 0.14, 0.59, 0.26},
                {0.23, 0.54, 0.45, 0.4, 0.93, 0.9, 0.98}})
            m_mlp.WeightInit(2, {
                {0.93, 0.49, 0.22, 0.1, 0.84, 0.48, 0.33},
                {0.49, 0.39, 0.93, 0.59, 0.22, 0.76, 0.41},
                {0.85, 1.0, 0.74, 0.13, 0.8, 0.9, 0.21}})
            m_mlp.Train()

            Dim outputMaxtrix As MatrixMLP.Matrix = m_mlp.outputArraySingle
            Dim sOutput = outputMaxtrix.ToString() 'WithFormat(dec:="0.0")

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As MatrixMLP.Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString() 'WithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub MLP3XORSigmoid()

            m_mlp.Init(learningRate:=0.6!, weightAdjustment:=0.09!)
            Init3XOR()

            m_mlp.nbIterations = 200
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=1)

            m_mlp.WeightInit(1, {
                {0.71, 0.6, 0.21, 0.85, 0.78, 0.78, 0.2},
                {0.89, 0.61, 0.81, 0.27, 0.77, 0.6, 0.26},
                {0.07, 0.22, 0.42, 0.53, 0.93, 0.14, 0.07},
                {0.46, 0.67, 0.44, 0.58, 0.05, 0.84, 0.61},
                {0.14, 0.76, 0.59, 0.48, 0.04, 0.53, 0.36},
                {0.59, 0.12, 0.6, 0.13, 0.28, 0.22, 0.38}})
            m_mlp.WeightInit(2, {
                {0.14, 0.17, 0.82, 0.88, 0.89, 0.49, 0.74},
                {0.23, 0.31, 0.47, 0.67, 0.25, 0.93, 0.77},
                {0.27, 0.72, 0.94, 0.59, 0.75, 0.85, 0.21}})
            m_mlp.Train()

            Dim outputMaxtrix As MatrixMLP.Matrix = m_mlp.outputArraySingle
            Dim sOutput = outputMaxtrix.ToStringWithFormat(dec:="0.0")

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As MatrixMLP.Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.03
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub MLP3XORDbleThreshold()

            m_mlp.Init(learningRate:=2.0!, weightAdjustment:=0.1!)
            Init3XOR()

            m_mlp.nbIterations = 100
            m_mlp.SetActivationFunction(TActivationFunction.DoubleThreshold, gain:=1, center:=2)

            m_mlp.WeightInit(1, {
                {0.12, 0.3, 0.12, 0.28, 0.12, 0.2, 0.32},
                {0.05, 0.6, 0.24, 0.41, 0.75, 0.57, 0.76},
                {0.51, 0.62, 0.39, 0.7, 0.29, 0.53, 0.28},
                {0.51, 0.43, 0.45, 0.4, 0.36, 0.56, 0.74},
                {0.77, 0.82, 0.95, 0.06, 0.84, 0.71, 0.27},
                {0.66, 0.29, 0.85, 0.32, 0.92, 0.48, 0.29}})
            m_mlp.WeightInit(2, {
                {0.71, 0.99, 0.73, 0.06, 0.95, 0.55, 0.57},
                {0.38, 0.5, 0.37, 0.85, 0.78, 1.0, 0.61},
                {0.87, 0.67, 0.87, 0.76, 0.64, 0.59, 0.27}})
            m_mlp.Train()

            Dim outputMaxtrix As MatrixMLP.Matrix = m_mlp.outputArraySingle
            Dim sOutput = outputMaxtrix.ToString() 'WithFormat(dec:="0.0")

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As MatrixMLP.Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToString() 'WithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub MLP3XORReLU()

            m_mlp.Init(learningRate:=0.1!, weightAdjustment:=0.2!)
            Init3XOR()

            m_mlp.nbIterations = 350
            m_mlp.SetActivationFunction(TActivationFunction.ReLu, gain:=0.5!, center:=0.1!)

            m_mlp.WeightInit(1, {
                {0.13, 0.26, 0.18, 0.05, 0.07, 0.46, 0.96},
                {0.28, 0.31, 0.28, 1.0, 0.47, 0.32, 0.11},
                {0.73, 0.87, 0.17, 0.22, 0.82, 0.97, 0.7},
                {0.27, 0.57, 0.66, 0.53, 0.56, 0.1, 0.64},
                {0.9, 0.92, 0.94, 0.29, 0.86, 0.83, 0.35},
                {0.62, 0.15, 0.53, 0.86, 0.89, 0.16, 0.57}})
            m_mlp.WeightInit(2, {
                {0.47, 0.85, 0.17, 0.55, 0.45, 0.81, 0.49},
                {0.29, 0.03, 0.95, 0.51, 0.46, 0.85, 0.7},
                {0.52, 0.52, 0.51, 0.36, 0.96, 0.65, 0.41}})
            m_mlp.Train()

            Dim outputMaxtrix As MatrixMLP.Matrix = m_mlp.outputArraySingle
            Dim sOutput = outputMaxtrix.ToStringWithFormat(dec:="0.0")

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As MatrixMLP.Matrix = expectedOutput ' Double(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0
            Dim rLoss! = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

    End Class

End Namespace