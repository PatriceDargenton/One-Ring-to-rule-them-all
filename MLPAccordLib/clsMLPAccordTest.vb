﻿
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Imports Microsoft.VisualStudio.TestTools.UnitTesting

Module modMLPAccordTest

    Sub Main()
        Console.WriteLine("Accord MLP with the classical XOR test.")
        AccordMLPTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()
    End Sub

    Public Sub AccordMLPTest()

        Dim mlp As New clsMLPAccord

        mlp.ShowMessage("Accord MLP test")
        mlp.ShowMessage("---------------")

        mlp.inputArray = m_inputArrayXOR
        mlp.targetArray = m_targetArrayXOR

        mlp.nbIterations = 5000 ' Sigmoid: works
        mlp.nbIterations = 5000 ' Hyperbolic tangent: works fine
        'mlp.nbIterations = 20000 ' Stochastic

        mlp.Initialize(learningRate:=0.05!, weightAdjustment:=0.1!)
        mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)

        'mlp.SetActivationFunctionOptimized(
        '    enumActivationFunctionOptimized.Sigmoid, gain:=2, center:=0)
        mlp.SetActivationFunctionOptimized(
            enumActivationFunctionOptimized.HyperbolicTangent, gain:=2, center:=0)

        mlp.Randomize()

        mlp.PrintWeights()

        WaitForKeyToStart()

        mlp.printOutput_ = True
        mlp.TrainVector() ' Works fine
        'mlp.Train() ' Works fine
        'mlp.Train(enumLearningMode.Systematic) ' Works fine
        'mlp.Train(enumLearningMode.SemiStochastic) ' Works
        'mlp.Train(enumLearningMode.Stochastic) ' Works

        mlp.ShowMessage("Accord MLP test: Done.")

    End Sub

End Module

Namespace AccordMLP

    <TestClass()> _
    Public Class clsMLPAccordTest

        Private m_mlp As New clsMLPAccord

        'Private m_mlp As New clsMLPClassic ' 18/18 success
        'Private m_mlp As New NetworkOOP.MultilayerPerceptron ' 12/18 success

        <TestInitialize()>
        Public Sub Init()
        End Sub

        Private Sub InitXOR()
            m_mlp.Initialize(learningRate:=0.01!, weightAdjustment:=0)
            m_mlp.inputArray = m_inputArrayXOR
            m_mlp.targetArray = m_targetArrayXOR
        End Sub

        Private Sub Init2XOR()
            m_mlp.Initialize(learningRate:=0.01!, weightAdjustment:=0)
            m_mlp.inputArray = m_inputArray2XOR
            m_mlp.targetArray = m_targetArray2XOR
        End Sub

        Private Sub Init3XOR()
            m_mlp.Initialize(learningRate:=0.01!, weightAdjustment:=0)
            m_mlp.inputArray = m_inputArray3XOR
            m_mlp.targetArray = m_targetArray3XOR
        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XORSemiStochastic()

            TestMLP1XORSemiStochastic(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XORStochastic()

            TestMLP1XORStochastic(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XOR4Layers()

            TestMLP1XOR4Layers(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XOR4LayersVect()

            TestMLP1XOR4Layers(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XOR5Layers()

            TestMLP1XOR5Layers(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XOR5LayersVect()

            TestMLP1XOR5Layers(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XORSigmoidStdr()

            TestMLP1XORSigmoid(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XORSigmoidVect()

            TestMLP1XORSigmoid(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XORHTangent()

            TestMLP1XORHTangent(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP1XORHTangentVect()

            TestMLP1XORHTangent(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP2XORSigmoid()

            TestMLP2XORSigmoid(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP2XORSigmoidVect()

            TestMLP2XORSigmoid(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP2XORHTangent()

            TestMLP2XORHTangent(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP2XORHTangentVect()

            TestMLP2XORHTangent(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP2XORHTangentVectAlgo2()

            m_mlp.PRBPLAlgo = True

            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.05!)
            Init2XOR()

            m_mlp.nbIterations = 700
            m_mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=1.9!, center:=0)

            m_mlp.InitializeWeights(1, {
                {-0.08, 0.24, 0.67, 0.65, 0.21},
                {-0.68, 0.38, 0.41, -0.45, -0.05},
                {0.52, 0.7, 0.11, 0.31, 0.34},
                {0.35, 0.58, 0.58, -0.33, 0.29}})
            m_mlp.InitializeWeights(2, {
                {0.76, 0.13, -0.57, 0.2, -0.16},
                {-0.78, 0.18, -0.3, 0.18, -0.47}})

            m_mlp.TrainVector()
            'm_mlp.Train() Does not work

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
        Public Sub AccordMLP3XORSigmoid()

            TestMLP3XORSigmoid(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP3XORSigmoidVect()

            TestMLP3XORSigmoid(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP3XORHTangent()

            TestMLP3XORHTangent(m_mlp)

        End Sub

        <TestMethod()>
        Public Sub AccordMLP3XORHTangentVect()

            TestMLP3XORHTangent(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

    End Class

End Namespace