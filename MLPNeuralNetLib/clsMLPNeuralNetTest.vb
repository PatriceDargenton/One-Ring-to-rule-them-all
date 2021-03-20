
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Imports Perceptron.clsMLPNeuralNet ' TrainingAlgorithmType
' BC40025: Type of this member is not CLS-compliant:
'Imports NeuralNetworkNET.SupervisedLearning.Algorithms ' TrainingAlgorithmType

Imports Microsoft.VisualStudio.TestTools.UnitTesting

Module modMLPNeuralNetTest

    Sub Main()
        Console.WriteLine("NeuralNet.NET MLP with the classical XOR test.")
        NeuralNetMLPXorTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()
    End Sub

    Public Sub NeuralNetMLPXorTest(Optional nbXor% = 1)

        Dim mlp As New clsMLPNeuralNet

        mlp.ShowMessage("NeuralNet.NET MLP test")
        mlp.ShowMessage("----------------------")

        mlp.inputArray = m_inputArrayXOR
        mlp.targetArray = m_targetArrayXOR

        mlp.Initialize(learningRate:=0)

        mlp.printOutput_ = True
        mlp.printOutputMatrix = False

        If nbXor = 1 Then
            mlp.InitializeStruct(m_neuronCountXOR271, addBiasColumn:=True)
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
            mlp.InitializeStruct(m_neuronCount3XOR673, addBiasColumn:=True)
        End If

        mlp.trainingAlgorithm = enumTrainingAlgorithm.RMSProp

        'mlp.nbIterationsBatch = mlp.minBatchSize ' Can be 1 using Sigmoid and RMSProp
        'mlp.nbIterations = 15000 ' Sigmoid: works
        'mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

        mlp.nbIterations = 4000 ' Hyperbolic tangent: works fine
        If nbXor = 3 Then mlp.nbIterations = 15000
        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

        mlp.Randomize()

        mlp.PrintWeights()

        WaitForKeyToStart()

        mlp.Train(learningMode:=enumLearningMode.VectorialBatch) ' Works fine

        mlp.ShowMessage("NeuralNet.NET MLP test: Done.")

        If nbXor > 1 Then Exit Sub

        WaitForKeyToContinue("Press a key to print MLP weights")
        mlp.PrintWeights()

    End Sub

End Module

Namespace NeuralNetMLP

    <TestClass()>
    Public Class clsMLPNeuralNetTest

        Private m_mlp As New clsMLPNeuralNet

        <TestMethod()>
        Public Sub NNMLP1XORSigmoidBatch1RMSProp() ' 270 msec

            InitXOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.0!)
            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.RMSProp
            m_mlp.nbIterationsBatch = 1

            m_mlp.nbIterations = 2100
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {-1.14, -0.66, 0.0},
                {-1.15, 0.74, 0.0}})
            m_mlp.InitializeWeights(2, {
               {0.89, -1.37, 0.0}})

            m_mlp.Train(learningMode:=enumLearningMode.Vectorial)
            'm_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub NNMLP1XORSigmoidBatch200RMSProp() ' 90 msec

            InitXOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.0!)
            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.RMSProp
            m_mlp.nbIterationsBatch = 200

            m_mlp.nbIterations = 2600
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {0.17, -0.19, 0},
                {0.9, -0.13, 0}})
            m_mlp.InitializeWeights(2, {
                {0.8, -1.39, 0}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.04
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub NNMLP1XORTanhBatch10RMSProp()

            InitXOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.0!)
            m_mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.RMSProp
            m_mlp.nbIterationsBatch = 10

            'm_mlp.nbIterations = 5000 ' 3000 batch
            'm_mlp.nbIterations = 4000 ' 1000 batch
            'm_mlp.nbIterations = 3000 ' 500 batch
            'm_mlp.nbIterations = 3000 ' 100 batch
            m_mlp.nbIterations = 4000 ' 10 batch
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {0.06, -0.11, 0},
                {0.64, 0.68, 0},
                {0.67, 0.31, 0}})
            m_mlp.InitializeWeights(2, {
                {-0.17, -0.95, 0.47, 0}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.04
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub NNMLP1XORTanhBatch100RMSProp()

            InitXOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.0!)
            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.RMSProp
            m_mlp.nbIterationsBatch = 100

            m_mlp.nbIterations = 5000 ' 100 batch
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {-0.13, 0.18, 0},
                {-0.77, 0.82, 0}})
            m_mlp.InitializeWeights(2, {
                {1.35, 1.37, 0}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.04
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub NNMLP1XOR4LayersTanhBatch100RMSProp() ' 150 msec

            InitXOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.0!)
            m_mlp.InitializeStruct(m_neuronCountXOR4Layers, addBiasColumn:=True)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.RMSProp
            m_mlp.nbIterationsBatch = 100

            m_mlp.nbIterations = 600
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {1.01, -0.05, 0},
                {0.33, -0.18, 0}})
            m_mlp.InitializeWeights(2, {
                {-1.11, -0.51, 0},
                {0.34, 0.19, 0}})
            m_mlp.InitializeWeights(3, {
                {0.28, -1.02, 0}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub NNMLP1XOR5LayersTanhBatch100RMSProp() ' 30 msec

            InitXOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.0!)
            m_mlp.InitializeStruct(m_neuronCountXOR5Layers, addBiasColumn:=True)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.RMSProp
            m_mlp.nbIterationsBatch = 100

            m_mlp.nbIterations = 500
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {0.9, -1.17, 0},
                {-0.15, -0.6, 0}})
            m_mlp.InitializeWeights(2, {
                {-0.8, -0.32, 0},
                {-0.02, 0.08, 0}})
            m_mlp.InitializeWeights(3, {
                {0.95, -0.19, 0},
                {-0.08, -0.24, 0}})
            m_mlp.InitializeWeights(4, {
                {-1.25, -0.12, 0}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.03
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub NNMLP2XORSigmoidBatch100RMSProp() ' 400 msec

            Init2XOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.0!)
            m_mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.RMSProp ' AdaMax ?
            m_mlp.nbIterationsBatch = 100

            m_mlp.nbIterations = 10000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {-0.01, -0.09, 0.81, 0.09, 0},
                {0.05, -0.65, -0.54, -0.26, 0},
                {-0.21, 0.83, -0.63, -0.01, 0},
                {0.81, 0.21, -0.14, -0.09, 0}})
            m_mlp.InitializeWeights(2, {
                {-0.1, 0.57, 0.77, 0.78, 0},
                {-0.75, 0.34, 0.87, 0.99, 0}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.04
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub NNMLP2XORSigmoidBatch200RMSProp3() ' 460 msec

            Init2XOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.0!)
            m_mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.RMSProp
            m_mlp.nbIterationsBatch = 200

            m_mlp.nbIterations = 11000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {-0.2, 0.25, 0.68, 0.0, 0},
                {-0.09, -0.07, -0.53, 0.02, 0},
                {0.85, -0.15, 0.27, 0.04, 0},
                {0.12, 0.53, -0.4, 0.11, 0}})
            m_mlp.InitializeWeights(2, {
                {0.25, 0.11, 0.12, 0.62, 0},
                {0.69, -0.11, 0.56, 0.03, 0}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.04
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub NNMLP2XORTanhBatch100RMSProp()

            Init2XOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.0!)
            m_mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.RMSProp
            m_mlp.nbIterationsBatch = 100

            m_mlp.nbIterations = 5000
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {0.01, -0.57, 0.45, -0.1, 0},
                {0.5, -0.65, -0.76, 0.4, 0},
                {-0.47, 0.31, 0.84, 0.49, 0},
                {-0.05, 0.05, 0.37, -0.01, 0}})
            m_mlp.InitializeWeights(2, {
                {-0.34, 0.88, -0.67, -0.47, 0},
                {0.16, 0.26, 0.96, -0.64, 0}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.03
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub NNMLP2XORTanhBatch200RMSProp() ' 110 msec

            Init2XOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.0!)
            m_mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.RMSProp
            m_mlp.nbIterationsBatch = 200

            m_mlp.nbIterations = 2800
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {0.6, -0.57, -0.65, 0.47, 0},
                {0.46, 0.53, -0.37, -0.12, 0},
                {0.65, -0.06, 0.53, -0.62, 0},
                {-0.12, 0.4, 0.34, -0.56, 0}})
            m_mlp.InitializeWeights(2, {
                {0.62, 0.64, -0.82, 0.99, 0},
                {0.73, -0.32, 0.03, 0.41, 0}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.03
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub NNMLP2XOR4LayersTanhBatch100RMSProp() ' 150 msec

            Init2XOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.0!)
            m_mlp.InitializeStruct(m_neuronCount2XOR4Layers, addBiasColumn:=True)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.RMSProp
            m_mlp.nbIterationsBatch = 100

            m_mlp.nbIterations = 2400
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {0.66, 0.16, 0.55, 0.44, 0},
                {-0.62, 0.11, 0.1, 0.72, 0},
                {0.53, -0.42, -0.37, 0.63, 0},
                {0.79, -0.54, 0.19, 0.2, 0}})
            m_mlp.InitializeWeights(2, {
                {-0.76, 0.41, 0.28, -0.78, 0},
                {0.32, -0.86, -0.16, -0.36, 0},
                {0.21, -0.66, 0.77, 0.38, 0},
                {-0.72, 0.6, -0.01, 0.25, 0}})
            m_mlp.InitializeWeights(3, {
                {-0.95, 0.68, -0.52, 0.44, 0},
                {-0.64, 0.78, -0.26, 0.77, 0}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub NNMLP2XOR5LayersTanhBatch100RMSProp() ' 350 msec

            Init2XOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.0!)
            m_mlp.InitializeStruct(m_neuronCount2XOR5Layers, addBiasColumn:=True)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.RMSProp
            m_mlp.nbIterationsBatch = 100

            m_mlp.nbIterations = 2500
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {0.32, 0.31, 0.7, 0.3, 0},
                {0.79, 0.34, 0.39, -0.78, 0},
                {-0.01, -0.54, 0.69, -0.5, 0},
                {-0.41, 0.59, -0.72, -0.13, 0}})
            m_mlp.InitializeWeights(2, {
                {0.42, -0.61, -0.59, 0.17, 0},
                {-0.28, -0.37, -0.19, -0.78, 0},
                {0.56, -0.49, 0.29, -0.86, 0},
                {-0.16, 0.03, -0.72, 0.8, 0}})
            m_mlp.InitializeWeights(3, {
                {-0.34, 0.32, 0.81, 0.51, 0},
                {0.0, -0.05, 0.43, -0.73, 0},
                {0.28, -0.27, 0.71, 0.85, 0},
                {-0.79, -0.77, 0.01, 0.02, 0}})
            m_mlp.InitializeWeights(4, {
                {0.39, -0.64, -0.91, 0.24, 0},
                {1.0, 0.41, 0.59, 0.94, 0}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArray2XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub NNMLP3XORSigmoidBatch100AdaMax() ' 390 msec

            Init3XOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.0!)
            m_mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.AdaMax
            m_mlp.nbIterationsBatch = 100

            m_mlp.nbIterations = 6000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {0.24, -0.03, 0.41, 0.63, -0.14, -0.02, 0},
                {-0.7, 0.62, 0.29, 0.47, 0.04, -0.61, 0},
                {-0.18, 0.16, -0.15, 0.47, 0.67, 0.6, 0},
                {-0.3, 0.57, -0.12, 0.33, 0.2, -0.17, 0},
                {-0.58, -0.08, -0.37, -0.34, 0.65, -0.61, 0},
                {-0.47, -0.3, -0.47, -0.15, 0.17, 0.3, 0}})
            m_mlp.InitializeWeights(2, {
                {0.12, 0.67, -0.31, -0.74, 0.74, 0.56, 0},
                {-0.81, 0.22, -0.34, 0.0, -0.5, 0.23, 0},
                {-0.02, -0.12, -0.46, 0.14, -0.75, 0.15, 0}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.03
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub NNMLP3XORTanhBatch100AdaMax() ' 160 msec

            Init3XOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.0!)
            m_mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.AdaMax
            m_mlp.nbIterationsBatch = 100

            m_mlp.nbIterations = 1400
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {-0.34, -0.12, -0.42, 0.69, -0.56, -0.08, 0},
                {-0.18, 0.12, 0.56, 0.67, -0.33, 0.1, 0},
                {-0.38, 0.47, -0.27, 0.4, -0.33, -0.26, 0},
                {0.71, 0.61, -0.03, -0.3, 0.64, -0.61, 0},
                {0.66, -0.61, 0.15, 0.37, 0.4, -0.35, 0},
                {0.62, -0.02, 0.42, -0.44, -0.4, -0.45, 0}})
            m_mlp.InitializeWeights(2, {
                {0.37, 0.35, 0.67, -0.21, -0.43, 0.27, 0},
                {-0.1, 0.7, -0.29, -0.32, 0.44, 0.01, 0},
                {-0.8, 0.73, 0.06, -0.68, -0.78, 0.53, 0}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub NNMLP3XORTanhBatch500AdaMax() ' 250 msec

            Init3XOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.0!)
            m_mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.AdaMax
            m_mlp.nbIterationsBatch = 500

            m_mlp.nbIterations = 2500
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {-0.42, 0.22, -0.61, 0.17, 0.03, 0.01, 0},
                {-0.09, 0.37, 0.39, -0.64, 0.36, -0.02, 0},
                {-0.44, -0.63, 0.21, 0.44, 0.09, -0.51, 0},
                {-0.67, -0.04, 0.69, -0.58, 0.27, -0.53, 0},
                {0.29, -0.06, -0.62, -0.1, -0.71, 0.24, 0},
                {-0.2, -0.45, -0.23, -0.34, -0.43, -0.4, 0}})
            m_mlp.InitializeWeights(2, {
                {0.08, -0.38, 0.56, 0.11, 0.67, -0.1, 0},
                {-0.31, 0.58, -0.29, 0.78, 0.18, -0.51, 0},
                {0.81, 0.52, 0.05, -0.72, 0.58, 0.64, 0}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub NNMLP3XORTanhBatch200AdaMax() ' 470 msec

            Init3XOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.0!)
            m_mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.AdaMax
            m_mlp.nbIterationsBatch = 200

            m_mlp.nbIterations = 4000
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {-0.26, -0.15, -0.49, -0.49, 0.33, 0.59, 0},
                {-0.55, -0.04, -0.41, 0.6, 0.36, -0.17, 0},
                {0.17, -0.25, 0.07, 0.06, 0.26, 0.11, 0},
                {0.63, -0.64, -0.1, 0.09, 0.42, -0.4, 0},
                {-0.33, 0.54, 0.0, 0.36, -0.56, 0.44, 0},
                {0.13, -0.3, -0.1, -0.61, -0.31, 0.08, 0}})
            m_mlp.InitializeWeights(2, {
                {-0.43, -0.75, 0.64, 0.23, -0.23, -0.34, 0},
                {-0.18, -0.68, 0.38, -0.03, 0.64, -0.01, 0},
                {0.32, -0.78, -0.55, 0.64, -0.54, 0.55, 0}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub NNMLP3XORTanhBatch100Adam() ' 210 msec

            Init3XOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.0!)
            m_mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.Adam
            m_mlp.nbIterationsBatch = 100

            m_mlp.nbIterations = 3200
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {-0.32, -0.35, 0.32, -0.03, 0.19, -0.24, 0},
                {0.66, -0.28, -0.25, -0.27, 0.08, 0.57, 0},
                {-0.4, 0.41, -0.11, -0.55, 0.14, 0.49, 0},
                {0.52, 0.16, -0.18, -0.49, 0.52, -0.61, 0},
                {0.14, -0.68, 0.32, 0.05, 0.25, -0.45, 0},
                {-0.29, -0.02, 0.02, -0.62, 0.6, -0.21, 0}})
            m_mlp.InitializeWeights(2, {
                {-0.11, 0.68, -0.19, 0.66, -0.43, 0.6, 0},
                {0.16, -0.74, -0.71, 0.43, -0.19, -0.48, 0},
                {0.32, 0.24, 0.04, 0.35, -0.25, -0.73, 0}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub NNMLPIrisFlowerAnalog()

            ' 93.3% prediction, 96.7% learning with 4000 iterations in 850 msec.

            InitIrisFlowerAnalog4Layers(m_mlp)
            m_mlp.Initialize(learningRate:=0)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.AdaDelta
            m_mlp.nbIterationsBatch = 80

            m_mlp.nbIterations = 4000
            m_mlp.minimalSuccessTreshold = 0.2
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {0.08, 0.49, -0.27, 0.36, 0.0},
                {0.22, -0.34, 0.57, -0.26, 0.0},
                {-0.64, -0.55, 0.62, -0.26, 0.0},
                {-0.46, -0.47, 0.19, -0.48, 0.0},
                {0.6, -0.21, -0.14, 0.09, 0.0},
                {0.39, 0.31, 0.53, -0.04, 0.0},
                {-0.2, 0.41, -0.19, -0.47, 0.0},
                {0.38, -0.22, -0.27, 0.42, 0.0},
                {-0.56, 0.4, 0.36, 0.02, 0.0}})
            m_mlp.InitializeWeights(2, {
                {-0.53, -0.52, 0.27, -0.23, -0.56, 0.52, 0.55, -0.08, -0.15, 0.0},
                {0.53, 0.05, -0.2, 0.09, 0.44, -0.55, -0.45, -0.19, -0.02, 0.0},
                {0.3, 0.41, -0.09, -0.23, 0.25, -0.46, -0.54, -0.36, -0.58, 0.0},
                {-0.26, 0.42, 0.57, 0.01, 0.1, 0.34, 0.33, 0.4, 0.55, 0.0},
                {-0.23, 0.29, -0.21, -0.47, 0.53, 0.44, -0.31, 0.22, 0.37, 0.0},
                {0.27, 0.33, -0.43, 0.01, 0.25, 0.08, 0.23, 0.09, 0.01, 0.0},
                {0.5, -0.47, 0.49, -0.49, -0.55, -0.15, 0.07, 0.14, -0.28, 0.0},
                {-0.07, -0.24, 0.51, 0.51, 0.17, 0.16, 0.4, 0.21, -0.05, 0.0},
                {-0.04, -0.01, -0.06, 0.28, -0.19, 0.46, 0.39, -0.29, 0.47, 0.0}})
            m_mlp.InitializeWeights(3, {
                {0.32, -0.52, 0.4, 0.19, -0.05, 0.51, 0.66, -0.41, -0.58, 0.0}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedSuccess# = 0.967
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.08
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            m_mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
                m_targetArrayIrisFlowerAnalogTest, nbOutputs:=1)
            Dim expectedSuccessPrediction# = 0.933
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub NNMLPIrisFlowerLogicalTanhBatch120RMSProp()

            ' 97.8% prediction, 98.6% learning with 12000 iterations in 2.1 sec.

            InitIrisFlowerLogical4Layers(m_mlp)
            m_mlp.Initialize(learningRate:=0)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.RMSProp
            m_mlp.nbIterationsBatch = 120

            m_mlp.nbIterations = 12000
            m_mlp.minimalSuccessTreshold = 0.3
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {0.33, -0.06, 0.19, -0.44, 0.0},
                {-0.17, 0.53, -0.42, 0.28, 0.0},
                {0.31, -0.44, -0.13, -0.32, 0.0},
                {0.44, -0.48, -0.06, -0.2, 0.0},
                {0.31, -0.3, -0.55, 0.16, 0.0},
                {0.44, -0.23, -0.17, 0.5, 0.0},
                {-0.23, 0.2, -0.54, -0.41, 0.0},
                {0.13, 0.12, 0.53, -0.15, 0.0},
                {0.51, 0.23, -0.48, -0.34, 0.0},
                {-0.39, 0.19, -0.48, -0.49, 0.0},
                {0.42, 0.39, 0.4, -0.04, 0.0},
                {-0.12, -0.11, 0.01, -0.42, 0.0},
                {-0.26, -0.2, -0.52, -0.05, 0.0},
                {-0.46, 0.48, -0.42, 0.24, 0.0},
                {0.55, -0.49, 0.52, -0.28, 0.0},
                {0.45, 0.31, 0.48, -0.17, 0.0}})
            m_mlp.InitializeWeights(2, {
                {-0.15, 0.02, -0.47, 0.24, -0.41, 0.4, -0.45, 0.34, -0.39, -0.05, 0.36, 0.09, -0.34, 0.3, -0.28, -0.23, 0.0},
                {0.37, 0.3, 0.11, 0.0, -0.16, 0.11, 0.4, 0.12, 0.15, 0.44, -0.2, 0.18, -0.02, 0.01, -0.44, -0.04, 0.0},
                {-0.46, -0.47, -0.27, -0.38, 0.13, -0.46, 0.25, 0.03, 0.48, 0.34, 0.28, 0.47, -0.21, -0.25, -0.37, 0.31, 0.0},
                {0.1, -0.32, -0.44, -0.06, 0.41, 0.17, 0.07, -0.05, -0.06, 0.38, -0.41, -0.41, -0.27, 0.34, 0.32, 0.14, 0.0},
                {-0.48, 0.13, 0.49, 0.49, 0.4, 0.08, -0.25, 0.26, -0.46, 0.46, 0.26, 0.0, 0.02, 0.4, 0.23, -0.26, 0.0},
                {-0.28, -0.42, -0.33, -0.23, 0.1, -0.09, -0.47, -0.29, -0.38, 0.31, 0.4, -0.23, -0.05, -0.31, -0.47, -0.18, 0.0},
                {0.0, -0.08, -0.14, -0.42, 0.32, 0.19, 0.24, 0.08, -0.49, -0.42, 0.49, 0.14, 0.45, -0.48, -0.42, 0.23, 0.0},
                {0.29, 0.14, -0.47, -0.07, -0.17, -0.46, 0.16, -0.42, 0.5, 0.22, -0.27, -0.12, -0.4, 0.12, -0.26, 0.31, 0.0}})
            m_mlp.InitializeWeights(3, {
                {0.05, 0.01, -0.32, 0.0, 0.17, 0.7, 0.04, -0.39, 0.0},
                {0.51, -0.37, 0.45, -0.2, -0.54, 0.05, -0.14, 0.57, 0.0},
                {-0.38, 0.16, -0.65, 0.71, 0.49, 0.25, -0.11, 0.73, 0.0}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedSuccess# = 0.986
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.03
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            m_mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
                m_targetArrayIrisFlowerLogicalTest, nbOutputs:=3)
            Dim expectedSuccessPrediction# = 0.978
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub NNMLPIrisFlowerLogicalSigmoidBatch100RMSProp()

            ' 95.6% prediction, 96.1% learning with 5000 iterations in 1.1 sec.

            InitIrisFlowerLogical4Layers(m_mlp)
            m_mlp.Initialize(learningRate:=0)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.RMSProp
            m_mlp.nbIterationsBatch = 100

            m_mlp.nbIterations = 5000
            m_mlp.minimalSuccessTreshold = 0.3
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {0.52, 0.07, 0.25, -0.31, 0.0},
                {0.42, -0.35, 0.53, 0.3, 0.0},
                {0.09, -0.35, 0.07, -0.06, 0.0},
                {0.51, 0.04, -0.44, -0.27, 0.0},
                {0.23, 0.36, 0.15, 0.13, 0.0},
                {0.24, 0.52, 0.09, 0.28, 0.0},
                {-0.48, 0.04, -0.42, -0.51, 0.0},
                {0.3, 0.5, 0.28, -0.31, 0.0},
                {0.03, -0.49, 0.52, 0.36, 0.0},
                {0.31, -0.01, 0.39, -0.01, 0.0},
                {-0.15, 0.02, 0.12, -0.18, 0.0},
                {-0.52, -0.04, 0.37, 0.16, 0.0},
                {-0.2, 0.28, -0.35, 0.01, 0.0},
                {0.3, 0.09, 0.13, -0.31, 0.0},
                {0.52, -0.22, 0.34, -0.14, 0.0},
                {0.12, 0.08, -0.46, -0.48, 0.0}})
            m_mlp.InitializeWeights(2, {
                {-0.27, -0.41, -0.02, 0.37, -0.04, 0.03, -0.3, -0.46, -0.4, -0.19, 0.08, -0.09, -0.11, 0.2, -0.13, -0.05, 0.0},
                {-0.32, -0.19, -0.46, 0.49, 0.4, 0.17, 0.46, 0.35, 0.41, 0.39, 0.47, 0.42, -0.03, 0.02, 0.0, 0.3, 0.0},
                {0.49, -0.03, 0.01, -0.15, 0.04, 0.18, 0.27, 0.24, 0.27, -0.04, -0.33, 0.13, 0.16, -0.48, -0.14, 0.27, 0.0},
                {0.47, -0.27, -0.24, 0.39, -0.33, -0.35, -0.19, -0.06, -0.47, -0.14, -0.38, 0.49, -0.3, -0.05, 0.48, 0.1, 0.0},
                {0.37, -0.06, -0.46, -0.22, -0.44, -0.34, 0.34, -0.15, 0.2, -0.2, -0.05, -0.28, -0.45, 0.2, 0.2, -0.07, 0.0},
                {-0.26, 0.28, 0.02, 0.27, -0.1, 0.41, -0.2, -0.06, -0.24, 0.43, -0.25, 0.18, 0.49, 0.18, 0.04, -0.43, 0.0},
                {-0.16, 0.34, -0.06, -0.09, -0.41, -0.32, -0.31, -0.27, 0.3, 0.27, -0.44, 0.1, 0.36, -0.23, -0.07, 0.47, 0.0},
                {0.06, 0.39, -0.27, 0.14, -0.09, 0.3, 0.33, -0.03, -0.31, -0.35, -0.14, -0.01, -0.1, -0.05, 0.47, -0.06, 0.0}})
            m_mlp.InitializeWeights(3, {
                {0.19, 0.18, 0.27, -0.42, 0.2, 0.56, 0.38, -0.73, 0.0},
                {-0.6, -0.53, 0.58, 0.42, 0.56, -0.58, -0.14, -0.44, 0.0},
                {-0.65, 0.24, -0.29, -0.05, -0.32, -0.1, -0.38, 0.51, 0.0}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedSuccess# = 0.961
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.08
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            m_mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
                m_targetArrayIrisFlowerLogicalTest, nbOutputs:=3)
            Dim expectedSuccessPrediction# = 0.956
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub NNMLPSunspotSigmoidBatch100()

            ' 100% prediction, 85.7% learning with 100 iterations in 8 msec.

            InitSunspot(m_mlp)
            m_mlp.nbLinesToLearn = 49
            m_mlp.InitializeStruct({7, 14, 1}, addBiasColumn:=True)
            m_mlp.Initialize(learningRate:=0)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.AdaDelta
            m_mlp.nbIterationsBatch = 100

            m_mlp.nbIterations = 100
            m_mlp.minimalSuccessTreshold = 0.3
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {-0.13, -0.15, -0.28, 0.37, -0.45, 0.07, -0.29, 0.0},
                {-0.09, -0.2, -0.42, 0.43, 0.01, 0.08, 0.45, 0.0},
                {0.21, -0.19, -0.36, 0.12, -0.47, -0.46, -0.49, 0.0},
                {-0.38, -0.48, -0.31, 0.34, -0.02, -0.23, -0.37, 0.0},
                {0.11, 0.02, 0.34, -0.1, -0.45, -0.5, -0.48, 0.0},
                {-0.24, 0.31, 0.22, 0.04, -0.43, 0.05, 0.08, 0.0},
                {-0.32, 0.36, 0.31, -0.35, -0.11, -0.27, -0.53, 0.0},
                {0.1, -0.1, -0.12, 0.13, 0.34, -0.4, -0.22, 0.0},
                {-0.42, 0.41, -0.41, 0.33, -0.19, -0.37, -0.35, 0.0},
                {0.49, -0.07, 0.35, 0.41, 0.45, 0.27, 0.24, 0.0},
                {0.28, 0.41, -0.18, -0.27, 0.13, -0.13, -0.46, 0.0},
                {0.27, -0.32, -0.36, -0.53, -0.35, 0.22, 0.25, 0.0},
                {0.09, 0.22, 0.02, -0.09, -0.34, -0.43, 0.35, 0.0},
                {0.17, -0.08, -0.39, -0.23, -0.11, -0.08, -0.38, 0.0}})
            m_mlp.InitializeWeights(2, {
                {-0.16, 0.08, 0.61, -0.02, -0.1, 0.28, -0.21, -0.09, -0.16, -0.27, -0.47, -0.12, -0.06, -0.26, 0.0}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedSuccess# = 0.857
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.2
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            'm_mlp.TestAllSamples(m_mlp.inputArrayTest, m_mlp.targetArrayTest, nbOutputs:=m_mlp.nbLinesToPredict)
            m_mlp.TestAllSamples(m_mlp.inputArrayTest, m_mlp.targetArrayTest, nbOutputs:=1)
            Dim expectedSuccessPrediction# = 1
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub NNMLPSunspotTanhBatch100()

            ' 100% prediction, 98% learning with 60 iterations in 5 msec.

            InitSunspot(m_mlp)
            m_mlp.nbLinesToLearn = 49
            m_mlp.InitializeStruct({7, 14, 1}, addBiasColumn:=True)
            m_mlp.Initialize(learningRate:=0)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.AdaDelta
            m_mlp.nbIterationsBatch = 60

            m_mlp.nbIterations = 60
            m_mlp.minimalSuccessTreshold = 0.3
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {0.27, 0.33, -0.29, 0.24, 0.31, 0.38, -0.42, 0.0},
                {-0.12, 0.44, -0.04, -0.01, -0.28, -0.14, 0.05, 0.0},
                {0.1, -0.03, -0.22, 0.35, -0.22, 0.37, 0.47, 0.0},
                {0.19, 0.41, 0.29, -0.06, 0.28, -0.41, -0.45, 0.0},
                {-0.2, -0.34, 0.52, 0.42, 0.25, 0.17, 0.24, 0.0},
                {0.49, -0.45, -0.14, 0.49, 0.37, -0.35, 0.31, 0.0},
                {-0.17, -0.13, -0.13, -0.1, -0.3, -0.46, 0.53, 0.0},
                {-0.09, 0.19, -0.1, 0.24, 0.44, 0.46, -0.18, 0.0},
                {-0.25, 0.36, 0.1, -0.23, -0.25, -0.25, 0.24, 0.0},
                {-0.03, -0.2, -0.14, 0.39, 0.12, -0.15, -0.19, 0.0},
                {0.3, 0.22, 0.04, 0.43, -0.36, -0.49, -0.1, 0.0},
                {-0.32, -0.19, 0.0, -0.45, 0.07, -0.18, 0.41, 0.0},
                {0.44, -0.11, -0.33, 0.28, 0.34, 0.51, 0.46, 0.0},
                {-0.44, -0.13, 0.06, 0.02, -0.26, 0.06, -0.3, 0.0}})
            m_mlp.InitializeWeights(2, {
                {0.37, -0.2, 0.59, 0.43, 0.1, -0.25, 0.36, 0.42, -0.13, -0.6, -0.32, -0.3, -0.58, 0.5, 0.0}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedSuccess# = 0.98
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.11
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            'm_mlp.TestAllSamples(m_mlp.inputArrayTest, m_mlp.targetArrayTest, nbOutputs:=m_mlp.nbLinesToPredict)
            m_mlp.TestAllSamples(m_mlp.inputArrayTest, m_mlp.targetArrayTest, nbOutputs:=1)
            Dim expectedSuccessPrediction# = 1
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub NNMLPIrisFlowerLogicalTanh()

            ' 97.8% prediction, 97.2% learning with 4400 iterations in 1.1 sec.

            TestMLPIrisFlowerLogicalTanh(m_mlp, nbIterations:=4400,
                expectedSuccess:=0.972, expectedLoss:=0.06)

        End Sub

        <TestMethod()>
        Public Sub NNMLPIrisFlowerLogicalSigmoid()

            ' 95.6% prediction, 96.7% learning with 7000 iterations in 1.7 sec.

            TestMLPIrisFlowerLogicalSigmoid(m_mlp, nbIterations:=7000,
                expectedSuccess:=0.967, expectedLoss:=0.07,
                expectedSuccessPrediction:=0.956)

        End Sub

        <TestMethod()>
        Public Sub NNMLPIrisFlowerAnalogTanh2()

            ' 96.7% prediction, 98.3% learning with 12000 iterations in 2.7 sec.

            TestMLPIrisFlowerAnalogTanh2(m_mlp, nbIterations:=12000,
                expectedSuccess:=0.983, expectedLoss:=0.03)

        End Sub

        <TestMethod()>
        Public Sub NNMLPIrisFlowerAnalogSigmoid()

            ' 90% prediction, 95% learning with 5000 iterations in 1.1 sec.

            TestMLPIrisFlowerAnalogSigmoid(m_mlp, nbIterations:=5000,
                expectedSuccess:=0.95, expectedLoss:=0.1,
                expectedSuccessPrediction:=0.9)

        End Sub

        <TestMethod()>
        Public Sub NNMLPSunspotSigmoid()

            ' 90.0% prediction, 73.5% learning with 4000 iterations in 300 msec.

            TestMLPSunspotSigmoid(m_mlp, nbIterations:=4000, expectedSuccess:=0.735)

        End Sub

        <TestMethod()>
        Public Sub NNMLPSunspotTanh()

            ' 90.0% prediction, 93.8% learning with 3000 iterations in 230 msec.

            TestMLPSunspotTanh(m_mlp, nbIterations:=3000, expectedSuccess:=0.938,
                expectedSuccessPrediction:=0.9)

        End Sub

    End Class

End Namespace