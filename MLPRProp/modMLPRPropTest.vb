
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPClassic ' enumLearningMode

Imports Microsoft.VisualStudio.TestTools.UnitTesting

#Const TestConsoleDemo = 0 ' 0: Off, 1: On

Module modMLPRPROPTTest

    Sub Main()

        'Dim mlp As New clsMLPRProp
        'Dim trainAcc# = 0, testAcc# = 0
        'mlp.ConsoleDemo(trainAcc, testAcc, multiThread:=False)
        'Console.ReadLine()
        'Exit Sub

        Console.WriteLine("MultiLayerPerceptron with the Resilient Propagation XOR test.")
        RPropMLPXorTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()

    End Sub

    Public Sub RPropMLPXorTest(Optional nbXor% = 1)

        Dim mlp As New clsMLPRProp

        mlp.ShowMessage("Resilient Propagation MLP Xor test")
        mlp.ShowMessage("----------------------------------")

        mlp.Initialize(learningRate:=0!)

        Dim nbIterations% = 3000

        mlp.printOutput_ = True
        mlp.printOutputMatrix = False
        mlp.nbIterations = nbIterations

        If nbXor = 1 Then
            mlp.inputArray = m_inputArrayXOR
            mlp.targetArray = m_targetArrayXOR
            'mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
            mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR4Layers2331, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR5Layers23331, addBiasColumn:=True)
            mlp.printOutputMatrix = True
        ElseIf nbXor = 2 Then
            mlp.inputArray = m_inputArray2XOR
            mlp.targetArray = m_targetArray2XOR
            mlp.InitializeStruct(m_neuronCount2XOR452, addBiasColumn:=True)
        ElseIf nbXor = 3 Then
            mlp.inputArray = m_inputArray3XOR
            mlp.targetArray = m_targetArray3XOR
            'mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
            mlp.InitializeStruct(m_neuronCount3XOR673, addBiasColumn:=True)
        End If

        'mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=0.2!)
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=0.2!)

        mlp.Randomize()
        'mlp.Randomize(minValue:=-10, maxValue:=10)
        mlp.PrintWeights()

        WaitForKeyToStart()

        'mlp.Train()
        mlp.TrainVectorBatch()

        mlp.ShowMessage("Resilient Propagation MLP Xor test: Done.")

        If nbXor > 1 Then Exit Sub

        WaitForKeyToContinue("Press a key to print MLP weights")
        mlp.PrintWeights()

    End Sub

End Module

Namespace RPropMLP

    <TestClass()>
    Public Class MultiLayerPerceptronTest

        Private m_mlp As New clsMLPRProp

        ' ToDo
        ' Weights are quite the same as MLP Classic, but not exactly:
        'Private m_mlp As New NetworkOOP.MultilayerPerceptron ' 15 success, 15 fails
        'Private m_mlp As New clsMLPAccord ' 14 success, 16 fails
        'Private m_mlp As New clsMLPEncog  ' 7 success, 23 fails
        'Private m_mlp As New clsMLPTensorFlow ' 1 success, 25 fails

        ' Weights are not stored in the same way:
        'Private m_mlp As New MatrixMLP.MultiLayerPerceptron ' 24/24 fails
        'Private m_mlp As New VectorizedMatrixMLP.clsVectorizedMatrixMLP ' 24/24 fails
        'Private m_mlp As New clsMLPTensor ' 24/24 fails

        <TestInitialize()>
        Public Sub Init()
        End Sub

        Private Sub InitXOR()
            m_mlp.inputArray = m_inputArrayXOR
            m_mlp.targetArray = m_targetArrayXOR
            m_mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
        End Sub

        Private Sub Init2XOR()
            m_mlp.inputArray = m_inputArray2XOR
            m_mlp.targetArray = m_targetArray2XOR
            m_mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)
        End Sub

        Private Sub Init3XOR()
            m_mlp.inputArray = m_inputArray3XOR
            m_mlp.targetArray = m_targetArray3XOR
            m_mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
        End Sub

        Private Sub InitIrisFlowerLogicalSoftMax(mlp As clsMLPGeneric)
            mlp.classificationObjective = True
            mlp.inputArray = m_inputArrayIrisFlowerTrain
            mlp.targetArray = m_targetArrayIrisFlowerLogicalTrain
            mlp.InitializeStruct(m_neuronCountIrisFlowerLogical4_16_83, addBiasColumn:=True)
        End Sub

        <TestMethod()>
        Public Sub RPropMLP1XORSigmoid()

            TestMLP1XORSigmoid(m_mlp, nbIterations:=200)

        End Sub

        <TestMethod()>
        Public Sub RPropMLP1XORSigmoidRProp()

            TestMLP1XORSigmoidRProp(m_mlp, nbIterations:=350,
                learningMode:=enumLearningMode.VectorialBatch)

        End Sub

        '<TestMethod()>
        Public Sub RPropMLP1XORTanhStdr()

            TestMLP1XORTanh(m_mlp, nbIterations:=1000)

        End Sub

        <TestMethod()>
        Public Sub RPropMLP1XORTanhRProp()

            TestMLP1XORTanhRProp(m_mlp, nbIterations:=500,
                learningMode:=enumLearningMode.VectorialBatch)

        End Sub


        '<TestMethod()>
        Public Sub RPropMLP1XORTanH()

            InitXOR()
            m_mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

            m_mlp.nbIterations = 400
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=0.9)

            m_mlp.InitializeWeights(1, {
                {0.07, 0.79, 0.94},
                {0.33, 0.33, 0.93}})
            m_mlp.InitializeWeights(2, {
                {0.63, 0.79, 0.69}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.04
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub RPropMLP2XORSigmoid()

            TestMLP2XORSigmoid(m_mlp, nbIterations:=300)

        End Sub

        <TestMethod()>
        Public Sub RPropMLP2XORTanh()

            TestMLP2XORTanh(m_mlp, nbIterations:=600, gain:=0.5)

        End Sub

        '<TestMethod()>
        Public Sub RPropMLP3XORSigmoidStdr()

            TestMLP3XORSigmoid(m_mlp, nbIterations:=10000)

        End Sub

        <TestMethod()>
        Public Sub RPropMLP3XORSigmoid()

            Init3XOR()
            m_mlp.Initialize(learningRate:=0)

            m_mlp.nbIterations = 1500
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=0.5!)

            m_mlp.InitializeWeights(1, {
                {-4.78, -7.05, 1.65, 1.47, 8.23, 4.28, -0.35},
                {0.86, -0.42, -4.26, 5.31, 2.0, -4.39, -7.38},
                {-6.19, 7.67, 0.05, -1.73, 0.61, -9.9, -2.59},
                {3.19, -3.1, -0.05, -5.94, 4.74, 5.08, -4.83},
                {-7.94, -3.01, -0.83, 6.7, 2.12, -8.24, -7.19},
                {-8.5, 2.3, 7.78, 2.87, 8.46, -7.95, -8.3}})
            m_mlp.InitializeWeights(2, {
                {1.67, -7.73, -3.78, 9.9, -8.07, -0.52, -5.58},
                {8.23, -9.11, 0.2, 5.01, 0.35, -9.57, 2.03},
                {6.05, -8.3, -2.59, -6.52, 9.2, -5.52, -1.19}})
            m_mlp.Train()

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub RPropMLP3XORTanhStdr()

            TestMLP3XORTanh(m_mlp, nbIterations:=8000, gain:=0.1)

        End Sub

        <TestMethod()>
        Public Sub RPropMLP3XORTanh()

            Init3XOR()
            m_mlp.InitializeStruct(m_neuronCount3XOR673, addBiasColumn:=True)

            m_mlp.Initialize(learningRate:=0)

            m_mlp.nbIterations = 4000
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=0.1!)

            m_mlp.InitializeWeights(1, {
                {7.92, 7.42, 5.91, -7.47, -8.62, 2.21, 6.11},
                {4.89, -7.41, -6.78, 5.16, 7.24, -2.04, -0.77},
                {4.08, 3.4, -1.62, 6.51, 7.11, 0.36, 3.15},
                {9.35, 4.58, -4.86, -0.76, -5.03, -3.15, -6.04},
                {7.47, -1.99, 4.5, 9.02, -8.14, 5.59, -1.82},
                {5.56, 7.46, -7.2, -6.04, -5.75, -1.96, 7.38},
                {-5.01, 8.25, -4.68, 4.37, 4.79, -7.37, 7.03}})
            m_mlp.InitializeWeights(2, {
                {-2.94, -0.38, 4.42, 5.67, 5.32, 7.96, 8.57, -7.16},
                {0.77, 3.29, 6.41, -4.64, 2.15, 7.42, 4.58, -1.27},
                {6.14, 5.38, 2.37, -8.95, 8.52, 5.93, -4.42, 2.55}})

            'm_mlp.TrainVector() ' Does not work fine
            m_mlp.Train()

            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub RPropMLPIrisFlowerAnalogTanh()

            ' 90.0% prediction, 93.3% learning with 200 iterations in 330 msec.
            TestMLPIrisFlowerAnalogTanh2(m_mlp,
                nbIterations:=200, expectedSuccess:=0.933, expectedSuccessPrediction:=0.9,
                expectedLoss:=0.1)

        End Sub

        <TestMethod()>
        Public Sub RPropMLPIrisFlowerAnalogSigmoidStdr()

            ' 93.3% prediction, 94.2% learning with 200 iterations in 220 msec.
            TestMLPIrisFlowerAnalogSigmoid(m_mlp, nbIterations:=200,
                expectedSuccess:=0.942, expectedLoss:=0.072, expectedSuccessPrediction:=0.933)

        End Sub

        <TestMethod()>
        Public Sub RPropMLPIrisFlowerLogicalSigmoidStdr()

            ' 95.6% prediction, 97.8% learning with 1500 iterations in 2.4 sec.
            m_mlp.classificationObjective = True
            TestMLPIrisFlowerLogicalSigmoid(m_mlp,
                nbIterations:=1500, expectedSuccess:=0.978, expectedSuccessPrediction:=0.956)

        End Sub

        <TestMethod()>
        Public Sub RPropMLPIrisFlowerLogicalSigmoidStdrMultiThread()

            ' 95.6% prediction, 97.8% learning with 1500 iterations in 2.4 sec. Single thread
            ' 95.6% prediction, 97.8% learning with 1500 iterations in 1.4 sec. 2 threads

            ' Intel Core i7-10810U 1.10GHz 1.61 GHz
            ' 95.6% prediction, 97.8% learning with 1500 iterations in 2 sec. Single thread
            ' 95.6% prediction, 97.8% learning with 1500 iterations in 640 msec. 10 threads

            m_mlp.multiThread = True
            m_mlp.classificationObjective = True
            TestMLPIrisFlowerLogicalSigmoid(m_mlp,
                nbIterations:=1500, expectedSuccess:=0.978, expectedSuccessPrediction:=0.956)

        End Sub

        <TestMethod()>
        Public Sub RPropMLPIrisFlowerLogicalTanhStdr()

            ' 95.6% prediction, 97.2% learning with 200 iterations in 300 msec.
            m_mlp.classificationObjective = True
            TestMLPIrisFlowerLogicalTanh(m_mlp,
                nbIterations:=200, expectedSuccess:=0.972, expectedSuccessPrediction:=0.956,
                expectedLoss:=0.04)

        End Sub

        <TestMethod()>
        Public Sub RPropMLPIrisFlowerLogicalTanhStdrMultiThread()

            ' 95.6% prediction, 97.2% learning with 200 iterations in 300 msec. Single thread
            ' 95.6% prediction, 97.2% learning with 200 iterations in 177 msec. 2 threads

            ' Intel Core i7-10810U 1.10GHz 1.61 GHz
            ' 95.6% prediction, 97.2% learning with 200 iterations in 262 msec. Single thread
            ' 95.6% prediction, 97.2% learning with 200 iterations in  89 msec. 10 threads

            m_mlp.classificationObjective = True
            m_mlp.multiThread = True
            TestMLPIrisFlowerLogicalTanh(m_mlp,
                nbIterations:=200, expectedSuccess:=0.972, expectedSuccessPrediction:=0.956,
                expectedLoss:=0.04)

        End Sub

        <TestMethod()>
        Public Sub RPropMLPIrisFlowerLogicalSoftMax()

            ' 95.6% prediction, 93.9% learning with 600 iterations in 525 msec.

            m_mlp.multiThread = False
            InitIrisFlowerLogicalSoftMax(m_mlp)
            m_mlp.InitializeStruct(m_neuronCountIrisFlowerLogical4_20_3, addBiasColumn:=True)

            m_mlp.nbIterations = 600
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {-3.99, 6.84, -9.19, -1.42, 8.45},
                {-2.55, 4.36, 4.14, -5.43, 2.22},
                {6.91, 2.69, 4.84, -0.02, 5.63},
                {7.2, -9.38, -9.28, -9.3, 1.19},
                {-3.62, 0.18, 7.89, -6.15, 4.64},
                {-7.27, 4.64, 5.44, 9.0, 6.21},
                {7.88, 1.84, -8.15, 3.0, 5.56},
                {0.04, 3.54, -3.09, 9.8, -7.44},
                {-3.29, -0.32, -0.73, -7.08, -0.59},
                {4.32, 5.7, 6.54, 8.93, 2.62},
                {8.45, -4.62, -4.0, 9.94, 0.42},
                {5.83, 8.94, 6.96, 3.94, 5.72},
                {2.81, 8.92, 5.13, -1.64, 4.34},
                {-4.93, 0.84, -8.17, 4.43, -4.42},
                {-6.34, 3.71, -9.07, 8.14, -5.52},
                {6.7, -9.09, 4.97, 4.44, -9.68},
                {-2.97, 8.1, 6.51, -3.61, 7.76},
                {2.51, -4.16, -8.09, -7.41, 9.73},
                {1.1, 6.59, 2.97, -5.93, -0.25},
                {-2.22, 4.55, -9.09, -1.42, -5.66}})

            m_mlp.InitializeWeights(2, {
                {-6.52, 3.87, -7.89, 3.34, -1.04, -5.26, -5.55, -2.14, 5.46, 3.72, 4.92, -6.03, -7.49, 3.62, -1.3, 4.71, -7.59, -1.25, 0.6, -8.17, 9.23},
                {-1.07, 9.25, 4.7, 4.48, -2.93, -9.25, 6.85, -1.61, 6.69, -7.85, -9.99, -3.61, 0.1, 6.84, 3.16, 5.99, -6.83, 7.43, 3.02, -1.95, 7.97},
                {-3.56, -1.13, -5.19, -2.87, 4.07, 9.34, 5.38, 5.04, -4.63, -4.2, 0.3, -3.25, -4.89, 4.55, 4.62, -2.58, 8.86, -8.11, -6.01, -2.4, 9.47}})

            m_mlp.minimalSuccessTreshold = 0.3
            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedSuccess# = 0.939
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.059
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            If m_mlp.successPC = 1 AndAlso m_mlp.minimalSuccessTreshold <= 0.05! Then
                Dim expectedOutput = m_targetArrayIrisFlowerLogical
                Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
                Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
                Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
                Assert.AreEqual(sExpectedOutput, sOutput)
            End If

            m_mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
                m_targetArrayIrisFlowerLogicalTest, nbOutputs:=3)
            Dim expectedSuccessPrediction# = 0.956
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub RPropMLPIrisFlowerLogicalSoftMaxMultiThread()

            ' 95.6% prediction, 93.9% learning with 600 iterations in 525 msec. Single thread
            ' 95.6% prediction, 93.9% learning with 600 iterations in 323 msec. 2 threads

            ' Intel Core i7-10810U 1.10GHz 1.61 GHz
            ' 95.6% prediction, 93.9% learning with 600 iterations in 468 msec. Single thread
            ' 95.6% prediction, 93.9% learning with 600 iterations in 174 msec. 10 threads

            m_mlp.multiThread = True
            InitIrisFlowerLogicalSoftMax(m_mlp)
            m_mlp.InitializeStruct(m_neuronCountIrisFlowerLogical4_20_3, addBiasColumn:=True)

            m_mlp.nbIterations = 600
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent)

            m_mlp.InitializeWeights(1, {
                {-3.99, 6.84, -9.19, -1.42, 8.45},
                {-2.55, 4.36, 4.14, -5.43, 2.22},
                {6.91, 2.69, 4.84, -0.02, 5.63},
                {7.2, -9.38, -9.28, -9.3, 1.19},
                {-3.62, 0.18, 7.89, -6.15, 4.64},
                {-7.27, 4.64, 5.44, 9.0, 6.21},
                {7.88, 1.84, -8.15, 3.0, 5.56},
                {0.04, 3.54, -3.09, 9.8, -7.44},
                {-3.29, -0.32, -0.73, -7.08, -0.59},
                {4.32, 5.7, 6.54, 8.93, 2.62},
                {8.45, -4.62, -4.0, 9.94, 0.42},
                {5.83, 8.94, 6.96, 3.94, 5.72},
                {2.81, 8.92, 5.13, -1.64, 4.34},
                {-4.93, 0.84, -8.17, 4.43, -4.42},
                {-6.34, 3.71, -9.07, 8.14, -5.52},
                {6.7, -9.09, 4.97, 4.44, -9.68},
                {-2.97, 8.1, 6.51, -3.61, 7.76},
                {2.51, -4.16, -8.09, -7.41, 9.73},
                {1.1, 6.59, 2.97, -5.93, -0.25},
                {-2.22, 4.55, -9.09, -1.42, -5.66}})

            m_mlp.InitializeWeights(2, {
                {-6.52, 3.87, -7.89, 3.34, -1.04, -5.26, -5.55, -2.14, 5.46, 3.72, 4.92, -6.03, -7.49, 3.62, -1.3, 4.71, -7.59, -1.25, 0.6, -8.17, 9.23},
                {-1.07, 9.25, 4.7, 4.48, -2.93, -9.25, 6.85, -1.61, 6.69, -7.85, -9.99, -3.61, 0.1, 6.84, 3.16, 5.99, -6.83, 7.43, 3.02, -1.95, 7.97},
                {-3.56, -1.13, -5.19, -2.87, 4.07, 9.34, 5.38, 5.04, -4.63, -4.2, 0.3, -3.25, -4.89, 4.55, 4.62, -2.58, 8.86, -8.11, -6.01, -2.4, 9.47}})

            m_mlp.minimalSuccessTreshold = 0.3
            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedSuccess# = 0.939
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.059
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            If m_mlp.successPC = 1 AndAlso m_mlp.minimalSuccessTreshold <= 0.05! Then
                Dim expectedOutput = m_targetArrayIrisFlowerLogical
                Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
                Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
                Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
                Assert.AreEqual(sExpectedOutput, sOutput)
            End If

            m_mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
                m_targetArrayIrisFlowerLogicalTest, nbOutputs:=3)
            Dim expectedSuccessPrediction# = 0.956
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub RPropMLPIrisFlowerLogicalSoftMax2()

            ' 100% prediction, 97.2% learning with 100 iterations in 100 msec.

            m_mlp.multiThread = False
            InitIrisFlowerLogicalSoftMax(m_mlp)
            m_mlp.InitializeStruct({4, 22, 3}, addBiasColumn:=True)

            m_mlp.nbIterations = 100
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {-8.22, -5.6, -1.29, 7.33, 4.14},
                {0.61, 7.58, 9.5, -7.43, -2.59},
                {7.48, -5.22, -7.7, -9.29, 3.33},
                {0.68, -5.42, -2.19, 7.87, -3.02},
                {-9.39, 5.63, 7.45, 3.89, 5.98},
                {-4.69, 5.23, -3.2, -5.92, -2.4},
                {-8.21, -8.64, 6.84, 4.64, 9.07},
                {-2.72, -1.95, -8.42, -1.45, 6.8},
                {-0.19, 4.8, -8.04, 1.74, -4.02},
                {-1.94, 4.73, 8.08, 0.48, 6.1},
                {-0.89, 8.42, 8.91, 1.87, -5.76},
                {-3.85, -3.05, 4.83, -8.64, -1.17},
                {5.38, 0.78, 5.43, 4.97, -4.38},
                {6.12, -2.06, -2.34, -8.36, -3.96},
                {-7.36, -7.0, 9.26, -8.93, 7.17},
                {-4.19, 3.67, -4.29, -2.09, -2.08},
                {0.57, 7.15, 6.32, -2.03, 8.49},
                {-6.62, -7.56, -5.02, 0.4, 2.92},
                {-9.66, 3.22, -9.77, 9.72, -8.58},
                {9.03, 9.37, -3.01, -3.88, -0.14},
                {-9.88, -2.93, 6.44, -5.57, 3.47},
                {-3.89, 9.16, 7.84, 4.7, 8.43}})

            m_mlp.InitializeWeights(2, {
                {2.47, -8.77, -3.08, 3.44, 8.26, 8.23, 4.45, -2.55, 6.48, -7.76, 3.68, -7.04, 7.26, -1.28, -4.29, -0.58, -7.23, 9.54, 9.65, 8.13, -3.56, -3.32, 9.59},
                {8.05, -2.2, -6.5, -9.29, 1.89, 4.5, 2.39, -5.78, -5.4, 0.28, 1.97, -9.55, 9.1, -3.7, -3.21, -8.54, -6.8, -5.42, 1.92, 0.52, -6.11, 6.18, 7.16},
                {-0.19, -2.28, 8.73, -5.95, -6.66, 9.62, 8.19, 6.56, 1.99, -4.2, -8.36, -1.13, -4.36, 4.76, 7.51, -7.44, 2.95, -5.91, 8.02, -0.93, 2.68, -4.71, -1.73}})

            m_mlp.minimalSuccessTreshold = 0.3

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedSuccess# = 0.972
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.021
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            If m_mlp.successPC = 1 AndAlso m_mlp.minimalSuccessTreshold <= 0.05! Then
                Dim expectedOutput = m_targetArrayIrisFlowerLogical
                Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
                Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
                Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
                Assert.AreEqual(sExpectedOutput, sOutput)
            End If

            m_mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
                m_targetArrayIrisFlowerLogicalTest, nbOutputs:=3)
            Dim expectedSuccessPrediction# = 1
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub RPropMLPIrisFlowerLogicalSoftMax3()

            ' 100% prediction, 99.4% learning with 50 iterations in 50 msec.

            m_mlp.multiThread = False
            InitIrisFlowerLogicalSoftMax(m_mlp)
            m_mlp.InitializeStruct(m_neuronCountIrisFlowerLogical4_20_3, addBiasColumn:=True)

            m_mlp.nbIterations = 50
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {-5.04, -7.2, -5.84, 9.15, -3.58},
                {-4.09, -1.88, 3.56, -8.21, 1.43},
                {6.77, 4.19, 8.31, 7.19, 0.44},
                {-1.47, 1.02, 2.89, -6.55, 0.34},
                {-5.13, -5.45, 6.41, 2.19, 0.39},
                {0.12, -0.59, 6.69, -2.66, 6.37},
                {-4.63, -9.91, -8.66, -0.31, 3.17},
                {3.73, 8.88, 6.39, 2.14, 0.06},
                {-4.96, -3.2, 1.23, 3.95, 8.89},
                {-1.33, 1.13, -9.06, 1.11, 8.57},
                {-7.09, -6.04, -2.53, 5.1, 0.55},
                {-9.59, -3.6, 1.98, -1.25, 6.29},
                {6.5, 1.43, -3.78, -4.59, -3.94},
                {6.68, 2.84, -1.38, -5.98, 6.72},
                {-0.35, 4.64, -9.25, 3.39, -4.71},
                {8.07, 3.32, -7.55, 3.3, -8.27},
                {8.99, -1.53, -4.42, -1.23, 3.46},
                {-8.59, 2.62, -3.76, 9.14, 2.76},
                {-2.67, -3.1, -2.36, 5.85, 3.56},
                {3.62, -9.43, -4.18, -2.1, -7.79}})

            m_mlp.InitializeWeights(2, {
                {5.82, -7.49, 6.92, 4.4, -1.09, -1.72, -6.79, 4.09, -0.19, 2.48, -2.91, -6.06, 8.68, -2.97, 7.3, -1.97, -4.15, 7.46, 1.96, -5.35, -5.94},
                {-3.4, -0.52, 1.26, -0.61, -7.25, -3.0, -5.1, 9.83, 1.68, 7.49, -2.5, 4.56, 1.09, -4.1, 6.48, 1.55, 1.18, 9.85, -4.82, 8.2, 8.53},
                {6.43, -3.34, -4.33, -6.61, -1.78, -9.39, 8.55, -4.47, -2.23, -6.88, 3.86, -6.75, 8.15, -0.78, 3.03, -4.34, -4.99, -3.84, -8.72, 8.31, 4.26}})

            m_mlp.minimalSuccessTreshold = 0.3

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedSuccess# = 0.994
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.007
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            If m_mlp.successPC = 1 AndAlso m_mlp.minimalSuccessTreshold <= 0.05! Then
                Dim expectedOutput = m_targetArrayIrisFlowerLogical
                Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
                Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
                Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
                Assert.AreEqual(sExpectedOutput, sOutput)
            End If

            m_mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
                m_targetArrayIrisFlowerLogicalTest, nbOutputs:=3)
            Dim expectedSuccessPrediction# = 1
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub RPropMLPIrisFlowerLogicalSoftMax4()

            ' 100% prediction, 99.4% learning with 50 iterations in 100 msec.

            m_mlp.multiThread = False
            InitIrisFlowerLogicalSoftMax(m_mlp)
            m_mlp.InitializeStruct({4, 18, 12, 3}, addBiasColumn:=True)

            m_mlp.nbIterations = 50
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {4.68, -1.7, -4.74, -6.61, 5.31},
                {1.62, -9.95, -4.84, -2.95, 3.94},
                {1.47, -6.52, 5.88, -9.51, -1.16},
                {7.62, -4.66, -6.32, -9.64, -4.79},
                {8.09, -2.45, 1.29, 8.4, 3.56},
                {8.67, -4.26, 2.52, -6.96, -6.96},
                {-1.44, -8.38, -8.62, -2.27, -9.88},
                {7.71, -5.12, -2.48, -5.82, -2.72},
                {-1.2, 6.92, -4.87, 3.89, -9.29},
                {0.86, -5.06, -4.38, 9.81, -4.66},
                {-9.45, 0.73, 8.22, -6.3, 2.17},
                {-2.87, 7.01, -3.14, -0.17, 6.64},
                {-4.12, -2.47, -7.88, -6.0, -4.62},
                {-0.15, -7.9, -1.86, -9.63, 1.13},
                {2.74, 7.82, 9.5, 3.08, 6.41},
                {-8.83, -7.58, 7.4, 7.69, -7.3},
                {3.73, -9.88, 2.72, 7.7, -7.51},
                {7.83, -6.6, 7.68, 5.57, 3.0}})

            m_mlp.InitializeWeights(2, {
                {-9.3, 8.02, 7.69, -2.46, -8.6, -8.73, 4.8, -8.87, -1.49, 0.86, -1.25, 6.8, -4.75, -1.33, 2.61, -7.27, 1.23, -4.85, -2.7},
                {1.0, -5.3, 9.61, -0.82, -2.87, -7.09, -4.24, 4.81, -5.58, -8.49, -2.45, -3.54, -5.58, 2.58, -2.63, 0.44, 4.72, -9.87, 1.96},
                {1.68, 5.14, -3.63, -8.71, -1.11, -3.17, 3.95, 6.93, 4.87, -5.95, -4.91, 9.76, -3.4, 8.25, 0.38, -5.43, -1.71, -8.91, -1.16},
                {0.56, -5.37, 5.63, -3.54, 0.38, 9.62, -9.03, -5.6, -5.67, -5.78, 7.87, 8.23, 7.88, -7.4, 9.27, 3.46, 2.16, -5.37, -6.58},
                {0.72, -7.66, 3.18, -4.02, 0.89, 0.76, 9.33, -8.26, -9.06, -1.79, 4.04, -1.99, 9.08, -0.65, -4.12, -0.44, -2.67, 6.05, -1.32},
                {5.99, -8.33, -2.08, -7.57, -0.38, 2.71, 7.0, -4.19, -2.79, 7.16, -2.67, -5.22, 8.22, 9.94, -5.13, 0.36, -3.5, 7.38, 8.64},
                {-5.26, 5.7, 1.05, 7.88, -9.98, 6.19, 0.29, 6.22, 8.79, -1.12, 2.01, 3.04, 1.95, 7.41, -6.52, -1.36, -8.25, -7.2, 9.91},
                {-6.44, 3.28, 3.9, -5.05, 2.12, -5.42, 3.62, -6.57, 5.82, -1.93, 4.21, 8.99, 9.38, 2.18, -1.31, 1.31, 2.98, -5.06, -6.21},
                {-2.1, -3.76, 9.33, 6.49, 8.21, 6.93, 5.2, -5.88, 5.38, -2.63, 4.74, 1.3, -6.87, -2.44, -3.42, 3.82, -4.63, 0.84, 0.74},
                {8.92, 3.29, -3.59, 2.77, 4.47, -1.85, -5.42, -0.11, 2.63, -9.14, -6.74, -3.9, 0.5, -6.3, 7.96, 4.91, 3.68, -5.77, 6.79},
                {5.69, -1.98, 6.93, -9.53, 8.24, 6.06, -8.93, 6.77, 1.08, 1.43, 2.8, 8.37, -6.68, -9.53, -7.76, -3.98, 0.5, -8.51, 3.47},
                {-5.14, -1.66, 3.63, -2.96, -7.43, -6.74, 2.12, -8.73, -5.75, -2.4, -4.83, 2.65, 2.91, -0.42, -1.42, 5.15, -7.27, -8.0, 2.83}})

            m_mlp.InitializeWeights(3, {
                {-1.83, 6.71, 3.4, 1.38, 3.5, 8.46, 9.83, -7.51, 9.94, -4.37, -1.74, 7.68, -2.53},
                {3.41, -9.8, 9.37, -1.56, -7.94, -9.03, 4.36, 7.2, -3.88, 4.79, 6.83, -0.83, -4.08},
                {8.78, 9.89, -1.22, 4.74, -4.68, 9.51, -5.98, 7.67, 3.32, 6.76, 1.46, 6.96, -9.87}})

            m_mlp.minimalSuccessTreshold = 0.3

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedSuccess# = 0.994
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.015
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            If m_mlp.successPC = 1 AndAlso m_mlp.minimalSuccessTreshold <= 0.05! Then
                Dim expectedOutput = m_targetArrayIrisFlowerLogical
                Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
                Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
                Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
                Assert.AreEqual(sExpectedOutput, sOutput)
            End If

            m_mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
                m_targetArrayIrisFlowerLogicalTest, nbOutputs:=3)
            Dim expectedSuccessPrediction# = 1
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub RPropMLPIrisFlowerLogicalSigmoidSoftMax()

            ' 97.8% prediction, 98.9% learning with 1100 iterations in 1.1 sec.

            m_mlp.multiThread = False
            InitIrisFlowerLogicalSoftMax(m_mlp)
            m_mlp.InitializeStruct(m_neuronCountIrisFlowerLogical4_20_3, addBiasColumn:=True)

            m_mlp.nbIterations = 1100
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {3.46, -6.28, 8.59, -2.36, 0.54},
                {-8.86, 9.13, -8.75, -8.18, 6.51},
                {-1.84, -5.13, 3.95, 3.9, 4.72},
                {2.39, -7.08, 8.13, 7.45, 5.81},
                {-0.09, -6.58, 9.93, -4.4, -1.07},
                {8.86, 1.38, -3.48, -0.12, 2.27},
                {-6.86, -9.23, -7.52, -8.27, -7.09},
                {-5.11, 8.72, -4.67, 2.74, 8.36},
                {6.96, -5.33, -6.97, 3.38, 1.78},
                {1.4, -9.1, -2.59, 5.32, -9.59},
                {2.99, 6.71, 7.23, 0.83, 6.9},
                {0.05, -6.2, 2.99, 8.71, 1.69},
                {-0.24, 2.6, 1.38, -0.45, 3.37},
                {-2.61, -7.61, 2.21, 0.99, -0.16},
                {3.68, 7.59, -4.61, 9.09, 8.85},
                {-4.76, -9.61, -3.45, 3.82, 7.53},
                {7.95, -6.03, 1.2, -0.54, 9.28},
                {-3.57, -6.47, 1.65, -5.17, 2.86},
                {-8.91, -4.28, -3.38, -8.95, -1.4},
                {-5.64, 3.29, 3.48, -9.99, -5.62}})

            m_mlp.InitializeWeights(2, {
                {-0.99, -1.31, 6.42, -4.52, -3.26, 5.4, 1.32, 8.15, 1.97, 1.66, -0.34, 7.25, 9.17, -8.82, 3.73, -4.21, -8.6, -8.09, 0.27, -3.06, -6.14},
                {0.74, -2.62, 8.13, -1.25, -2.04, 0.97, -5.66, 0.49, 4.5, 1.95, -3.09, -3.47, -0.56, -1.48, -0.74, -2.45, -4.28, -7.27, -6.12, 5.11, -4.61},
                {9.68, 3.18, -6.79, -6.16, -3.45, -4.56, -2.68, -7.19, -7.05, 1.38, -6.8, 3.07, -9.48, 8.27, -8.69, 8.3, 6.74, 8.78, -5.57, -3.02, -2.34}})

            m_mlp.minimalSuccessTreshold = 0.3
            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedSuccess# = 0.989
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.017
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            If m_mlp.successPC = 1 AndAlso m_mlp.minimalSuccessTreshold <= 0.05! Then
                Dim expectedOutput = m_targetArrayIrisFlowerLogical
                Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
                Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
                Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
                Assert.AreEqual(sExpectedOutput, sOutput)
            End If

            m_mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
                m_targetArrayIrisFlowerLogicalTest, nbOutputs:=3)
            Dim expectedSuccessPrediction# = 0.978
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub RPropMLPIrisFlowerLogicalSigmoid()

            ' 97.8% prediction, 98.6% learning with 1600 iterations in 1.7 sec.

            m_mlp.multiThread = False
            InitIrisFlowerLogical4Layers(m_mlp)
            m_mlp.InitializeStruct({4, 21, 3}, addBiasColumn:=True)

            m_mlp.nbIterations = 1600
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=0.3!)

            m_mlp.InitializeWeights(1, {
                {0.23, 0.48, 0.02, -0.1, -0.34},
                {0.03, 0.12, -0.22, 0.46, 0.33},
                {0.3, -0.1, -0.1, -0.31, 0.02},
                {-0.48, -0.22, 0.35, 0.36, -0.26},
                {-0.26, 0.1, 0.3, 0.07, 0.18},
                {-0.4, -0.26, 0.07, 0.07, 0.2},
                {0.18, 0.07, 0.05, -0.07, 0.23},
                {-0.3, -0.19, -0.49, 0.11, -0.18},
                {0.16, -0.3, 0.47, -0.22, -0.31},
                {-0.23, -0.39, 0.44, -0.42, -0.33},
                {-0.08, -0.01, -0.23, -0.12, 0.22},
                {0.41, 0.09, 0.35, 0.03, -0.47},
                {-0.11, -0.25, -0.4, 0.23, 0.42},
                {0.35, 0.25, 0.29, 0.32, 0.47},
                {0.34, -0.14, 0.13, 0.42, 0.44},
                {0.1, 0.02, -0.21, 0.33, -0.37},
                {0.14, 0.03, -0.24, -0.42, -0.22},
                {-0.44, -0.49, -0.21, 0.22, -0.34},
                {0.47, 0.07, 0.43, -0.09, -0.39},
                {-0.25, -0.14, -0.16, 0.02, -0.18},
                {-0.39, -0.39, -0.17, -0.44, 0.36}})

            m_mlp.InitializeWeights(2, {
                {-0.15, -0.06, -0.37, -0.11, -0.19, -0.14, 0.2, 0.4, 0.25, 0.26, 0.07, -0.03, 0.34, 0.06, 0.19, 0.28, 0.03, -0.25, 0.18, -0.11, -0.44, 0.49},
                {0.03, -0.01, -0.23, 0.42, -0.35, -0.19, 0.31, 0.02, -0.11, -0.3, -0.3, -0.28, 0.44, 0.12, 0.38, 0.41, -0.09, 0.3, -0.23, 0.17, -0.33, 0.16},
                {0.03, 0.28, 0.03, 0.28, 0.32, 0.04, 0.08, 0.35, 0.15, 0.17, 0.3, -0.15, 0.41, 0.14, -0.39, -0.11, -0.29, -0.11, -0.41, -0.27, -0.13, -0.12}})

            m_mlp.minimalSuccessTreshold = 0.3

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedSuccess# = 0.986
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.015
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            If m_mlp.successPC = 1 AndAlso m_mlp.minimalSuccessTreshold <= 0.05! Then
                Dim expectedOutput = m_targetArrayIrisFlowerLogical
                Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
                Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
                Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
                Assert.AreEqual(sExpectedOutput, sOutput)
            End If

            m_mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
                m_targetArrayIrisFlowerLogicalTest, nbOutputs:=3)
            Dim expectedSuccessPrediction# = 0.978
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub RPropMLPSunspotSigmoid()

            ' 90.0% prediction, 70.8% learning with 200 iterations in 45 msec.

            TestMLPSunspotSigmoid(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        '<TestMethod()>
        'Public Sub RPropMLPSunspotSigmoidMultiThread()

        '    ' 90.0% prediction, 70.8% learning with 200 iterations in 45 msec.
        '    ' With 49 samples, works only using ThreadCount = 1: it does not work yet!
        '    m_mlp.multiThread = True
        '    TestMLPSunspotSigmoid(m_mlp, learningMode:=enumLearningMode.Vectorial)

        'End Sub

        <TestMethod()>
        Public Sub RPropMLPSunspotTanh()

            ' 100.0% prediction, 70.8% learning with 200 iterations in 45 msec.

            TestMLPSunspotTanh(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

        <TestMethod()>
        Public Sub RPropMLPSunspotTanhStdrMultiThread()

            ' 100.0% prediction, 70.8% learning with 200 iterations in 45 msec.

            m_mlp.multiThread = True ' Ok with 48 samples, but not with 49 samples!
            TestMLPSunspotTanh(m_mlp, learningMode:=enumLearningMode.Vectorial)

        End Sub

#If TestConsoleDemo Then

        Const expectedAccuracy# = 0.95

        <TestMethod()>
        Public Sub RPropMLPConsoleDemo()
            Dim trainAcc# = 0, testAcc# = 0
            m_mlp.ConsoleDemo(trainAcc, testAcc, multiThread:=False)
            Assert.AreEqual(True, trainAcc >= expectedAccuracy)
            Assert.AreEqual(True, testAcc >= expectedAccuracy)
        End Sub

        <TestMethod()>
        Public Sub RPropMLPConsoleDemoMultiThread()
            Dim trainAcc# = 0, testAcc# = 0
            m_mlp.ConsoleDemo(trainAcc, testAcc, multiThread:=True)
            Assert.AreEqual(True, trainAcc >= expectedAccuracy)
            Assert.AreEqual(True, testAcc >= expectedAccuracy)
        End Sub

        <TestMethod()>
        Public Sub RPropMLPConsoleDemoTrainAndTest()
            Dim trainAcc# = 0, testAcc# = 0
            m_mlp.ConsoleDemo(trainAcc, testAcc, multiThread:=False, trainAndTest:=True)
            Assert.AreEqual(True, trainAcc >= expectedAccuracy)
            Assert.AreEqual(True, testAcc >= expectedAccuracy)
        End Sub

        <TestMethod()>
        Public Sub RPropMLPConsoleDemoTrainAndTestMultiThread()
            Dim trainAcc# = 0, testAcc# = 0
            m_mlp.ConsoleDemo(trainAcc, testAcc, multiThread:=True, trainAndTest:=True)
            Assert.AreEqual(True, trainAcc >= expectedAccuracy)
            Assert.AreEqual(True, testAcc >= expectedAccuracy)
        End Sub

        <TestMethod()>
        Public Sub RPropMLPConsoleDemoTrainAndTestMultiThreadOriginal()
            ' This is the original console demo
            Dim trainAcc# = 0, testAcc# = 0
            m_mlp.ConsoleDemo(trainAcc, testAcc, multiThread:=True, trainAndTest:=True, fastMode:=False)
            Assert.AreEqual(True, trainAcc >= expectedAccuracy)
            Assert.AreEqual(True, testAcc >= expectedAccuracy)
        End Sub

#End If

    End Class

End Namespace