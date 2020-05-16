
Imports Perceptron.NetworkOOP
Imports Perceptron.Activation
Imports Perceptron.Utilities

Imports Microsoft.VisualStudio.TestTools.UnitTesting
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Module Main

    Sub Main()

        OOPMLPScenario1()

    End Sub

    Public Sub OOPMLPScenario1()

        Dim standard As New Randoms.Standard(
            New Range(-1, 1), seed:=DateTime.Now.Millisecond)

        Dim mlp As New MultilayerPerceptron(
            learning_rate:=0.5,
            momentum:=0.8, randomizer:=standard,
            activation:=New BipolarSigmoid(Alpha:=0.5))

        'mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=0.5, center:=0)
        'mlp.ActivationFunction = Nothing

        mlp.ShowMessage("Object-oriented programming MLP test")
        mlp.ShowMessage("------------------------------------")

        'num_input:=2, num_hidden:={5}, num_output:=1
        mlp.InitStruct({2, 5, 1}, addBiasColumn:=True)
        'mlp.InitStruct({4, 4, 2}, addBiasColumn:=True)
        mlp.Randomize()
        mlp.PrintWeights()

        Console.WriteLine()
        Console.WriteLine("Press a key to start.")
        Console.ReadKey()
        Console.WriteLine()

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
        'For iteration As Integer = 0 To nbIterations - 1
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

        mlp.nbIterations = 10000
        mlp.inputArray = m_inputArrayXOR
        mlp.targetArray = m_targetArrayXOR
        mlp.printOutput_ = True
        mlp.Train()
        'mlp.Train(enumLearningMode.SemiStochastique) ' Does not work?
        'mlp.Train(enumLearningMode.Stochastique)

        mlp.ShowMessage("Object-oriented programming MLP test: Done.")

    End Sub

End Module

Namespace OOPMLP

    <TestClass()>
    Public Class MultiLayerPerceptronOOPTest

        Private m_mlp As New MultilayerPerceptron

        ' Weights are quite the same as MLP Classic, but not exactly:
        'Private m_mlp As New clsMLPClassic ' 3/8 fails

        ' Weights are not stored in the same way:
        'Private m_mlp As New MatrixMLP.MultiLayerPerceptron ' 6/6 fails
        'Private m_mlp As New VectorizedMatrixMLP.clsVectorizedMatrixMLP ' 6/6 fails

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
        Public Sub MLPOOPXORSigmoid()

            ' OOP activation function: before Init()
            'm_mlp.ActivationFunction = New Sigmoid(alpha:=1, center:=1)

            m_mlp.Init(learningRate:=1.1!, weightAdjustment:=0.5!)
            InitXOR()

            m_mlp.nbIterations = 700
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=1)

            m_mlp.WeightInit(1, {
                {0.42, 0.82, 0.15},
                {0.2, 0.8, 0.28}})
            m_mlp.WeightInit(2, {
                {0.71, 0.25, 0.35}})

            m_mlp.PrintWeights()

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As MatrixMLP.Matrix = expectedOutput ' Double(,) -> Matrix

            Dim outputMaxtrix As MatrixMLP.Matrix = m_mlp.outputArraySingle
            Dim sOutput = outputMaxtrix.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.05
            Dim rLoss# = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub MLPOOPXORSigmoidWithoutBias()

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
        Public Sub MLPOOPXORSigmoidWithoutBias231()

            m_mlp.Init(learningRate:=0.08!, weightAdjustment:=0.02!)
            InitXOR()
            m_mlp.InitStruct(m_neuronCountXOR231, addBiasColumn:=False)

            m_mlp.nbIterations = 12000
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
        Public Sub MLPOOPXORSigmoidWithoutBias2()

            ' OOP activation function: before Init()
            'm_mlp.ActivationFunction = New Sigmoid(alpha:=1, center:=2.2)

            m_mlp.Init(learningRate:=0.9!, weightAdjustment:=0.05!)
            InitXOR()
            m_mlp.InitStruct(m_neuronCountXOR231, addBiasColumn:=False)

            m_mlp.nbIterations = 60000
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=2.2!)

            m_mlp.WeightInit(1, {
                {0.66, 0.53},
                {0.65, 0.69},
                {0.82, 0.56}})
            m_mlp.WeightInit(2, {
                {0.62, 0.54, 0.5}})

            m_mlp.PrintWeights()

            m_mlp.Train()

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As MatrixMLP.Matrix = expectedOutput ' Double(,) -> Matrix

            Dim outputMaxtrix As MatrixMLP.Matrix = m_mlp.outputArraySingle
            Dim sOutput = outputMaxtrix.ToStringWithFormat(dec:="0.0")

            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sOutput, sExpectedOutput)

            Dim rExpectedLoss# = 0.03
            Dim rLoss# = m_mlp.ComputeAverageError()
            Dim rLossRounded# = Math.Round(rLoss, 2)
            'Assert.AreEqual(rExpectedLoss, rLossRounded)
            Assert.AreEqual(rLossRounded <= rExpectedLoss, True)

        End Sub

        <TestMethod()>
        Public Sub MLPOOP2XORSigmoid()

            ' OOP activation function: before Init()
            'm_mlp.ActivationFunction = New Sigmoid(alpha:=1, center:=2)

            m_mlp.Init(learningRate:=0.8!, weightAdjustment:=0.1!)
            Init2XOR()

            m_mlp.nbIterations = 2000
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1, center:=2)

            m_mlp.WeightInit(1, {
                {0.68, 0.08, 0.83, 0.76, 0.87},
                {0.38, 0.93, 0.9, 0.5, 0.92},
                {0.98, 0.98, 0.97, 0.15, 0.11},
                {0.77, 0.85, 0.27, 0.67, 0.69}})
            m_mlp.WeightInit(2, {
                {0.72, 0.26, 0.06, 0.44, 0.45},
                {0.7, 0.88, 0.04, 0.53, 0.21}})

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
        Public Sub MLPOOP3XORSigmoid()

            ' OOP activation function: before Init()
            'm_mlp.ActivationFunction = New Sigmoid(alpha:=1, center:=1.8!)

            m_mlp.Init(learningRate:=1.0!, weightAdjustment:=0.3!)
            Init3XOR()

            m_mlp.nbIterations = 150
            m_mlp.SetActivationFunction(TActivationFunction.Sigmoid, gain:=1.1!, center:=2)

            m_mlp.WeightInit(1, {
                {0.22, 0.22, 0.82, 0.65, 0.79, 0.29, 0.5},
                {0.04, 0.03, 0.67, 0.4, 0.64, 0.9, 0.35},
                {0.05, 0.28, 0.11, 0.78, 0.64, 0.73, 0.92},
                {0.69, 0.21, 0.84, 0.92, 0.34, 0.99, 0.96},
                {0.97, 0.14, 0.19, 0.78, 0.17, 0.06, 0.09},
                {0.61, 0.18, 0.17, 0.52, 0.52, 0.1, 0.86}})
            m_mlp.WeightInit(2, {
                {0.75, 0.93, 0.37, 0.93, 0.12, 0.18, 0.69},
                {0.33, 0.56, 0.32, 0.93, 0.27, 0.96, 0.11},
                {0.66, 0.22, 0.16, 0.44, 0.17, 0.58, 0.2}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArray3XOR
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
        Public Sub MLPOOP3XORHTangent()

            ' OOP activation function: before Init()
            'm_mlp.ActivationFunction = New HyperbolicTangent(Alpha:=1, center:=0.5!)

            m_mlp.Init(learningRate:=0.1!, weightAdjustment:=0.1!)
            Init3XOR()

            m_mlp.nbIterations = 300
            m_mlp.SetActivationFunction(TActivationFunction.HyperbolicTangent, gain:=1, center:=0.5!)

            m_mlp.WeightInit(1, {
                {0.19, 0.43, 0.4, 0.37, 0.83, 0.66, 0.39},
                {0.36, 0.59, 0.63, 0.15, 0.02, 0.02, 0.77},
                {0.98, 0.45, 0.89, 0.02, 0.93, 0.63, 0.22},
                {0.16, 0.94, 0.84, 0.3, 0.68, 0.32, 0.17},
                {0.47, 0.15, 0.23, 0.52, 0.24, 0.18, 0.72},
                {0.79, 0.67, 0.64, 0.47, 0.52, 0.31, 0.63}})
            m_mlp.WeightInit(2, {
                {0.93, 0.9, 0.31, 0.94, 0.18, 0.5, 0.86},
                {0.6, 0.04, 0.62, 0.06, 0.06, 0.33, 0.75},
                {0.02, 0.23, 0.72, 0.58, 0.48, 0.45, 0.35}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArray3XOR
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
        Public Sub MLPOOP3XORGaussian()

            m_mlp.Init(learningRate:=0.1!, weightAdjustment:=0.09!)
            Init3XOR()

            m_mlp.nbIterations = 150
            m_mlp.SetActivationFunction(TActivationFunction.Gaussian, gain:=1, center:=1)

            m_mlp.WeightInit(1, {
                {0.64, 0.03, 0.82, 0.43, 0.84, 0.69, 0.69},
                {0.63, 0.08, 0.77, 0.86, 0.23, 0.16, 0.71},
                {0.56, 0.7, 0.63, 0.51, 0.14, 0.5, 0.39},
                {0.86, 0.98, 0.6, 0.64, 0.28, 0.05, 0.69},
                {0.13, 0.83, 0.03, 0.1, 0.68, 0.36, 0.45},
                {1.0, 0.31, 0.75, 0.49, 0.25, 0.03, 0.16}})
            m_mlp.WeightInit(2, {
                {0.52, 0.31, 0.46, 0.99, 0.34, 0.07, 0.86},
                {0.42, 0.57, 0.38, 0.72, 1.0, 0.03, 0.75},
                {0.29, 0.32, 0.14, 0.32, 0.53, 0.91, 0.41}})

            m_mlp.Train()

            Dim expectedOutput = m_targetArray3XOR
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

    End Class

End Namespace