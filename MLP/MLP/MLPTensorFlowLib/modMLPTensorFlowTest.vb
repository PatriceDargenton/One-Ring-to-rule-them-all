
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Module modMLPTensorFlowTest

    Sub MainTensorFlowMLP()
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