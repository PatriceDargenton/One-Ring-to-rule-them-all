
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Module modMLPBrightWireTest

    Sub MainBrightWireMLP()
        Console.WriteLine("BrightWire.NET MLP with the classical XOR test.")
        BrightWireMLPXorTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()
    End Sub

    Public Sub BrightWireMLPXorTest(Optional nbXor% = 1)

        Dim mlp As New clsMLPBrightWire

        mlp.ShowMessage("BrightWire.NET MLP test")
        mlp.ShowMessage("-----------------------")

        mlp.inputArray = m_inputArrayXOR
        mlp.targetArray = m_targetArrayXOR

        mlp.Initialize(learningRate:=0.1)

        mlp.printOutput_ = True
        mlp.printOutputMatrix = False

        If nbXor = 1 Then
            mlp.InitializeStruct(m_neuronCountXOR241, addBiasColumn:=True)
            mlp.printOutputMatrix = True
            mlp.inputArray = m_inputArrayXOR
            mlp.targetArray = m_targetArrayXOR
        ElseIf nbXor = 2 Then
            mlp.inputArray = m_inputArray2XOR
            mlp.targetArray = m_targetArray2XOR
            mlp.InitializeStruct({4, 9, 2}, addBiasColumn:=True)
        ElseIf nbXor = 3 Then
            mlp.inputArray = m_inputArray3XOR
            mlp.targetArray = m_targetArray3XOR
            mlp.InitializeStruct({6, 12, 3}, addBiasColumn:=True)
        End If

        mlp.trainingAlgorithm = enumTrainingAlgorithm.RMSProp

        mlp.nbIterationsBatch = 20
        mlp.nbIterations = 2000 ' Sigmoid: works
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

        'mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)
        'mlp.SetActivationFunction(enumActivationFunction.ReLu)

        mlp.Randomize()

        mlp.PrintWeights()

        WaitForKeyToStart()

        mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

        mlp.ShowMessage("BrightWire.NET MLP test: Done.")

        If nbXor > 1 Then Exit Sub

        WaitForKeyToContinue("Press a key to print MLP weights")
        mlp.PrintWeights()

    End Sub

End Module