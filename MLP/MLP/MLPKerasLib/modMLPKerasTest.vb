
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Module modMLPKerasTest

    Sub MainKerasMLP()
        Console.WriteLine("Keras MLP with the classical XOR test.")
        KerasMLPXorTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()
    End Sub

    Public Sub KerasMLPXorTest(Optional nbXor% = 1)

        Dim mlp As New clsMLPKeras

        mlp.ShowMessage("Keras MLP Xor test")
        mlp.ShowMessage("------------------")

        mlp.inputArray = m_inputArrayXOR
        mlp.targetArray = m_targetArrayXOR

        mlp.nbIterations = 1500 ' Sigmoid: works
        'mlp.nbIterations = 2500 ' Sigmoid: works
        'mlp.nbIterations = 2000 ' Hyperbolic tangent: works fine

        'mlp.Initialize(learningRate:=0.001!)
        'mlp.Initialize(learningRate:=0.01!)
        mlp.Initialize(learningRate:=0.02!)

        mlp.printOutput_ = True
        mlp.printOutputMatrix = False

        If nbXor = 1 Then
            'mlp.nbIterations = 1000
            mlp.inputArray = m_inputArrayXOR
            mlp.targetArray = m_targetArrayXOR
            'mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR291, addBiasColumn:=False)
            'mlp.InitializeStruct(m_neuronCountXOR2_10_1, addBiasColumn:=False)
            mlp.InitializeStruct(m_neuronCountXOR2_16_1, addBiasColumn:=False)
            'mlp.InitializeStruct(m_neuronCountXOR4Layers2331, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR4Layers2661, addBiasColumn:=False)
            'mlp.InitializeStruct(m_neuronCountXOR5Layers23331, addBiasColumn:=True)
            mlp.printOutputMatrix = True
        ElseIf nbXor = 2 Then
            mlp.nbIterations = 1500
            mlp.inputArray = m_inputArray2XOR
            mlp.targetArray = m_targetArray2XOR
            'mlp.InitializeStruct(m_neuronCount2XOR482, addBiasColumn:=False)
            mlp.InitializeStruct(m_neuronCount2XOR4_10_2, addBiasColumn:=False)
        ElseIf nbXor = 3 Then
            mlp.nbIterations = 1500
            mlp.inputArray = m_inputArray3XOR
            mlp.targetArray = m_targetArray3XOR
            'mlp.InitializeStruct(m_neuronCount3XOR673, addBiasColumn:=False)
            'mlp.InitializeStruct(m_neuronCount3XOR683, addBiasColumn:=False)
            mlp.InitializeStruct(m_neuronCount3XOR6_10_3, addBiasColumn:=False)
        End If

        mlp.SetActivationFunction(enumActivationFunction.Sigmoid)
        'mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

        mlp.Randomize()

        mlp.PrintWeights()

        WaitForKeyToStart()

        mlp.Train(learningMode:=enumLearningMode.VectorialBatch) ' Works fine

        mlp.ShowMessage("Keras MLP Xor test: Done.")

        If nbXor > 1 Then Exit Sub

        WaitForKeyToContinue("Press a key to print MLP weights")
        mlp.PrintWeights()

    End Sub

End Module