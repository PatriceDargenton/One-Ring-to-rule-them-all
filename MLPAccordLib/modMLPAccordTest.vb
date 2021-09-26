
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Module modMLPAccordTest

    Sub Main()
        Console.WriteLine("Accord MLP with the classical XOR test.")
        AccordMLPXorTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()
    End Sub

    Public Sub AccordMLPXorTest(Optional nbXor% = 1)

        Dim mlp As New clsMLPAccord

        mlp.ShowMessage("Accord MLP Xor test")
        mlp.ShowMessage("-------------------")

        mlp.inputArray = m_inputArrayXOR
        mlp.targetArray = m_targetArrayXOR

        'mlp.nbIterations = 5000 ' Sigmoid: works
        mlp.nbIterations = 2000 ' Hyperbolic tangent: works fine
        'mlp.nbIterations = 20000 ' Stochastic

        mlp.Initialize(learningRate:=0.05!, weightAdjustment:=0.1!)

        mlp.printOutput_ = True
        mlp.printOutputMatrix = False

        If nbXor = 1 Then
            mlp.inputArray = m_inputArrayXOR
            mlp.targetArray = m_targetArrayXOR
            'num_input:=2, num_hidden:={5}, num_output:=1
            'mlp.InitializeStruct({2, 5, 1}, addBiasColumn:=True)
            mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR4Layers2331, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR5Layers23331, addBiasColumn:=True)
            mlp.printOutputMatrix = True
        ElseIf nbXor = 2 Then
            mlp.inputArray = m_inputArray2XOR
            mlp.targetArray = m_targetArray2XOR
            mlp.InitializeStruct(m_neuronCount2XOR462, addBiasColumn:=True)
        ElseIf nbXor = 3 Then
            mlp.inputArray = m_inputArray3XOR
            mlp.targetArray = m_targetArray3XOR
            mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
        End If

        'mlp.SetActivationFunctionOptimized(
        '    enumActivationFunctionOptimized.Sigmoid, gain:=2)
        mlp.SetActivationFunctionOptimized(
            enumActivationFunctionOptimized.HyperbolicTangent, gain:=2)

        mlp.Randomize()

        mlp.PrintWeights()

        WaitForKeyToStart()

        mlp.TrainVector() ' Works fine
        'mlp.Train() ' Works fine
        'mlp.Train(enumLearningMode.Systematic) ' Works fine
        'mlp.Train(enumLearningMode.SemiStochastic) ' Works
        'mlp.Train(enumLearningMode.Stochastic) ' Works

        mlp.ShowMessage("Accord MLP Xor test: Done.")

        If nbXor > 1 Then Exit Sub

        WaitForKeyToContinue("Press a key to print MLP weights")
        mlp.PrintWeights()

    End Sub

End Module