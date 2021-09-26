
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode

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

        mlp.printOutput_ = True
        mlp.printOutputMatrix = False
        mlp.nbIterations = 1000

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
        'mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=0.2!)
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=0.7!)
        mlp.nbIterationsBatch = 10

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