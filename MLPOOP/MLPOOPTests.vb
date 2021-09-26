
Imports Perceptron.Activation
Imports Perceptron.Utilities
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Module Main

    Sub MainOOP()

        OOPMLPXorTest()

    End Sub

    Public Sub OOPMLPXorTest(Optional nbXor% = 1)

        Const minValue# = -1
        Const maxValue# = 1
        'Const minValue# = -0.5
        'Const maxValue# = 0.5
        Dim standard As New Randoms.Standard(
            New Range(minValue, maxValue), seed:=DateTime.Now.Millisecond)

        Dim mlp As New clsMLPOOP(
            learning_rate:=0.5,
            momentum:=0.8, randomizer:=standard,
            activation:=New HyperbolicTangent(alpha:=0.5#))

        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=0.5)

        mlp.ShowMessage("Object-oriented programming MLP Xor test")
        mlp.ShowMessage("----------------------------------------")

        mlp.printOutput_ = True
        mlp.printOutputMatrix = False
        mlp.nbIterations = 2000

        If nbXor = 1 Then
            'num_input:=2, num_hidden:={5}, num_output:=1
            mlp.inputArray = m_inputArrayXOR
            mlp.targetArray = m_targetArrayXOR
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

        mlp.Randomize()
        ' See above: standard As New Randoms.Standard
        'mlp.Randomize(minValue:=-0.5, maxValue:=0.5)
        mlp.PrintWeights()

        WaitForKeyToStart()

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
        'For iteration = 0 To nbIterations - 1
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

        mlp.Train()
        'mlp.Train(enumLearningMode.SemiStochastic) ' Works
        'mlp.Train(enumLearningMode.Stochastic) ' Works

        mlp.ShowMessage("Object-oriented programming MLP Xor test: Done.")

        If nbXor > 1 Then Exit Sub

        WaitForKeyToContinue("Press a key to print MLP weights")
        mlp.PrintWeights()

    End Sub

End Module