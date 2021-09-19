
' Vectorized-MultiLayerPerceptron
' From https://github.com/HectorPulido/Vectorized-multilayer-neural-network : C# -> VB .NET conversion

'Imports Perceptron.VectorizedMatrixMLP
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Module modMatrixVecMLPTest

    Public Sub VectorizedMatrixMLPTest()

Retry:
        Console.WriteLine("")
        Console.WriteLine("")
        Console.WriteLine("Vectorized Matrix MLP Test, choose an option from the following list:")
        Console.WriteLine("0: Exit")
        Console.WriteLine("1: 1 XOR")
        Console.WriteLine("2: 2 XOR")
        Console.WriteLine("3: 3 XOR")
        Console.WriteLine("4: IRIS (Logical)")
        Console.WriteLine("5: IRIS (Analog)")
        Console.WriteLine("6: Sunspot")

        Dim k = Console.ReadKey
        Console.WriteLine("")
        Select Case k.KeyChar
            Case "0"c : Exit Sub
            Case "1"c : VectorizedMatrixMLPXorTest(nbXor:=1)
            Case "2"c : VectorizedMatrixMLPXorTest(nbXor:=2)
            Case "3"c : VectorizedMatrixMLPXorTest(nbXor:=3)
            Case "4"c
                ' Works only using sigmoid activation
                MLPGenericIrisFlowerTest(New clsVectorizedMatrixMLP,
                    "Vectorized Matrix MLP Iris flower logical test", nbIterations:=1000, sigmoid:=True)
            Case "5"c
                ' Works only using sigmoid activation, poor results!
                MLPGenericIrisFlowerTestAnalog(New clsVectorizedMatrixMLP,
                    "Vectorized Matrix MLP Iris flower analog test", sigmoid:=True)
            Case "6"c
                ' Works only using sigmoid activation
                MLPGenericSunspotTest(New clsVectorizedMatrixMLP,
                    "Vectorized Matrix MLP Sunspot test", sigmoid:=True)
        End Select

        GoTo Retry

    End Sub

    Public Sub VectorizedMatrixMLPXorTest(Optional nbXor% = 1)

        Dim mlp As New clsVectorizedMatrixMLP

        mlp.ShowMessage("Vectorized Matrix MLP Xor test")
        mlp.ShowMessage("------------------------------")

        mlp.inputArray = m_inputArrayXOR
        mlp.targetArray = m_targetArrayXOR

        mlp.nbIterations = 2000 ' Sigmoid: works
        'mlp.nbIterations = 5000 ' Hyperbolic tangent: works
        'mlp.nbIterations = 1000 ' Gaussian: works fine
        'mlp.nbIterations = 500 ' Sinus: works fine
        'mlp.nbIterations = 1000 ' Arc tangent: works fine
        'mlp.nbIterations = 1000 ' ELU: works fine (but only one XOR)
        'mlp.nbIterations = 100000 ' ReLU: Does not work yet, but this next one yes:
        'mlp.nbIterations = 5000 ' ReLUSigmoid: works fine
        'mlp.nbIterations = 5000 ' Double threshold: works fine
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid)
        'mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

        mlp.printOutput_ = True
        mlp.printOutputMatrix = False

        If nbXor = 1 Then
            mlp.inputArray = m_inputArrayXOR
            mlp.targetArray = m_targetArrayXOR
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

        mlp.Initialize(learningRate:=0.1, weightAdjustment:=1)

        mlp.Randomize()

        mlp.PrintWeights()

        WaitForKeyToStart()

        mlp.TrainVector()
        'mlp.Train() ' Works fine
        'mlp.Train(enumLearningMode.Systematic) ' Works fine
        'mlp.Train(enumLearningMode.SemiStochastic) ' Works
        'mlp.Train(enumLearningMode.Stochastic) ' Works

        mlp.ShowMessage("Vectorized Matrix MLP Xor test: Done.")

        If nbXor > 1 Then Exit Sub

        WaitForKeyToContinue("Press a key to print MLP weights")
        mlp.PrintWeights()

    End Sub

End Module