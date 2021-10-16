
' Matrix-MultiLayerPerceptron: clsMPLMatrix
' From https://github.com/nlabiris/perceptrons : C# -> VB .NET conversion

'Imports Perceptron.MatrixMLP ' clsMPLMatrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Module modMatrixMLPTest

    Sub MainMatrixMLP()
        Console.WriteLine("Matrix MLP with the classical XOR test.")
        MatrixMLPTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()
    End Sub

    Public Sub MatrixMLPTest()

Retry:
        Console.WriteLine("")
        Console.WriteLine("")
        Console.WriteLine("Matrix MLP Test, choose an option from the following list:")
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
            Case "1"c : MatrixMLPXorTest(nbXor:=1)
            Case "2"c : MatrixMLPXorTest(nbXor:=2)
            Case "3"c : MatrixMLPXorTest(nbXor:=3)
            Case "4"c
                ' Three layers only, good results!
                MLPGenericIrisFlowerTest(New clsMPLMatrix,
                    "Matrix MLP Iris flower logical test", nbIterations:=4000, threeLayers:=True)
            Case "5"c
                ' Three layers only, good results!
                MLPGenericIrisFlowerTestAnalog(New clsMPLMatrix,
                    "Matrix MLP Iris flower analog test", nbIterations:=4000, threeLayers:=True)
            Case "6"c
                MLPGenericSunspotTest(New clsMPLMatrix,
                    "Matrix MLP Sunspot test")
        End Select

        GoTo Retry

    End Sub

    Public Sub MatrixMLPXorTest(Optional nbXor% = 1)

        Dim mlp As New clsMPLMatrix()

        mlp.ShowMessage("Matrix MLP Xor test")
        mlp.ShowMessage("-------------------")

        Dim nbIterations%

        ' Works
        nbIterations = 5000 '100000
        'mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.Sigmoid, center:=2)

        ' Sometimes works 
        'nbIterations = 100000
        mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.HyperbolicTangent,
            gain:=2)
        'mlp.Init(learningRate:=0.05, weightAdjustment:=0.05)

        ' Works
        'nbIterations = 10000
        'mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.ELU, center:=-2)

        mlp.nbIterations = nbIterations
        mlp.printOutput_ = True
        mlp.printOutputMatrix = False

        If nbXor = 1 Then

            'Dim nbOutput = 1
            'Dim training As New ML_TrainingData(inputsLength:=2, targetsLength:=nbOutput)
            'training.Create()
            'Dim inputs!(,) = training.GetInputs
            'Dim targets!(,) = training.GetOutputs

            'mlp.inputArray = inputs
            'mlp.targetArray = targets
            mlp.inputArray = m_inputArrayXOR
            mlp.targetArray = m_targetArrayXOR
            mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
            ' Not implemented:
            'mlp.InitializeStruct(m_neuronCountXOR4Layers2331, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR5Layers23331, addBiasColumn:=True)
            mlp.printOutputMatrix = True
        ElseIf nbXor = 2 Then
            mlp.inputArray = m_inputArray2XOR
            mlp.targetArray = m_targetArray2XOR
            mlp.InitializeStruct(m_neuronCount2XOR462, addBiasColumn:=True)
        ElseIf nbXor = 3 Then
            'mlp.nbIterations = 10000
            mlp.inputArray = m_inputArray3XOR
            mlp.targetArray = m_targetArray3XOR
            mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
        End If

        mlp.Initialize(learningRate:=0.1, weightAdjustment:=0.1)

        mlp.Randomize(-1, 2)
        mlp.PrintWeights()

        WaitForKeyToStart()

        mlp.Train()
        'mlp.Train(enumLearningMode.Stochastic)

        'mlp.TestAllSamples(inputs, nbOutput)
        'mlp.targetArray = targets
        'mlp.ComputeAverageError()

        mlp.ShowMessage("Matrix MLP Xor test: Done.")

        If nbXor > 1 Then Exit Sub

        WaitForKeyToContinue("Press a key to print MLP weights")
        mlp.PrintWeights()

    End Sub

End Module