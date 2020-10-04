
' Matrix-MultiLayerPerceptron
' From: https://github.com/nlabiris/perceptrons : C# -> VB .NET conversion

Imports Perceptron.MatrixMLP ' MultiLayerPerceptron
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Module modMatrixMLPTest

    Public Sub MatrixMLPTest()

        XORTest()
        NextTest()

        ' Three layers only, poor results!
        MLPGenericIrisFlowerTest(New MatrixMLP.MultiLayerPerceptron,
            "Matrix MLP Iris flower test", nbIterations:=4000, threeLayers:=True)

    End Sub

    Public Sub XORTest()

        Console.WriteLine("Matrix-MultiLayerPerceptron with the classical XOR test.")
        Console.WriteLine("Matrix-MLP may not converge each time, run again if not.")

        MatrixMLPXorTest()
        NextTest()

        MatrixMLPXorTest(nbXor:=2)
        NextTest()

        MatrixMLPXorTest(nbXor:=3)

    End Sub

    Public Sub MatrixMLPXorTest(Optional nbXor% = 1)

        Dim mlp As New MultiLayerPerceptron()

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

            mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
            ' Not implemented:
            'mlp.InitializeStruct(m_neuronCountXOR4Layers2331, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR5Layers23331, addBiasColumn:=True)
            mlp.printOutputMatrix = True
            mlp.inputArray = m_inputArrayXOR
            mlp.targetArray = m_targetArrayXOR
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