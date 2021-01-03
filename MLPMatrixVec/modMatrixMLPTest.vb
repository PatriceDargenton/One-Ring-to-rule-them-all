
' Vectorized-MultiLayerPerceptron
' From https://github.com/HectorPulido/Vectorized-multilayer-neural-network : C# -> VB .NET conversion

Imports Perceptron.VectorizedMatrixMLP
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Module modMatrixVecMLPTest

    Public Sub VectorizedMatrixMLPTest()

        XORTest()
        NextTest()

        ' Works only using sigmoid activation
        MLPGenericIrisFlowerTest(New VectorizedMatrixMLP.clsVectorizedMatrixMLP,
            "Vectorized Matrix MLP Iris flower test", nbIterations:=1000, sigmoid:=True)
        NextTest()

        ' Works only using sigmoid activation, poor results!
        MLPGenericIrisFlowerTestAnalog(New VectorizedMatrixMLP.clsVectorizedMatrixMLP,
            "Vectorized Matrix MLP Iris flower test", sigmoid:=True)

    End Sub

    Public Sub XORTest()

        Console.WriteLine("Vectorized Matrix MLP with the classical XOR test.")

        VectorizedMatrixMLPXorTest()
        NextTest()

        VectorizedMatrixMLPXorTest(nbXor:=2)
        NextTest()

        VectorizedMatrixMLPXorTest(nbXor:=3)

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