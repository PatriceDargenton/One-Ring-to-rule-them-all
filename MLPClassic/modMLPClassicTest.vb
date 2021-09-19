
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPClassic ' enumLearningMode

Module modMLPClassicTest

    Sub Main()
        Console.WriteLine("MultiLayerPerceptron with the classical XOR test.")
        ClassicMLPXorTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()
    End Sub

    Public Sub ClassicMLPXorTest(Optional nbXor% = 1)

        Dim mlp As New clsMLPClassic

        mlp.ShowMessage("Classic MLP Xor test")
        mlp.ShowMessage("--------------------")

        mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

        Dim nbIterations%

        'mlp.SetActivationFunction(enumActivationFunction.Sigmoid)
        'nbIterations = 10000 ' Sigmoid: works

        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent)
        nbIterations = 2000 ' Hyperbolic tangent: works fine

        'mlp.SetActivationFunction(enumActivationFunction.Gaussian)
        'nbIterations = 1000 ' Gaussian: works fine

        'mlp.SetActivationFunction(enumActivationFunction.Sinus)
        'nbIterations = 1000 ' Sinus: works fine

        'mlp.SetActivationFunction(enumActivationFunction.ArcTangent)
        'nbIterations = 1000 ' ArcTangent: works fine

        'mlp.SetActivationFunction(enumActivationFunction.ELU)
        'nbIterations = 2000 ' ELU: works

        'mlp.SetActivationFunction(enumActivationFunction.ReLu, gain:=0.9)
        'nbIterations = 1000 ' ReLU: works fine

        'mlp.SetActivationFunction(enumActivationFunction.ReLuSigmoid)
        'nbIterations = 10000 ' ReLUSigmoid: works?

        'mlp.SetActivationFunction(enumActivationFunction.DoubleThreshold)
        'nbIterations = 10000 ' DoubleThreshold: works fine

        mlp.printOutput_ = True
        mlp.printOutputMatrix = False
        mlp.nbIterations = nbIterations

        If nbXor = 1 Then
            mlp.inputArray = m_inputArrayXOR
            mlp.targetArray = m_targetArrayXOR
            mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR231, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR4Layers2331, addBiasColumn:=True)
            'mlp.InitializeStruct(m_neuronCountXOR5Layers23331, addBiasColumn:=True)
            mlp.printOutputMatrix = True
            mlp.nbIterations = 4000
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
        'mlp.Randomize(minValue:=-0.5, maxValue:=0.5)
        mlp.PrintWeights()

        WaitForKeyToStart()

        'mlp.InitWeights(1, {
        '     {0.28, 0.28, 0.76},
        '     {0.25, 0.88, 0.62}})
        'mlp.InitWeights(2, {
        '     {0.56, 0.92, 0.19}})

        mlp.Train()
        'mlp.Train(enumLearningMode.SemiStochastic)
        'mlp.Train(enumLearningMode.Stochastic)

        mlp.ShowMessage("Classic MLP Xor test: Done.")

        If nbXor > 1 Then Exit Sub

        WaitForKeyToContinue("Press a key to print MLP weights")
        mlp.PrintWeights()

    End Sub

End Module