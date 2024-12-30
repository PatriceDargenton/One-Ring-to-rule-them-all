
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode

#If NeuralNetworkNETEngine Then
Imports Perceptron.clsMLPNeuralNet ' TrainingAlgorithmType
' BC40025: Type of this member is not CLS-compliant:
'Imports NeuralNetworkNET.SupervisedLearning.Algorithms ' TrainingAlgorithmType
#End If

Module modMLPNeuralNetTest

    Sub MainNeuralNetMLP()
        Console.WriteLine("NeuralNet.NET MLP with the classical XOR test.")
        NeuralNetMLPXorTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()
    End Sub

    Public Sub NeuralNetMLPXorTest(Optional nbXor% = 1)

#If NeuralNetworkNETEngine Then

        Dim mlp As New clsMLPNeuralNet

        mlp.ShowMessage("NeuralNet.NET MLP test")
        mlp.ShowMessage("----------------------")

        mlp.inputArray = m_inputArrayXOR
        mlp.targetArray = m_targetArrayXOR

        mlp.Initialize(learningRate:=0)

        mlp.printOutput_ = True
        mlp.printOutputMatrix = False

        If nbXor = 1 Then
            mlp.InitializeStruct(m_neuronCountXOR271, addBiasColumn:=True)
            mlp.printOutputMatrix = True
            mlp.inputArray = m_inputArrayXOR
            mlp.targetArray = m_targetArrayXOR
        ElseIf nbXor = 2 Then
            mlp.inputArray = m_inputArray2XOR
            mlp.targetArray = m_targetArray2XOR
            mlp.InitializeStruct(m_neuronCount2XOR462, addBiasColumn:=True)
        ElseIf nbXor = 3 Then
            mlp.inputArray = m_inputArray3XOR
            mlp.targetArray = m_targetArray3XOR
            mlp.InitializeStruct(m_neuronCount3XOR673, addBiasColumn:=True)
        End If

        mlp.trainingAlgorithm = enumTrainingAlgorithm.RMSProp

        'mlp.nbIterationsBatch = mlp.minBatchSize ' Can be 1 using Sigmoid and RMSProp
        'mlp.nbIterations = 15000 ' Sigmoid: works
        'mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

        mlp.nbIterations = 4000 ' Hyperbolic tangent: works fine
        If nbXor = 3 Then mlp.nbIterations = 15000
        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

        mlp.Randomize()

        mlp.PrintWeights()

        WaitForKeyToStart()

        mlp.Train(learningMode:=enumLearningMode.VectorialBatch) ' Works fine

        mlp.ShowMessage("NeuralNet.NET MLP test: Done.")

        If nbXor > 1 Then Exit Sub

        WaitForKeyToContinue("Press a key to print MLP weights")
        mlp.PrintWeights()

#End If

    End Sub

End Module