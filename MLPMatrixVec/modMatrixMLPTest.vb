
' Vectorized-MultiLayerPerceptron
' From https://github.com/HectorPulido/Vectorized-multilayer-neural-network : C# -> VB .NET conversion

Imports Perceptron.VectorizedMatrixMLP
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Module modMatrixVecMLPTest

    Sub Main()
        Console.WriteLine("Vectorized-MultiLayerPerceptron with the classical XOR test.")
        VectorizedMatrixMLPTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()
    End Sub

    Public Sub VectorizedMatrixMLPTest()

        Dim mlp As New clsVectorizedMatrixMLP

        mlp.ShowMessage("Vectorized Matrix MLP test")
        mlp.ShowMessage("--------------------------")

        mlp.inputArray = m_inputArrayXOR
        mlp.targetArray = m_targetArrayXOR

        mlp.nbIterations = 10000 ' Sigmoid: works
        'mlp.nbIterations = 5000 ' Hyperbolic tangent: works
        'mlp.nbIterations = 1000 ' Gaussian: works fine
        'mlp.nbIterations = 500 ' Sinus: works fine
        'mlp.nbIterations = 1000 ' Arc tangent: works fine
        'mlp.nbIterations = 1000 ' ELU: works fine (but only one XOR)
        'mlp.nbIterations = 100000 ' ReLU: Does not work yet, but this next one yes:
        'mlp.nbIterations = 5000 ' ReLUSigmoid: works fine
        'mlp.nbIterations = 5000 ' Double threshold: works fine
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=1, center:=0)

        mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
        mlp.Init(learningRate:=0.1, weightAdjustment:=1)

        mlp.Randomize()

        mlp.PrintWeights()

        Console.WriteLine()
        Console.WriteLine("Press a key to start.")
        Console.ReadKey()
        Console.WriteLine()

        mlp.printOutput_ = True
        mlp.TrainVector()
        'mlp.Train() ' Works fine
        'mlp.Train(enumLearningMode.Systematic) ' Works fine
        'mlp.Train(enumLearningMode.SemiStochastic) ' Works
        'mlp.Train(enumLearningMode.Stochastic) ' Works

        mlp.ShowMessage("Vectorized Matrix MLP test: Done.")

    End Sub

End Module