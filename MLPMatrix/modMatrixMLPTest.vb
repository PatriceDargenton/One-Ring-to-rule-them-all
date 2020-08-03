
' Matrix-MultiLayerPerceptron
' From: https://github.com/nlabiris/perceptrons : C# -> VB .NET conversion

Imports Perceptron.MatrixMLP ' MultiLayerPerceptron
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Module modMatrixMLPTest

    Sub Main()
        Console.WriteLine("Matrix-MultiLayerPerceptron with the classical XOR test.")
        Console.WriteLine("Matrix-MLP may not converge each time, run again if not.")
        MatrixMLPTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()
    End Sub

    Public Sub MatrixMLPTest()

        Dim mlp As New MultiLayerPerceptron()

        mlp.ShowMessage("Matrix MLP test")
        mlp.ShowMessage("---------------")

        Dim nbIterations%

        ' Works
        nbIterations = 100000
        mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.Sigmoid,
            gain:=1, center:=2)

        ' Sometimes works 
        'nbIterations = 100000
        'mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.HyperbolicTangent,
        '    gain:=1, center:=0)
        'mlp.Init(learningRate:=0.05, weightAdjustment:=0.05)

        ' Works
        'nbIterations = 10000
        'mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.ELU,
        '    gain:=1, center:=-2)

        ' Doesn't work
        'nbIterations = 1000000
        'mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.ReLU, gain:=1, center:=0)

        ' Doesn't work
        'mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.ReLUSigmoid, gain:=1, center:=0)

        mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
        mlp.Initialize(learningRate:=0.1, weightAdjustment:=0.1)

        mlp.Randomize(-1, 2)
        mlp.PrintWeights()

        Console.WriteLine()
        Console.WriteLine("Press a key to start.")
        Console.ReadKey()
        Console.WriteLine()

        Dim nbOutput = 1
        Dim training As New ML_TrainingData(inputsLength:=2, targetsLength:=nbOutput)
        training.Create()
        Dim inputs!(,) = training.GetInputs
        Dim targets!(,) = training.GetOutputs

        mlp.nbIterations = nbIterations
        mlp.printOutput_ = True
        mlp.inputArray = inputs
        mlp.targetArray = targets
        'mlp.Train()
        mlp.Train(enumLearningMode.Stochastic)

        mlp.TestAllSamples(inputs, nbOutput)
        mlp.targetArray = targets
        mlp.ComputeAverageError()

        mlp.ShowMessage("Matrix MLP test: Done.")

    End Sub

End Module