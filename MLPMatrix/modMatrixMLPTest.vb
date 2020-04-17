
' Patrice Dargenton
' Matrix-MultiLayerPerceptron
' From: https://github.com/nlabiris/perceptrons : C# -> VB .NET conversion

Option Infer On

Namespace MatrixMLP

    Module modMatrixMLPTest

        Sub Main()
            Console.WriteLine("Matrix-MultiLayerPerceptron with the classical XOR test.")
            Console.WriteLine("Matrix-MLP may not converge each time, run again if not.")
            MatrixMLPTest()
            Console.WriteLine("Press a key to quit.")
            Console.ReadKey()
        End Sub

        Public Sub MatrixMLPTest()

            Dim p As New MultiLayerPerceptron()

            p.ShowMessage("Matrix MLP test")
            p.ShowMessage("---------------")

            Dim nbIterations%

            ' Sometimes works
            'nbIterations = 100000
            'p.SetActivationFunctionForMatrixForMatrix(TActivationFunctionForMatrix.Sigmoid, gain:=1, center:=0)

            ' Works fine
            nbIterations = 100000
            p.SetActivationFunctionForMatrix(TActivationFunctionForMatrix.HyperbolicTangent,
                gain:=1, center:=0)

            ' Works fine
            'nbIterations = 100000
            'p.SetActivationFunctionForMatrix(TActivationFunctionForMatrix.ELU, gain:=1, center:=0)

            ' Doesn't work
            'nbIterations = 1000000
            'p.SetActivationFunctionForMatrix(TActivationFunctionForMatrix.ReLU, gain:=1, center:=0)

            ' Doesn't work
            'p.SetActivationFunctionForMatrix(TActivationFunctionForMatrix.ReLUSigmoid, gain:=1, center:=0)

            p.InitStruct(m_neuronCountXOR, addBiasColumn:=True)
            p.Init(learningRate:=0.1, weightAdjustment:=0.1)

            p.Randomize(-1, 2)
            Dim nbOutput% = 1
            Dim training As New ML_TrainingData(inputsLength:=2, targetsLength:=nbOutput)
            training.Create()
            Dim inputs!(,) = training.GetInputs
            Dim targets!(,) = training.GetOutputs

            p.nbIterations = nbIterations
            p.printOutput_ = True
            p.inputArray = inputs
            p.targetArray = targets
            'p.Train()
            'p.Train(clsMLPGeneric.enumLearningMode.SemiStochastique)
            p.Train(clsMLPGeneric.enumLearningMode.Stochastique)

            p.TestAllSamples(inputs, nbOutput)
            p.output = p.outputArray
            p.targetArray = targets
            p.ComputeAverageError()

            p.ShowMessage("Matrix MLP test: Done.")

        End Sub

    End Module

End Namespace