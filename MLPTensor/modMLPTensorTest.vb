
Imports Perceptron.DLFramework ' Tensor
Imports Perceptron.DLFramework.Layers ' Linear, Sequential
Imports Perceptron.DLFramework.Layers.Loss ' MeanSquaredError
Imports Perceptron.DLFramework.Optimizers ' StochasticGradientDescent

Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Module modMLPTensorTest

    Sub Main()
        'SimpleTest()
        Console.WriteLine("Tensor MultiLayerPerceptron with the classical XOR test.")
        TensorMLPXorTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()
    End Sub

    Public Sub SimpleTest()

        Dim r As New Random

        ' Works with an easy test:
        'Dim sourceDble As Double(,) = {{0, 0}, {0, 1}, {1, 0}, {1, 1}}
        'Dim targetDble As Double(,) = {{0}, {1}, {0}, {1}}
        ' Does not work with the classical and difficult XOR test:
        Dim sourceDble As Double(,) = {{1, 0}, {0, 0}, {0, 1}, {1, 1}}
        Dim targetDble As Double(,) = {{1}, {0}, {1}, {0}}

        Dim sourceMatrix As Matrix = sourceDble
        Dim data As New Tensor(sourceMatrix, autoGrad:=True)

        Dim targetMatrix As Matrix = targetDble
        Dim target As New Tensor(targetMatrix, autoGrad:=True)

        Dim seq As New Sequential()
        seq.Layers.Add(New Linear(2, 3, r, addBias:=True))
        seq.Layers.Add(New Linear(3, 1, r, addBias:=True))

        Dim sgd As New StochasticGradientDescent(seq.Parameters, 0.1!)

        Dim mse As New MeanSquaredError()

        For i = 0 To 20
            Dim pred = seq.Forward(data)
            Dim loss = mse.Forward(pred, target)
            loss.Backward(New Tensor(Matrix.Ones(loss.Data.c, loss.Data.r)))
            sgd.Step_(zero:=True)
            Debug.WriteLine("Epoch: {" & i & "} Loss: { " & loss.ToString() & "}")
        Next

    End Sub

    Public Sub TensorMLPXorTest(Optional nbXor% = 1)

        Dim mlp As New clsMLPTensor

        mlp.ShowMessage("Tensor MLP Xor test")
        mlp.ShowMessage("-------------------")

        mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.05!)

        Dim nbIterations%

        mlp.SetActivationFunctionOptimized(
            enumActivationFunctionOptimized.Sigmoid)
        'mlp.SetActivationFunctionOptimized(
        '    enumActivationFunctionOptimized.HyperbolicTangent, gain:=2)

        ' Works only for 1 XOR, and 2 XOR in not vectorized learning mode:
        'mlp.SetActivationFunctionOptimized(
        '    enumActivationFunctionOptimized.ELU) ', center:=0.4)

        nbIterations = 2000

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
        ElseIf nbXor = 2 Then
            mlp.inputArray = m_inputArray2XOR
            mlp.targetArray = m_targetArray2XOR
            'mlp.InitializeStruct(m_neuronCount2XOR462, addBiasColumn:=True)
            mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)
        ElseIf nbXor = 3 Then
            mlp.inputArray = m_inputArray3XOR
            mlp.targetArray = m_targetArray3XOR
            mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
        End If

        mlp.Randomize()

        mlp.PrintWeights()

        WaitForKeyToStart()

        mlp.TrainVector() ' Works fine
        'mlp.Train()
        'mlp.Train(enumLearningMode.Systematic) ' Works fine
        'mlp.Train(enumLearningMode.SemiStochastic) ' Works
        'mlp.Train(enumLearningMode.Stochastic) ' Works

        mlp.ShowMessage("Tensor MLP Xor test: Done.")

        If nbXor > 1 Then Exit Sub

        WaitForKeyToContinue("Press a key to print MLP weights")
        mlp.PrintWeights()

    End Sub

End Module