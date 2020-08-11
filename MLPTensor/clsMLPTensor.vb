
' From https://github.com/HectorPulido/Machine-learning-Framework-Csharp : C# -> VB .NET conversion

Imports Perceptron.DLFramework ' Tensor
Imports Perceptron.DLFramework.Layers ' Linear, Sequential
Imports Perceptron.DLFramework.Layers.Loss ' MeanSquaredError
Imports Perceptron.DLFramework.Optimizers ' StochasticGradientDescent
Imports Perceptron.Utility ' Matrix

Public Class clsMLPTensor : Inherits clsVectorizedMLPGeneric

    Private input, target, pred, loss As Tensor
    Private weights As List(Of Tensor)
    Private seq As Sequential
    Private mse As MeanSquaredError
    Private sgd As StochasticGradientDescent

    Private nbHiddenNeuronsTensor%
    Private nbHiddenNeuronsTensorWithBias%

    Public Overrides Sub InitializeStruct(neuronCount%(), addBiasColumn As Boolean)

        Me.useBias = addBiasColumn

        If Not Me.useBias Then
            Throw New NotImplementedException(
                "useBias=False is not implemented for clsTensorMLP!")
        End If

        Me.layerCount = neuronCount.Length
        Me.nbInputNeurons = neuronCount(0)
        Me.nbHiddenNeurons = neuronCount(1)
        Me.nbOutputNeurons = neuronCount(Me.layerCount - 1)
        Me.neuronCount = neuronCount
        Me.nbHiddenNeuronsTensor = Me.nbHiddenNeurons + Me.nbInputNeurons
        Me.nbHiddenNeuronsTensorWithBias = Me.nbHiddenNeuronsTensor
        If Me.useBias Then Me.nbHiddenNeuronsTensorWithBias += 1

        Me.weights = New List(Of Tensor)
        For i = 1 To Me.layerCount - 1
            Dim nbNodes1 = Me.nbHiddenNeurons
            Dim nbNodes2 = Me.nbHiddenNeurons
            If Me.useBias AndAlso i > 1 Then nbNodes1 += 1
            If Me.useBias AndAlso i < Me.layerCount - 1 Then nbNodes2 += 1
            If i = 1 Then nbNodes1 = Me.nbInputNeurons
            If i = Me.layerCount - 1 Then nbNodes2 = Me.nbOutputNeurons
            Me.weights.Add(New Tensor(
                Matrix.Zeros(nbNodes1, nbNodes2), autoGrad:=True))
        Next

        Me.seq = New Sequential()
        For i = 0 To Me.layerCount - 1
            Dim nbNodes1 = Me.nbHiddenNeuronsTensorWithBias
            Dim nbNodes2 = Me.nbHiddenNeuronsTensorWithBias
            If i = 0 Then nbNodes1 = Me.nbInputNeurons
            If i = Me.layerCount - 1 Then nbNodes2 = Me.nbOutputNeurons
            Me.seq.Layers.Add(New Linear(nbNodes1, nbNodes2, Me.useBias))
            AddLayerWithActivationFunction()
        Next

        InitializeSequential()

        Me.mse = New MeanSquaredError()

    End Sub

    Private Sub AddLayerWithActivationFunction()
        Select Case Me.m_actFunc
            Case enumActivationFunction.Sigmoid : Me.seq.Layers.Add(
                New SigmoidLayer(Me.m_center))
            Case enumActivationFunction.HyperbolicTangent : Me.seq.Layers.Add(
                New HyperbolicTangentLayer(Me.m_center))
            Case enumActivationFunction.ELU : Me.seq.Layers.Add(
                New ELULayer(Me.m_center))
        End Select
    End Sub

    Public Overrides Sub InitializeWeights(layer%, weights#(,))
        Dim wMatrix As Matrix = weights
        Me.weights(layer - 1) = New Tensor(wMatrix, autoGrad:=True)
    End Sub

    Public Sub WeightInitLayerLinear(layer%, weights#(,),
        Optional addBias As Boolean = True, Optional bias#(,) = Nothing)

        Dim wMatrix As Matrix = weights

        Dim i = layer - 1
        Dim nbNodes1 = Me.nbHiddenNeuronsTensorWithBias
        Dim nbNodes2 = Me.nbHiddenNeuronsTensorWithBias
        If i = 0 Then nbNodes1 = Me.nbInputNeurons
        If i = Me.layerCount - 1 Then nbNodes2 = Me.nbOutputNeurons

        Dim j%
        Dim biasMatrix As Matrix
        If addBias Then
            If IsNothing(bias) Then
                biasMatrix = Matrix.Zeros(1, nbNodes2) ' For functionnal tests
            Else
                biasMatrix = bias ' To save and restore weights
            End If
            j = i * 2
            Me.seq.Layers(j) = New Linear(nbNodes1, nbNodes2, wMatrix, biasMatrix)
        Else
            j = i
            Me.seq.Layers(j) = New Linear(nbNodes1, nbNodes2, wMatrix, addBias:=False)
        End If

    End Sub

    Public Sub InitializeSequential()
        Me.sgd = New StochasticGradientDescent(
            Me.seq.Parameters, Me.learningRate, Me.weightAdjustment)
        'Debug.WriteLine("seq.prm=" & Me.seq.ParametersToString)
    End Sub

    Public Overrides Sub Randomize(Optional minValue! = 0, Optional maxValue! = 1)

        Me.weights = New List(Of Tensor)
        Me.rnd = New Random
        Me.seq = New Sequential()
        For i = 1 To Me.layerCount - 1
            Dim nbNodes1 = Me.nbHiddenNeurons
            Dim nbNodes2 = Me.nbHiddenNeurons
            If i = 1 Then nbNodes1 = Me.nbInputNeurons
            If Me.useBias AndAlso i > 1 Then nbNodes1 += 1
            If Me.useBias AndAlso i < Me.layerCount - 1 Then nbNodes2 += 1
            If i = Me.layerCount - 1 Then nbNodes2 = Me.nbOutputNeurons
            Me.weights.Add(New Tensor(Matrix.Randomize(
                nbNodes1, nbNodes2, Me.rnd), autoGrad:=True))
        Next

        For i = 0 To Me.layerCount - 1
            Dim nbNodes1 = Me.nbInputNeurons
            Dim nbNodes2 = Me.nbHiddenNeurons + Me.nbInputNeurons
            If Me.useBias Then nbNodes2 += 1
            If i > 0 Then nbNodes1 = nbNodes2
            If i = Me.layerCount - 1 Then nbNodes2 = Me.nbOutputNeurons
            Me.seq.Layers.Add(New Linear(nbNodes1, nbNodes2, Me.rnd, Me.useBias))
            AddLayerWithActivationFunction()
        Next
        InitializeSequential()

    End Sub

    Private Sub SetInputOneSample(input!())

        Dim inputsDble#(0, input.Length - 1)
        'inputsDble = clsMLPHelper.FillArray2(inputsDble, input, 0)
        clsMLPHelper.Fill2DArrayOfDoubleByArrayOfSingle(inputsDble, input, 0)
        Dim inputMatrix As Matrix = inputsDble
        Me.input = New Tensor(inputMatrix, autoGrad:=True)

    End Sub

    Private Sub SetInputAllSamples()

        Dim inputArrayDble#(,) = clsMLPHelper.Convert2DArrayOfSingleToDouble(Me.inputArray)
        Dim inputMatrix As Matrix = inputArrayDble
        Me.input = New Tensor(inputMatrix, autoGrad:=True)

    End Sub

    Private Sub SetTargetOneSample(target!())

        Dim targetsDble#(0, target.Length - 1)
        'targetsDble = clsMLPHelper.FillArray2(targetsDble, target, 0)
        clsMLPHelper.Fill2DArrayOfDoubleByArrayOfSingle(targetsDble, target, 0)
        Dim targetMatrix As Matrix = targetsDble
        Me.target = New Tensor(targetMatrix, autoGrad:=True)

    End Sub

    Private Sub SetTargetAllSamples()

        Dim targetArrayDble#(,) = clsMLPHelper.Convert2DArrayOfSingleToDouble(Me.targetArray)
        Dim targetMatrix As Matrix = targetArrayDble
        Me.target = New Tensor(targetMatrix, autoGrad:=True)

    End Sub

    Public Sub SetOuput1D()

        Dim output As Matrix = Me.pred.Data
        Me.lastOutputArray1DSingle = output.ToArraySingle()

    End Sub

    Private Sub SetOuputAllSamples()

        Me.output = Me.pred.Data

    End Sub

    Public Overrides Sub TrainVector()

        Me.vectorizedLearningMode = True
        For iteration = 0 To Me.nbIterations - 1
            TrainVectorOneIteration()
            If Me.printOutput_ Then PrintOutput(iteration)
        Next
        Me.output = Me.pred.Data

    End Sub

    Public Sub TrainVectorOneIteration()

        SetInputAllSamples()
        ForwardPropogateSignal()
        SetTargetAllSamples()
        ComputeError()
        BackwardPropagateError()

    End Sub

    Public Overrides Sub TestOneSample(input!())
        SetInputOneSample(input)
        ForwardPropogateSignal()
        SetOuput1D()
    End Sub

    Public Overrides Sub TrainOneSample(input!(), target!())

        TestOneSample(input)
        SetTargetOneSample(target)
        ComputeError()
        BackwardPropagateError()
        ComputeAverageErrorFromLastError()

    End Sub

    Private Sub ForwardPropogateSignal()
        Me.pred = Me.seq.Forward(Me.input)
    End Sub

    Public Sub BackwardPropagateError()
        Me.loss.Backward(New Tensor(Matrix.Ones(Me.loss.Data.r, Me.loss.Data.c)))
        Me.sgd.Step_()
    End Sub

    Public Overrides Sub ComputeError()
        ' Calculate the error: ERROR = TARGETS - OUTPUTS
        Me.loss = Me.mse.Forward(Me.pred, Me.target)
        Me.lastError = Me.loss.Data
    End Sub

    Public Function ComputeAverageErrorFromAllSamples!()
        ' Calculate the error: ERROR = TARGETS - OUTPUTS
        Dim m As Matrix = Me.targetArray
        Me.lastError = m - Me.output
        ComputeAverageErrorFromLastError()
        Return Me.averageError
    End Function

    Public Overrides Sub PrintWeights()

        Me.PrintParameters()

        For i = 0 To Me.layerCount - 1
            ShowMessage("Neuron count(" & i & ")=" & Me.neuronCount(i))
        Next

        ShowMessage("")

        Dim j = 0
        For Each w In Me.weights
            ShowMessage("W(" & j + 1 & ")=" & w.ToString & vbLf)
            j += 1
        Next

        ShowMessage("")

        j = 0
        For Each layer In Me.seq.Layers
            Dim k = 0
            For Each tensr In layer.Parameters
                Dim m As Matrix = tensr.Data
                ShowMessage("Layer(" & j + 1 & "," & k + 1 & ").W=" & m.ToString & vbLf)
                k += 1
            Next
            If layer.Parameters.Count > 0 Then j += 1
        Next

    End Sub

    Public Overrides Sub PrintOutput(iteration%)

        If ShowThisIteration(iteration) Then
            If Not Me.vectorizedLearningMode Then
                Dim nbTargets = Me.targetArray.GetLength(1)
                TestAllSamples(Me.inputArray, nbOutputs:=nbTargets)
            Else
                SetOuputAllSamples()
            End If
            ComputeAverageError()
            Dim sMsg$ = vbLf & "Iteration n°" & iteration + 1 & "/" & nbIterations & vbLf &
                "Output: " & Me.output.ToString() & vbLf &
                "Average error: " & Me.averageError.ToString(format6Dec)
            ShowMessage(sMsg)

            'ShowMessage("pred=" & Me.pred.ToString)
            'ShowMessage("loss=" & Me.loss.ToString)
            'ShowMessage("weights=" & Me.weights.ToString)

        End If

    End Sub

End Class
