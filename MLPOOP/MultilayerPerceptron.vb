
' From: https://github.com/RutledgePaulV/multilayer-perceptron-vb

Imports Perceptron.Layers
Imports Perceptron.Activation
Imports Perceptron.Data
Imports Perceptron.Neurons
Imports Perceptron.Randoms

Namespace NetworkOOP

    Public Class MultilayerPerceptron : Inherits clsMLPGeneric

        Private LastError As MatrixMLP.Matrix

        Public Property TotalError As Double

        'Public Property Momentum As Double -> weightAdjustment
        'Public Property LearningRate As Double

        Public Property Bias As Neuron
        Public Property Randomizer As BaseRandom
        Public Property ActivationFunction As BaseActivation

        Public Property Layers As List(Of BaseLayer)

        Public Property InputLayer As InputLayer
        Public Property OutputLayer As OutputLayer
        Public Property HiddenLayers As List(Of HiddenLayer)

        Public Property Outputs As List(Of List(Of Double))

        Public Sub New()
        End Sub

        Public Sub New(learning_rate As Single, momentum As Single,
            randomizer As BaseRandom, activation As BaseActivation)

            'setting properties
            Me.weightAdjustment = momentum
            Me.Randomizer = randomizer
            Me.learningRate = learning_rate
            Me.ActivationFunction = activation

        End Sub

        Public Overrides Sub InitStruct(neuronCount() As Integer, addBiasColumn As Boolean)

            Dim num_input% = neuronCount(0)
            Me.layerCount = neuronCount.GetLength(0)
            Dim num_output% = neuronCount(Me.layerCount - 1)

            Me.useBias = addBiasColumn
            If addBiasColumn Then
                'setting bias
                Me.Bias = New Neuron(NeuronType.Input)
                Me.Bias.Input = 1
                Me.Bias.Output = 1
            Else
                Me.Bias = Nothing
            End If

            'initializing lists
            Me.Layers = New List(Of BaseLayer)
            Me.HiddenLayers = New List(Of HiddenLayer)

            'creating layers
            Me.InputLayer = New InputLayer(num_input, Me.ActivationFunction)
            Me.Layers.Add(InputLayer)
            Dim numLayer% = 0
            For Each i In neuronCount
                numLayer += 1
                If numLayer = 1 Then Continue For
                If numLayer = Me.layerCount Then Exit For
                Dim hiddenLayer = New HiddenLayer(i, Me.ActivationFunction)
                Me.HiddenLayers.Add(hiddenLayer)
                Me.Layers.Add(hiddenLayer)
            Next
            Me.OutputLayer = New OutputLayer(num_output, Me.ActivationFunction)
            Me.Layers.Add(OutputLayer)
            Me.layerCount = Me.Layers.Count

            WeightInitStruct()

        End Sub

        Private Sub WeightInitStruct()

            'connecting layers (creating weights)
            For x As Integer = 0 To Me.Layers.Count - 2
                Me.Layers(x).ConnectChildInit(Layers(x + 1))

                'connecting bias
                If Me.useBias Then Me.Layers(x + 1).ConnectBiasInit(Bias)
            Next

        End Sub

        Public Overrides Sub WeightInit(layer As Integer, weights(,) As Double)

            Me.Layers(layer).RestoreWeightsWithBias(weights, Me.useBias, Me.Bias,
                Me.Layers(layer - 1))

        End Sub

        Public Overrides Sub Randomize(Optional minValue As Single = 0, Optional maxValue As Single = 1)

            For x As Integer = 0 To Me.Layers.Count - 2
                Me.Layers(x).InitChild(Layers(x + 1), Me.Randomizer)
                If Me.useBias Then Me.Layers(x + 1).InitBias(Me.Bias, Me.Randomizer)
            Next

        End Sub

        Public Sub TrainOneSampleOOP(data As List(Of Training), ByRef TotalError#)

            Me.Outputs = New List(Of List(Of Double))
            For Each item In data
                Me.InputLayer.SetInput(item.Input)
                ForwardPropogate()
                Me.OutputLayer.AssignErrors(item.Output)
                BackwardPropogate()
                TotalError += Me.OutputLayer.CalculateSquaredError()
                Me.Outputs.Add(Me.OutputLayer.ExtractOutputs)
            Next

        End Sub

        Public Sub TrainOneIteration(ByVal data As List(Of Training))

            Me.Outputs = New List(Of List(Of Double))
            TotalError = 0.0
            For Each item In data
                Me.InputLayer.SetInput(item.Input)
                ForwardPropogate()
                Me.OutputLayer.AssignErrors(item.Output)
                BackwardPropogate()
                TotalError += Me.OutputLayer.CalculateSquaredError()
                Me.Outputs.Add(Me.OutputLayer.ExtractOutputs)
            Next

        End Sub

        Public Overrides Sub TrainOneSample(input() As Single, target() As Single)

            Dim data As New List(Of Training)
            Dim inputDble#() = clsMLPHelper.ConvertSingleToDouble1D(input)
            Dim targetDble#() = clsMLPHelper.ConvertSingleToDouble1D(target)
            data.Add(New Training(inputDble, targetDble))
            Dim TotalError# = 0
            TrainOneSampleOOP(data, TotalError)
            Me.averageError = CSng(TotalError / target.GetLength(0))

            Dim lst = Me.OutputLayer.ExtractOutputs()
            Me.lastOutputArray = lst.ToArray()
            Me.lastOutputArraySingle = clsMLPHelper.ConvertDoubleToSingle(Me.lastOutputArray)

        End Sub

        Public Overrides Sub TestOneSample(input() As Single)

            Dim inputDble#() = clsMLPHelper.ConvertSingleToDouble1D(input)
            Dim lst As List(Of Double) = inputDble.ToList
            Dim data As New Testing(lst)
            Me.InputLayer.SetInput(data.Input)
            ForwardPropogate()

            Dim lstRes = Me.OutputLayer.ExtractOutputs()
            Me.lastOutputArray = lstRes.ToArray()
            Me.lastOutputArraySingle = clsMLPHelper.ConvertDoubleToSingle(Me.lastOutputArray)

        End Sub

        Private Sub ForwardPropogate()

            For x As Integer = 1 To Me.Layers.Count - 1
                For Each node In Me.Layers(x).Neurons
                    node.Input = 0.0
                    For Each w In node.WeightsToParent
                        node.Input += w.Parent.Output * w.Value
                    Next
                    'adding bias
                    If Me.useBias Then node.Input +=
                        node.WeightToBias.Parent.Output * node.WeightToBias.Value

                    If IsNothing(Me.lambdaFnc) AndAlso
                       IsNothing(Layers(x).ActivationFunction) Then
                        Throw New ArgumentException("Activation function undefined!")
                    End If

                    If IsNothing(Me.ActivationFunction) Then
                        ' Generic activation function
                        node.Output = Me.lambdaFnc.Invoke(node.Input)
                    Else
                        ' OOP activation function
                        node.Output = Me.Layers(x).ActivationFunction.Evaluate(node.Input)
                    End If
                Next
            Next

        End Sub

        Private Sub BackwardPropogate()

            'updating weights for all other layers
            For x = Me.Layers.Count - 1 To 1 Step -1
                For Each node In Me.Layers(x).Neurons

                    'if not output layer, then errors need to be backpropogated from child layer to parent
                    If node.Type <> NeuronType.Output Then
                        node.ErrorDelta = 0.0
                        For Each w In node.WeightsToChild
                            node.ErrorDelta += w.Value * w.Child.ErrorDelta * w.Child.Primed
                        Next
                    End If

                    'calculating derivative value of input
                    'node.Primed = Layers(x).ActivationFunction.AbstractedDerivative(node.Output)
                    If IsNothing(Me.ActivationFunction) Then
                        node.Primed = Me.lambdaFncD.Invoke(node.Input)
                    Else
                        node.Primed = Me.Layers(x).ActivationFunction.Derivative(node.Input)
                    End If

                    'adjusting weight values between parent layer
                    For Each w In node.WeightsToParent
                        Dim adjustment = Me.learningRate * node.ErrorDelta *
                            node.Primed * w.Parent.Output
                        w.Value += adjustment + w.Previous * Me.weightAdjustment
                        w.Previous = adjustment
                    Next

                    'adjusting weights between bias
                    If Me.useBias Then
                        Dim biasAdjustment = Me.learningRate * node.ErrorDelta *
                            node.Primed * node.WeightToBias.Parent.Output
                        node.WeightToBias.Value += biasAdjustment +
                            node.WeightToBias.Previous * Me.weightAdjustment
                        node.WeightToBias.Previous = biasAdjustment
                    End If
                Next
            Next

        End Sub

        Public Overrides Sub ComputeError()
            ' Calculate the error: ERROR = TARGETS - OUTPUTS
            Dim m As MatrixMLP.Matrix = Me.targetArray
            Dim output As MatrixMLP.Matrix = Me.outputArray
            Me.LastError = m - output
        End Sub

        Public Overrides Sub ComputeAverageErrorFromLastError()
            ' Compute first abs then average:
            Me.averageError = CSng(Me.LastError.Abs.Average)
        End Sub

        Public Overrides Function ComputeAverageError() As Single
            Me.ComputeError()
            Me.ComputeAverageErrorFromLastError()
            Return Me.averageError
        End Function

        Public Overrides Sub PrintWeights()

            Me.PrintParameters()

            For i As Integer = 0 To Me.Layers.Count - 1
                Dim iNeuronCount% = Me.Layers(i).Neurons.Count
                ShowMessage("Neuron count(" & i & ")=" & iNeuronCount)
            Next

            For i As Integer = 1 To Me.Layers.Count - 1
                ShowMessage("W(" & i & ")=" & Layers(i).PrintWeights())
            Next

        End Sub

        Public Overrides Sub PrintOutput(iteration As Integer)

            If ShowThisIteration(iteration) Then

                Dim nbTargets% = Me.targetArray.GetLength(1)
                TestAllSamples(Me.inputArray, nbTargets)
                ComputeAverageError()
                Dim outputMaxtrix As MatrixMLP.Matrix = Me.outputArraySingle
                Dim sMsg$ = vbLf & "Iteration n°" & iteration + 1 & "/" & nbIterations & vbLf &
                    "Output: " & outputMaxtrix.ToString() & vbLf &
                    "Average error: " & Me.averageError.ToString("0.000000")
                ShowMessage(sMsg)

            End If

        End Sub

        Public Function PrintOutputOOP$()

            Dim sb As New System.Text.StringBuilder("{" & vbCrLf)
            Dim nbOuputs% = Me.Outputs.Count
            Dim numOuput% = 0
            For Each outp In Me.Outputs
                Dim nbd% = outp.Count
                Dim numd% = 0
                sb.Append(" {")
                For Each ld In outp
                    sb.Append(ld.ToString("0.00").Replace(",", "."))
                    numd += 1
                    If numd < nbd Then sb.Append(", ")
                Next
                numOuput += 1
                If numOuput < nbOuputs Then sb.Append("}," & vbCrLf)
            Next
            sb.Append("}}")
            Return sb.ToString

        End Function

    End Class

End Namespace