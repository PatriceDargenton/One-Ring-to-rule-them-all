
' From: https://github.com/RutledgePaulV/multilayer-perceptron-vb

Imports Perceptron.Layers
Imports Perceptron.Activation
Imports Perceptron.Data
Imports Perceptron.Neurons
Imports Perceptron.Randoms
Imports Perceptron.Utility ' Matrix

Namespace NetworkOOP

    Public Class MultilayerPerceptron : Inherits clsMLPGeneric

        Public Property TotalError#

        'Public Property Momentum# -> weightAdjustment

        Public Property Bias As Neuron
        Public Property Randomizer As BaseRandom
        Public Property ActivationFunction As BaseActivation

        Public Property Layers As List(Of BaseLayer)

        Public Property InputLayer As InputLayer
        Public Property OutputLayer As OutputLayer
        Public Property HiddenLayers As List(Of HiddenLayer)

        Public Property Outputs As List(Of List(Of Double))

        Public Sub New()
            Dim standard As New Randoms.Standard(
                New Utilities.Range(-1, 1), seed:=DateTime.Now.Millisecond)
            Me.Randomizer = standard
        End Sub

        Public Sub New(learning_rate!, momentum!,
            randomizer As BaseRandom, activation As BaseActivation)

            'setting properties
            Me.weightAdjustment = momentum
            Me.Randomizer = randomizer
            Me.learningRate = learning_rate
            Me.ActivationFunction = activation

        End Sub

        Public Overrides Function GetActivationFunctionType() As enumActivationFunctionType
            Return enumActivationFunctionType.BothNormalAndSpecificCode
        End Function

        Public Overrides Sub InitializeStruct(neuronCount%(), addBiasColumn As Boolean)

            MyBase.InitializeStruct(neuronCount, addBiasColumn)

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
            Me.InputLayer = New InputLayer(Me.nbInputNeurons, Me.ActivationFunction)
            Me.Layers.Add(InputLayer)
            Dim numLayer = 0
            For Each i In neuronCount
                numLayer += 1
                If numLayer = 1 Then Continue For
                If numLayer = Me.layerCount Then Exit For
                Dim hiddenLayer = New HiddenLayer(i, Me.ActivationFunction)
                Me.HiddenLayers.Add(hiddenLayer)
                Me.Layers.Add(hiddenLayer)
            Next
            Me.OutputLayer = New OutputLayer(Me.nbOutputNeurons, Me.ActivationFunction)
            Me.Layers.Add(OutputLayer)
            Me.layerCount = Me.Layers.Count

            WeightInitStruct()

        End Sub

        Private Sub WeightInitStruct()

            'connecting layers (creating weights)
            For x = 0 To Me.Layers.Count - 2
                Me.Layers(x).ConnectChildInit(Layers(x + 1))

                'connecting bias
                If Me.useBias Then Me.Layers(x + 1).ConnectBiasInit(Bias)
            Next

        End Sub

        Public Overrides Sub InitializeWeights(layer%, weights#(,))

            Me.Layers(layer).RestoreWeightsWithBias(weights, Me.useBias, Me.Bias,
                Me.Layers(layer - 1))

        End Sub

        Public Overrides Sub Randomize(Optional minValue! = -0.5!, Optional maxValue! = 0.5!)

            For x = 0 To Me.Layers.Count - 2
                Me.Layers(x).InitChild(Layers(x + 1), Me.Randomizer)
                If Me.useBias Then Me.Layers(x + 1).InitBias(Me.Bias, Me.Randomizer)
            Next

        End Sub

        Public Sub TrainOneSampleOOP(data As List(Of Training))

            Me.Outputs = New List(Of List(Of Double))
            Me.TotalError = 0.0
            For Each item In data
                Me.InputLayer.SetInput(item.Input)
                ForwardPropogateSignal()
                Me.OutputLayer.AssignErrors(item.Output)
                BackwardPropogateErrorComputeGradientAndAdjustWeights()
                Me.TotalError += Me.OutputLayer.CalculateSquaredError()
                Me.Outputs.Add(Me.OutputLayer.ExtractOutputs)
            Next

        End Sub

        Public Sub TrainOneIteration(data As List(Of Training))

            Me.Outputs = New List(Of List(Of Double))
            Me.TotalError = 0.0
            For Each item In data
                Me.InputLayer.SetInput(item.Input)
                ForwardPropogateSignal()
                Me.OutputLayer.AssignErrors(item.Output)
                BackwardPropogateErrorComputeGradientAndAdjustWeights()
                Me.TotalError += Me.OutputLayer.CalculateSquaredError()
                Me.Outputs.Add(Me.OutputLayer.ExtractOutputs)
            Next

        End Sub

        Private Sub SetInputOneSample(input!())

            Dim inputDble#() = clsMLPHelper.Convert1DArrayOfSingleToDouble(input)
            Dim lst As List(Of Double) = inputDble.ToList
            Dim data As New Testing(lst)
            Me.InputLayer.SetInput(data.Input)

        End Sub

        Private Function SetInputAndTargetOneSample(input!(), target!()) As List(Of Training)

            Dim data As New List(Of Training)
            Dim inputDble#() = clsMLPHelper.Convert1DArrayOfSingleToDouble(input)
            Dim targetDble#() = clsMLPHelper.Convert1DArrayOfSingleToDouble(target)
            data.Add(New Training(inputDble, targetDble))
            Return data

        End Function

        Public Overrides Sub TrainOneSample(input!(), target!())

            Dim data = SetInputAndTargetOneSample(input, target)
            TrainOneSampleOOP(data)
            Me.averageError = CSng(Me.TotalError / target.GetLength(0))
            SetOuput1D()

        End Sub

        Public Sub SetOuput1D()
            Dim lst = Me.OutputLayer.ExtractOutputs()
            Me.lastOutputArray1D = lst.ToArray()
            Me.lastOutputArray1DSingle = clsMLPHelper.Convert1DArrayOfDoubleToSingle(Me.lastOutputArray1D)
        End Sub

        Private Sub SetOutput()
            ' 29/11/2020
            Dim outputs2D#(0, Me.nbOutputNeurons - 1)
            clsMLPHelper.Fill2DArrayOfDouble(outputs2D, Me.lastOutputArray1D, 0)
            Me.output = outputs2D
        End Sub

        Public Overrides Sub TestOneSample(input!())

            SetInputOneSample(input)
            ForwardPropogateSignal()
            SetOuput1D()
            SetOutput() ' 29/11/2020

        End Sub

        Private Sub ForwardPropogateSignal()

            For x = 1 To Me.Layers.Count - 1
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

        Private Sub BackwardPropogateErrorComputeGradientAndAdjustWeights()

            ' Backward propagate error from the output layer through to the first layer
            ' Gradient descend: Compute gradient and adjust weights

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
                        w.Value += adjustment '+ w.Previous * Me.weightAdjustment
                        If Me.weightAdjustment <> 0 Then _
                            w.Value += w.Previous * Me.weightAdjustment
                        w.Previous = adjustment
                    Next

                    'adjusting weights between bias
                    If Me.useBias Then
                        Dim biasAdjustment = Me.learningRate * node.ErrorDelta *
                            node.Primed * node.WeightToBias.Parent.Output
                        node.WeightToBias.Value += biasAdjustment '+
                        'node.WeightToBias.Previous * Me.weightAdjustment
                        If Me.weightAdjustment <> 0 Then _
                            node.WeightToBias.Value +=
                                node.WeightToBias.Previous * Me.weightAdjustment
                        node.WeightToBias.Previous = biasAdjustment
                    End If
                Next
            Next

        End Sub

        Public Overrides Sub PrintWeights()

            Me.PrintParameters()

            For i = 0 To Me.Layers.Count - 1
                ShowMessage("Neuron count(" & i & ")=" & Me.Layers(i).Neurons.Count)
            Next

            ShowMessage("")

            For i = 1 To Me.Layers.Count - 1
                ShowMessage("W(" & i & ")=" & Layers(i).PrintWeights())
            Next

        End Sub

        Public Function PrintOutputOOP$()

            Dim sb As New System.Text.StringBuilder("{" & vbCrLf)
            Dim nbOuputs = Me.Outputs.Count
            Dim numOuput = 0
            For Each outp In Me.Outputs
                Dim nbd = outp.Count
                Dim numd = 0
                sb.Append(" {")
                For Each ld In outp
                    sb.Append(ld.ToString(format2Dec).ReplaceCommaByDot())
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