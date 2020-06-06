
' From: https://github.com/nlabiris/perceptrons : C# -> VB .NET conversion

Imports Perceptron.MLP.ActivationFunction

Namespace MatrixMLP

    ' Note: Me.weightAdjustment is not used in this implementation

    ''' <summary>
    ''' Multi-Layer Perceptron
    ''' </summary>
    Class MultiLayerPerceptron : Inherits clsMLPGeneric

        ''' <summary>
        ''' hidden x input weights matrix
        ''' </summary>
        Public weights_ih As Matrix

        ''' <summary>
        ''' ouput x hidden weights matrix
        ''' </summary>
        Public weights_ho As Matrix

        ''' <summary>
        ''' Hidden bias matrix
        ''' </summary>
        Public bias_h As Matrix

        ''' <summary>
        ''' Output bias matrix
        ''' </summary>
        Public bias_o As Matrix

        Private input, hidden As Matrix

        ''' <summary>
        ''' Output matrix (returned to compute average error, and discrete error)
        ''' </summary>
        Public output As Matrix

        ''' <summary>
        ''' Last error of the output matrix
        ''' </summary>
        Private lastError_ As Matrix

        Public Overrides Sub InitializeStruct(neuronCount%(), addBiasColumn As Boolean)

            Dim inputNodes = neuronCount(0)
            Dim hiddenNodes = neuronCount(1)
            Me.layerCount = neuronCount.Length

            If Me.layerCount <> 3 Then
                ' ToDo: declare and use Me.weights_ih2 to compute 2 hidden layers
                MsgBox("This Matrix implementation can only compute one hidden layer!",
                    MsgBoxStyle.Exclamation)
                Me.layerCount = 3
            End If

            Dim outputNodes = neuronCount(Me.layerCount - 1)
            Me.weights_ih = New Matrix(hiddenNodes, inputNodes)
            Me.weights_ho = New Matrix(outputNodes, hiddenNodes)

            Me.useBias = addBiasColumn
            If Me.useBias Then
                Me.bias_h = New Matrix(hiddenNodes, 1)
                Me.bias_o = New Matrix(outputNodes, 1)
            End If

        End Sub

        Public Sub SetActivationFunctionForMatrix(fctAct As TActivationFunctionForMatrix,
            gain!, center!)

            Select Case fctAct
                Case TActivationFunctionForMatrix.Sigmoid
                    Me.m_actFunc = TActivationFunction.Sigmoid
                    Me.activFnc = New SigmoidFunction
                    If gain <> 1.0! Then MsgBox(
                        "gain must be 1 for Sigmoid activation function for Matrix",
                        MsgBoxStyle.Exclamation)
                Case TActivationFunctionForMatrix.HyperbolicTangent
                    Me.m_actFunc = TActivationFunction.HyperbolicTangent
                    Me.activFnc = New HyperbolicTangentFunction
                    If gain <> 1.0! Then MsgBox(
                        "gain must be 1 for hyperbolic tangent activation function for Matrix",
                        MsgBoxStyle.Exclamation)
                Case TActivationFunctionForMatrix.ELU
                    Me.m_actFunc = TActivationFunction.ELU
                    ' gain <> 1 is possible
                    Me.activFnc = New ELUFunction
                Case Else
                    Me.activFnc = Nothing
            End Select

            Me.lambdaFnc = Function(x#) Me.activFnc.Activation(x, gain, center)
            Me.lambdaFncD = Function(x#) Me.activFnc.DerivativeFromOriginalFunction(x, gain)

            ' Matrix implementation requires activation function expressed from 
            '  its direct function: f'(x)=g(f(x))
            If Not IsNothing(Me.activFnc) AndAlso
               Not Me.activFnc.DoesDerivativeDependOnOriginalFunction() Then _
                MsgBox("Activation function must be like this form: f'(x)=g(f(x))",
                    MsgBoxStyle.Exclamation)

        End Sub

        ''' <summary>
        ''' Randomize weights
        ''' </summary>
        Public Overrides Sub Randomize(Optional minValue! = 0, Optional maxValue! = 1)

            Me.weights_ih.Randomize(minValue, maxValue)
            Me.weights_ho.Randomize(minValue, maxValue)

            If Me.useBias Then
                Me.bias_h.Randomize(minValue, maxValue)
                Me.bias_o.Randomize(minValue, maxValue)
            End If

        End Sub

        Public Overrides Sub InitializeWeights(layer%, weights#(,))
            Throw New NotImplementedException(
                "InitializeWeights() is not implemented for clsMatrixMLP!")
        End Sub

        ''' <summary>
        ''' Test one sample
        ''' </summary>
        Public Overrides Sub TestOneSample(input!())
            ForwardPropogateSignal(input)
            Me.lastOutputArray1DSingle = Me.output.ToVectorArraySingle()
        End Sub

        Public Overrides Sub TestOneSample(input!(), ByRef ouput!())
            TestOneSample(input)
            ouput = Me.lastOutputArray1DSingle
        End Sub

        ''' <summary>
        ''' Propagate the input signal into the MLP
        ''' </summary>
        Private Sub ForwardPropogateSignal(inputsArray!())

            ' Generating the Hidden Outputs
            Me.input = Matrix.FromArraySingle(inputsArray)
            If Me.useBias Then
                Me.hidden = Matrix.MultiplyAddAndMap(Me.weights_ih, Me.input, Me.bias_h, Me.lambdaFnc)
            Else
                Me.hidden = Matrix.MultiplyAndMap(Me.weights_ih, Me.input, Me.lambdaFnc)
            End If

            ' Generating the output's output!
            Dim output As Matrix
            If Me.useBias Then
                output = Matrix.MultiplyAddAndMap(Me.weights_ho, Me.hidden, Me.bias_o, Me.lambdaFnc)
            Else
                output = Matrix.MultiplyAndMap(Me.weights_ho, Me.hidden, Me.lambdaFnc)
            End If
            Me.output = output

        End Sub

        Private Sub BackwardPropagateError()

            ' Calculate gradient
            ' Calculate hidden -> output delta weights
            ' Adjust the weights by deltas
            ' Calculate the hidden layer errors
            ' Me.weightAdjustment is not used in this implementation
            BackwardPropagateErrorComputeGradientAndAdjustWeights(
                Me.output, Me.lastError_, Me.hidden, Me.learningRate,
                Me.weights_ho, Me.bias_o)

            ' Calculate the hidden layer errors
            Dim hidden_errors = Matrix.TransposeAndMultiply1(Me.weights_ho, Me.lastError_)

            ' Calculate hidden gradient
            ' Calculate input -> hidden delta weights
            ' Adjust the bias by its deltas (which is just the gradients)
            BackwardPropagateErrorComputeGradientAndAdjustWeights(
                Me.hidden, hidden_errors, Me.input, Me.learningRate,
                Me.weights_ih, Me.bias_h)

        End Sub

        ''' <summary>
        ''' Train MLP with one sample
        ''' </summary>
        Public Overrides Sub TrainOneSample(inputsArray!(), targetsArray!())

            ForwardPropogateSignal(inputsArray)

            ' Calculate the error: ERROR = TARGETS - OUTPUTS
            ComputeErrorOneSample(targetsArray)
            ComputeAverageErrorFromLastError()

            BackwardPropagateError()

        End Sub

        ''' <summary>
        ''' Gradient descend: Compute gradient and adjust weights
        ''' </summary>
        Public Sub BackwardPropagateErrorComputeGradientAndAdjustWeights(
            final As Matrix, error_ As Matrix, original As Matrix,
            adjustment!, ByRef weight As Matrix, ByRef bias As Matrix)

            ' Calculate gradient
            Dim gradient = Matrix.Map(final, lambdaFncD)
            gradient.Multiply(error_)
            gradient.Multiply(adjustment)

            ' Calculate original -> final delta weights
            Dim weight_deltas = Matrix.TransposeAndMultiply2(original, gradient)

            ' Adjust the weights by deltas
            weight.Add(weight_deltas)

            ' Adjust the bias by its deltas (which is just the gradients)
            If Me.useBias Then bias.Add(gradient)

        End Sub

        ''' <summary>
        ''' Compute error from output and target matrices
        ''' </summary>
        Public Sub ComputeErrorOneSample(target!())

            ' Calculate the error: ERROR = TARGETS - OUTPUTS
            Me.lastError_ = Matrix.SubtractFromArraySingle(target, Me.output)

        End Sub

        Public Overrides Sub ComputeError()
            ' Calculate the error: ERROR = TARGETS - OUTPUTS
            Dim m As Matrix = Me.targetArray
            Me.lastError_ = m - Me.output
        End Sub

        Public Overrides Sub ComputeAverageErrorFromLastError()

            ' Compute first abs then average:
            'Me.averageError = CSng(Matrix.Abs(Me.lastError_).Average)
            Me.averageError = CSng(Me.lastError_.Abs.Average)

        End Sub

        Public Overrides Sub TrainSystematic(inputs!(,), targets!(,),
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

            MyBase.TrainSystematic(inputs, targets, learningMode)
            Me.output = Me.outputArray

        End Sub

        ''' <summary>
        ''' Print weights for functionnal test
        ''' </summary>
        Public Overrides Sub PrintWeights()

            Me.PrintParameters()

            Dim inputNodes = Me.weights_ih.Rows
            Dim hiddenNodes = Me.weights_ih.Cols
            Dim outputNodes = Me.weights_ho.Rows
            For i = 0 To Me.layerCount - 1
                Dim iNeuronCount = inputNodes
                If i > 0 Then iNeuronCount = hiddenNodes
                If i >= Me.layerCount - 1 Then iNeuronCount = outputNodes
                ShowMessage("Neuron count(" & i & ")=" & iNeuronCount)
            Next

            ShowMessage("")
            ShowMessage("Me.weights_ih=" & Me.weights_ih.ToString())
            ShowMessage("Me.weights_ho=" & Me.weights_ho.ToString())

            If Me.useBias Then
                ShowMessage("Me.bias_h=" & Me.bias_h.ToString())
                ShowMessage("Me.bias_o=" & Me.bias_o.ToString())
            End If

        End Sub

        Public Overrides Sub PrintOutput(iteration%)

            If ShowThisIteration(iteration) Then

                Dim nbTargets = Me.targetArray.GetLength(1)
                TestAllSamples(Me.inputArray, nbTargets)
                Me.output = Me.outputArray
                ComputeAverageError()
                Dim sMsg$ = vbLf & "Iteration n°" & iteration + 1 & "/" & nbIterations & vbLf &
                    "Output: " & Me.output.ToString() & vbLf &
                    "Average error: " & Me.averageError.ToString(format6Dec)
                ShowMessage(sMsg)

            End If

        End Sub

    End Class

End Namespace