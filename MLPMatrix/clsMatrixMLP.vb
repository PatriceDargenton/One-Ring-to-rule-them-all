
' From: https://github.com/nlabiris/perceptrons : C# -> VB .NET conversion

Option Infer On ' Lambda function

Imports Perceptron.MLP.ActivationFunction

Namespace MatrixMLP

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

        ''' <summary>
        ''' Output matrix (returned to compute average error, and discrete error)
        ''' </summary>
        Public output As Matrix

        ''' <summary>
        ''' Last error of the output matrix
        ''' </summary>
        Public lastError As Matrix

        ''' <summary>
        ''' Constructor
        ''' </summary>
        Public Sub New()
        End Sub

        Public Overrides Sub InitStruct(neuronCount%(), addBiasColumn As Boolean)

            Dim inputNodes% = neuronCount(0)
            Dim hiddenNodes% = neuronCount(1)
            Me.layerCount = neuronCount.Length
            Dim outputNodes% = neuronCount(Me.layerCount - 1)
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
                    Me.activFnc = New SigmoidFunction
                Case TActivationFunctionForMatrix.HyperbolicTangent
                    Me.activFnc = New HyperbolicTangentFunction
                Case TActivationFunctionForMatrix.ELU
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
                MsgBox("Activation function must be like this form: f'(x)=g(f(x))" &
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

        Public Overrides Sub WeightInit(layer%, weights#(,))
            Throw New NotImplementedException("WeightInit() is not implemented for clsMatrixMLP!")
        End Sub

        ''' <summary>
        ''' Print weights for functionnal test
        ''' </summary>
        Public Overrides Sub PrintWeights()

            Me.PrintParameters()

            Dim inputNodes% = Me.weights_ih.Rows
            Dim hiddenNodes% = Me.weights_ih.Cols
            Dim outputNodes% = Me.weights_ho.Rows
            For i As Integer = 0 To Me.layerCount - 1
                Dim iNeuronCount% = inputNodes
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

        ''' <summary>
        ''' Propagate the input signal into the MLP
        ''' </summary>
        Public Function FeedForward(inputsArray!()) As Single()

            ' Generating the Hidden Outputs
            Dim inputs = Matrix.FromArraySingle(inputsArray)
            Dim hidden As Matrix
            If Me.useBias Then
                hidden = Matrix.MultiplyAddAndMap(Me.weights_ih, inputs, Me.bias_h, Me.lambdaFnc)
            Else
                hidden = Matrix.MultiplyAndMap(Me.weights_ih, inputs, Me.lambdaFnc)
            End If

            ' Generating the output's output!
            Dim output As Matrix
            If Me.useBias Then
                output = Matrix.MultiplyAddAndMap(Me.weights_ho, hidden, Me.bias_o, Me.lambdaFnc)
            Else
                output = Matrix.MultiplyAndMap(Me.weights_ho, hidden, Me.lambdaFnc)
            End If
            Me.output = output

            Dim aSng = output.ToVectorArraySingle()
            Return aSng

        End Function

        ''' <summary>
        ''' Train MLP with one sample
        ''' </summary>
        Public Overrides Sub TrainOneSample(input!(), target!())

            Train_internal(input, target,
                backwardLearningRate:=Me.weightAdjustment,
                forewardLearningRate:=Me.learningRate)

        End Sub

        Private Sub Train_internal(inputsArray!(), targetsArray!(),
            backwardLearningRate!, forewardLearningRate!)

            Dim inputs = Matrix.FromArraySingle(inputsArray)

            ' Generating the Hidden Outputs
            Dim hidden As Matrix
            If Me.useBias Then
                hidden = Matrix.MultiplyAddAndMap(Me.weights_ih, inputs, Me.bias_h, Me.lambdaFnc)
            Else
                hidden = Matrix.MultiplyAndMap(Me.weights_ih, inputs, Me.lambdaFnc)
            End If

            ' Generating the output's output!
            Dim outputs As Matrix
            If Me.useBias Then
                outputs = Matrix.MultiplyAddAndMap(Me.weights_ho, hidden, Me.bias_o, Me.lambdaFnc)
            Else
                outputs = Matrix.MultiplyAndMap(Me.weights_ho, hidden, Me.lambdaFnc)
            End If
            Me.output = outputs

            ' Calculate the error: ERROR = TARGETS - OUTPUTS
            ComputeErrorOneSample(targetsArray)

            ' Calculate gradient
            ' Calculate hidden -> output delta weights
            ' Adjust the weights by deltas
            ' Calculate the hidden layer errors
            ComputeGradient(outputs, Me.lastError, hidden, backwardLearningRate,
                Me.weights_ho, Me.bias_o)

            ' Calculate the hidden layer errors
            Dim hidden_errors = Matrix.TransposeAndMultiply1(Me.weights_ho, Me.lastError)

            ' Calculate hidden gradient
            ' Calculate input -> hidden delta weights
            ' Adjust the bias by its deltas (which is just the gradients)
            ComputeGradient(hidden, hidden_errors, inputs, forewardLearningRate,
                Me.weights_ih, Me.bias_h)

        End Sub

        ''' <summary>
        ''' Compute gradient and return weight and bias matrices
        ''' </summary>
        Public Sub ComputeGradient(final As Matrix, error_ As Matrix, original As Matrix,
            learningRate!, ByRef weight As Matrix, ByRef bias As Matrix)

            ' Calculate gradient
            Dim gradient = Matrix.Map(final, lambdaFncD)
            gradient.Multiply(error_)
            gradient.Multiply(learningRate)

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
            Me.lastError = Matrix.SubtractFromArraySingle(target, Me.output)

        End Sub

        Public Overrides Sub ComputeError()
            ' Calculate the error: ERROR = TARGETS - OUTPUTS
            Dim m As Matrix = Me.targetArray
            Me.lastError = m - Me.output
        End Sub

        Public Overrides Sub ComputeAverageErrorFromLastError()

            ' Compute first abs then average:
            'Me.averageError = CSng(Matrix.Abs(Me.lastError).Average)
            Me.averageError = CSng(Me.lastError.Abs.Average)

        End Sub

        Public Overrides Function ComputeAverageError!()

            Me.ComputeError()
            Me.ComputeAverageErrorFromLastError()
            Return Me.averageError

        End Function

        Public Overrides Sub TrainSystematic(inputs!(,), targets!(,),
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

            MyBase.TrainSystematic(inputs, targets, learningMode)
            Me.output = Me.outputArray

        End Sub

        Public Overrides Sub PrintOutput(iteration%)
            If ShowThisIteration(iteration) Then
                ComputeAverageErrorFromLastError()
                Dim nbTargets% = Me.targetArray.GetLength(1)
                TestAllSamples(Me.inputArray, nbTargets)
                Me.output = Me.outputArray
                Dim sMsg$ = vbLf & "Iteration n°" & iteration + 1 & "/" & nbIterations & vbLf &
                    "Output: " & Me.output.ToString() & vbLf &
                    "Average error: " & Me.averageError.ToString("0.000000")
                ShowMessage(sMsg)
            End If
        End Sub

        ''' <summary>
        ''' Test one sample
        ''' </summary>
        Public Overrides Sub TestOneSample(input!())
            Me.lastOutputArraySingle = Me.FeedForward(input)
        End Sub

    End Class

End Namespace