
' From: https://github.com/nlabiris/perceptrons : C# -> VB .NET conversion

Imports Perceptron.Utility ' Matrix
Imports Perceptron.MLP.ActivationFunction
Imports System.Text ' StringBuilder

Namespace MatrixMLP

    ' Note: Me.weightAdjustment is not used in this implementation

    ''' <summary>
    ''' Multi-Layer Perceptron
    ''' </summary>
    Class MultiLayerPerceptron : Inherits clsMLPGeneric

        ''' <summary>
        ''' hidden x input weights matrix
        ''' </summary>
        Private weights_ih As Matrix

        ''' <summary>
        ''' ouput x hidden weights matrix
        ''' </summary>
        Private weights_ho As Matrix

        ''' <summary>
        ''' Hidden bias matrix
        ''' </summary>
        Private bias_h As Matrix

        ''' <summary>
        ''' Output bias matrix
        ''' </summary>
        Private bias_o As Matrix

        Private m_weights!()()
        Private m_biases!()()

        Private input, hidden As Matrix

        Private nbHiddenNeurons%

        Public Overrides Sub InitializeStruct(neuronCount%(), addBiasColumn As Boolean)

            MyBase.InitializeStruct(neuronCount, addBiasColumn)
            Me.nbHiddenNeurons = Me.neuronCount(1)

            Me.weightAdjustment = 0 ' Not used

            If Me.layerCount <> 3 Then
                ' ToDo: declare and use Me.weights_ih2 to compute 2 hidden layers
                Throw New ArgumentException(
                    "This Matrix implementation can only compute one hidden layer!")
                Me.layerCount = 3
            End If

            Me.nbOutputNeurons = neuronCount(Me.layerCount - 1)

            Dim dbleArray_ih#(Me.nbHiddenNeurons - 1, Me.nbInputNeurons - 1)
            Me.weights_ih = dbleArray_ih

            Dim dbleArray_ho#(Me.nbOutputNeurons - 1, Me.nbHiddenNeurons - 1)
            Me.weights_ho = dbleArray_ho

            Me.useBias = addBiasColumn
            If Me.useBias Then
                Dim dbleArray_bh#(Me.nbHiddenNeurons - 1, 0)
                Dim dbleArray_bo#(Me.nbOutputNeurons - 1, 0)
                Me.bias_h = dbleArray_bh
                Me.bias_o = dbleArray_bo
            End If

        End Sub

        ''' <summary>
        ''' Randomize weights
        ''' </summary>
        Public Overrides Sub Randomize(Optional minValue! = -0.5!, Optional maxValue! = 0.5!)

            Me.rnd = New Random()

            Me.weights_ih.Randomize(Me.rnd, minValue, maxValue)
            Me.weights_ho.Randomize(Me.rnd, minValue, maxValue)

            If Me.useBias Then
                Me.bias_h.Randomize(Me.rnd, minValue, maxValue)
                Me.bias_o.Randomize(Me.rnd, minValue, maxValue)
            End If

        End Sub

        Public Overrides Sub InitializeWeights(layer%, weights#(,))

            If layer = 1 Then
                ReDim Me.m_weights(0 To Me.layerCount - 2)
                ReDim Me.m_biases(0 To Me.layerCount - 2)
            End If

            Dim i = layer - 1
            Dim nbNeuronsLayer = Me.neuronCount(i + 1)
            Dim nbBiases = nbNeuronsLayer
            Dim n = nbBiases * Me.neuronCount(i)
            ReDim Me.m_weights(i)(0 To n - 1)
            ReDim Me.m_biases(i)(0 To nbBiases - 1)
            For j = 0 To nbBiases - 1
                Me.m_biases(i)(j) = 0
            Next j
            Dim nbNeuronsPreviousLayer = Me.neuronCount(i)
            Dim l = 0
            For j = 0 To nbNeuronsLayer - 1
                Dim nbWeights = nbNeuronsPreviousLayer
                For k = 0 To nbWeights - 1
                    Dim r = weights(j, k)
                    Me.m_weights(i)(l) = CSng(r)
                    l += 1
                Next k
                If Me.useBias Then
                    Dim r = weights(j, nbWeights)
                    Me.m_biases(i)(j) = CSng(r)
                End If
            Next j

            If layer = Me.layerCount - 1 Then

                Dim w1, w2, b1, b2 As Matrix

                Dim input = Me.nbInputNeurons
                Dim hidden = Me.neuronCount(1)
                Dim ouput = Me.nbOutputNeurons
                w1 = clsMLPHelper.TransformArrayTo2DArray(Me.m_weights(0), hidden, input)
                w2 = clsMLPHelper.TransformArrayTo2DArray(Me.m_weights(1), ouput, hidden)
                b1 = clsMLPHelper.TransformArrayTo2DArray(Me.m_biases(0), hidden, 1)
                b2 = clsMLPHelper.TransformArrayTo2DArray(Me.m_biases(1), ouput, 1)

                Me.weights_ih = w1
                Me.weights_ho = w2
                Me.bias_h = b1
                Me.bias_o = b2

            End If

        End Sub

        ''' <summary>
        ''' Test one sample
        ''' </summary>
        Public Overrides Sub TestOneSample(input!())
            ForwardPropogateSignal(input)
            Me.lastOutputArray1DSingle = Me.output.ToArrayOfSingle()
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
            If Me.useBias Then
                Me.output = Matrix.MultiplyAddAndMap(Me.weights_ho, Me.hidden, Me.bias_o, Me.lambdaFnc)
            Else
                Me.output = Matrix.MultiplyAndMap(Me.weights_ho, Me.hidden, Me.lambdaFnc)
            End If

        End Sub

        Private Sub BackwardPropagateError()

            ' Calculate gradient
            ' Calculate hidden -> output delta weights
            ' Adjust the weights by deltas
            ' Calculate the hidden layer errors
            ' Me.weightAdjustment is not used in this implementation
            BackwardPropagateErrorComputeGradientAndAdjustWeights(
                Me.output, Me.lastError, Me.hidden, Me.learningRate,
                Me.weights_ho, Me.bias_o)

            ' Calculate the hidden layer errors
            Dim hidden_errors = Matrix.TransposeAndMultiply1(Me.weights_ho, Me.lastError)

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
            ComputeErrorOneSampleSpecific(targetsArray)
            ComputeAverageErrorFromLastError()

            BackwardPropagateError()

        End Sub

        ''' <summary>
        ''' Gradient descend: Compute gradient and adjust weights
        ''' </summary>
        Public Sub BackwardPropagateErrorComputeGradientAndAdjustWeights(
            final As Matrix, error_ As Matrix, original As Matrix, adjustment!,
            ByRef weight As Matrix, ByRef bias As Matrix)

            ' Calculate gradient
            Dim gradient = Matrix.Map(final, lambdaFncDFOF)
            gradient *= error_
            gradient *= adjustment

            ' Calculate original -> final delta weights
            Dim weight_deltas = Matrix.TransposeAndMultiply2(original, gradient)

            ' Adjust the weights by deltas
            weight += weight_deltas

            ' Adjust the bias by its deltas (which is just the gradients)
            If Me.useBias Then bias += gradient

        End Sub

        ''' <summary>
        ''' Compute error from output and target matrices
        ''' </summary>
        Private Sub ComputeErrorOneSampleSpecific(target!())

            ' Calculate the error: ERROR = TARGETS - OUTPUTS
            Me.lastError = Matrix.SubtractFromArraySingle(target, Me.output)

        End Sub

        Public Overrides Sub TrainSystematic(inputs!(,), targets!(,),
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

            MyBase.TrainSystematic(inputs, targets, learningMode)

        End Sub

        Public Overrides Function ShowWeights$(Optional format$ = format2Dec)

            GetWeights()
            Dim weights = MyBase.ShowWeights(format)
            Return weights

        End Function

        Private Sub GetWeights()

            ReDim Me.m_weights(0 To Me.layerCount - 2)
            ReDim Me.m_biases(0 To Me.layerCount - 2)

            Dim w1 As Double(,) = Me.weights_ih
            Me.m_weights(0) = clsMLPHelper.Transform2DArrayDoubleToArraySingle(w1)

            Dim w2 As Double(,) = Me.weights_ho
            Me.m_weights(1) = clsMLPHelper.Transform2DArrayDoubleToArraySingle(w2)

            Dim w3 As Double(,) = Me.bias_h
            Me.m_biases(0) = clsMLPHelper.Transform2DArrayDoubleToArraySingle2(w3)

            Dim w4 As Double(,) = Me.bias_o
            Me.m_biases(1) = clsMLPHelper.Transform2DArrayDoubleToArraySingle2(w4)

        End Sub

        Public Overrides Function GetWeight!(layer%, neuron%, weight%)

            Dim nbNeuronsPreviousLayer = Me.neuronCount(layer - 1)
            If weight >= nbNeuronsPreviousLayer Then
                Dim l2% = weight - nbNeuronsPreviousLayer + neuron
                Dim bias_ = Me.m_biases(layer - 1)(l2)
                Return bias_
            End If
            Dim l% = neuron * nbNeuronsPreviousLayer + weight
            Dim weight_ = Me.m_weights(layer - 1)(l)
            Return weight_

        End Function

    End Class

End Namespace