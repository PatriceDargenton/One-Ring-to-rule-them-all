
Imports System.Runtime.InteropServices ' OutAttribute <Out>
Imports Perceptron.Utility ' Matrix

Namespace VectorizedMatrixMLP

    Public Class clsVectorizedMatrixMLP : Inherits clsVectorizedMLPGeneric

        Private neuronCountWithBias%()
        Private m_input, m_target, Zlast As Matrix
        Private w, error_, Z, A, delta As Matrix()

        Public Property input As Matrix
            Get
                Return Me.m_input
            End Get
            Set(value As Matrix)
                Me.m_input = value
            End Set
        End Property

        Public Property target As Matrix
            Get
                Return Me.m_target
            End Get
            Set(value As Matrix)
                Me.m_target = value
            End Set
        End Property

        Public Overrides Sub InitializeStruct(neuronCount%(), addBiasColumn As Boolean)

            MyBase.InitializeStruct(neuronCount, addBiasColumn)

            Me.input = Me.inputArray
            Me.target = Me.targetArray

            ReDim Me.neuronCount(0 To Me.layerCount - 1)
            ReDim Me.neuronCountWithBias(0 To Me.layerCount - 1)
            For i = 0 To Me.layerCount - 1
                Me.neuronCount(i) = neuronCount(i)
                Me.neuronCountWithBias(i) = neuronCount(i)
                If Me.useBias AndAlso i > 0 AndAlso i < Me.layerCount - 1 Then _
                    Me.neuronCountWithBias(i) += 1
            Next
            Me.exampleCount = Me.target.r
            Me.w = New Matrix(Me.layerCount - 1 - 1) {}

        End Sub

        Public Overrides Sub Randomize(Optional minValue! = -0.5!, Optional maxValue! = 0.5!)

            'Me.rnd = New Random(Seed:=1)
            Me.rnd = New Random()

            For i = 0 To Me.w.Length - 1
                Dim nbNeurons = Me.neuronCountWithBias(i)
                If Me.useBias Then nbNeurons += 1
                Dim nbNeuronsNextLayer = Me.neuronCountWithBias(i + 1)
                Me.w(i) = Matrix.Randomize(
                    nbNeurons, nbNeuronsNextLayer, Me.rnd, minValue, maxValue) * 2 - 1
                'Me.w(i).Randomize(minValue, maxValue)
            Next

        End Sub

        Public Overrides Sub InitializeWeights(layer%, weights#(,))
            Me.w(layer - 1) = weights
        End Sub

        Public Overrides Sub TrainVector()

            Me.learningMode = enumLearningMode.Vectorial
            Me.vectorizedLearningMode = True
            For iteration = 0 To Me.nbIterations - 1
                TrainVectorOneIteration()
                If Me.printOutput_ Then PrintOutput(iteration)
            Next
            ComputeAverageError()

        End Sub

        Public Sub ForwardPropagateSignal()

            Me.error_ = Nothing

            ForwardPropagateSignalInternal()

            Dim maxLayer = layerCount - 1
            Dim maxIndex = Me.A.Length - 1
            Me.Zlast = Me.Z(maxLayer)
            ' Cut first column for last layer
            Dim zr = Me.Z(maxLayer).r
            Dim zc = Me.Z(maxLayer).c
            If Me.useBias Then Zlast = Zlast.Slice(0, 1, zr, zc)

            Me.output = Me.A(maxIndex)
            ' Cut first column for last index of result matrix
            Dim ar = Me.A(maxIndex).r
            Dim ac = Me.A(maxIndex).c
            If Me.useBias Then Me.output = Me.output.Slice(0, 1, ar, ac)

        End Sub

        Public Sub ComputeErrorInternal()
            Me.error_ = New Matrix(Me.layerCount - 1) {}
            Me.error_(Me.layerCount - 1) = Me.output - Me.target
        End Sub

        Public Sub BackwardPropagateError()
            ComputeErrorInternal()
            BackwardPropagateErrorInternal()
            ComputeGradientAndAdjustWeights()
        End Sub

        Public Overrides Sub TrainVectorOneIteration()
            ForwardPropagateSignal()
            BackwardPropagateError()
        End Sub

        Private Sub ForwardPropagateSignalInternal()

            Me.Z = New Matrix(layerCount - 1) {}
            Me.A = New Matrix(layerCount - 1) {}

            Me.Z(0) = Me.m_input
            ' Column added with 1 for all examples
            If Me.useBias Then Me.Z(0) = Me.Z(0).AddColumn(Matrix.Ones(Me.exampleCount, 1))
            Me.A(0) = Me.Z(0)

            For i = 1 To layerCount - 1
                Me.Z(i) = Me.A(i - 1) * Me.w(i - 1)
                ' Column added with 1 for all examples
                If Me.useBias Then Me.Z(i) = Me.Z(i).AddColumn(Matrix.Ones(Me.exampleCount, 1))
                Me.A(i) = Matrix.Map(Me.Z(i), Me.lambdaFnc)
            Next

            ' How use Relu
            ' Change all sigmoid function, for relu function
            ' Last A must have no Nonlinear function Matrix, Last A must be Equal To Last Z;
            '  because of that Last Delta has not derivated Matrix "Last Delta = Last error Error * 1";
            ' The learning rate must be smaller, like 0.001
            ' Optionaly you can use a Softmax layer to make a classifier
            ' Use if Relu OR iregularized Values
            If Me.activFnc.IsNonLinear Then Me.A(Me.A.Length - 1) = Me.Z(Me.Z.Length - 1)

        End Sub

        Private Sub BackwardPropagateErrorInternal()

            Me.delta = New Matrix(Me.layerCount - 1) {}
            Me.delta(Me.layerCount - 1) = Me.error_(Me.layerCount - 1) *
                Matrix.Map(Zlast, Me.lambdaFncD)

            For i = Me.layerCount - 2 To 0 Step -1

                Dim d = Me.delta(i + 1)
                Dim t = Me.w(i).T
                Me.error_(i) = d * t

                Me.delta(i) = Me.error_(i) * Matrix.Map(Z(i), Me.lambdaFncD)

                ' Cut first column
                If Me.useBias Then Me.delta(i) = Me.delta(i).Slice(0, 1,
                    Me.delta(i).r, Me.delta(i).c)

            Next

        End Sub

        Private Sub ComputeGradientAndAdjustWeights()

            ' Gradient descend: Compute gradient and adjust weights

            For i = 0 To w.Length - 1
                Dim gradient = Me.A(i).T * Me.delta(i + 1)
                Me.w(i) -= gradient * Me.learningRate
                ' 30/05/2020 weightAdjustment
                If Me.weightAdjustment <> 0 Then Me.w(i) -= gradient * Me.weightAdjustment
            Next

        End Sub

        Public Sub SetLastError()
            Me.lastError = Me.error_(Me.layerCount - 1)
        End Sub

        Public Overrides Sub TrainOneSample(input!(), target!())
            SetInputOneSample(input)
            SetTargetOneSample(target)
            Me.exampleCount = 1
            TrainVectorOneIteration()
        End Sub

        Private Sub SetInputOneSample(input!())
            Dim inputDble#(0, input.Length - 1)
            clsMLPHelper.Fill2DArrayOfDoubleByArrayOfSingle(inputDble, input, 0)
            Dim matrixInput As Matrix = inputDble
            Me.input = matrixInput
        End Sub

        Private Sub SetTargetOneSample(target!())
            Dim targetsDble#(0, target.Length - 1)
            clsMLPHelper.Fill2DArrayOfDoubleByArrayOfSingle(targetsDble, target, 0)
            Me.target = targetsDble
        End Sub

        Public Overrides Sub SetOuput1D()
            Me.lastOutputArray1DSingle = Me.output.ToArraySingle()
        End Sub

        Public Overrides Sub TestOneSample(input!())

            ' Resize output to one sample
            If Me.exampleCount > 1 Then
                Me.exampleCount = 1
                Dim nbOutputs% = Me.neuronCount(Me.layerCount - 1)
                Dim target!(nbOutputs - 1)
                SetTargetOneSample(target)
            End If

            SetInputOneSample(input)
            ForwardPropagateSignal()
            SetOuput1D()

        End Sub

        Public Overrides Function ShowWeights$()

            Dim sb As New System.Text.StringBuilder
            sb.Append(Me.ShowParameters())

            For i = 0 To Me.layerCount - 1
                sb.AppendLine("Neuron count(" & i & ")=" & Me.neuronCount(i))
            Next

            sb.AppendLine("")

            For i = 0 To Me.w.Length - 1
                sb.AppendLine("W(" & i + 1 & ")=" & Me.w(i).ToString & vbLf)
            Next

            Return sb.ToString()

        End Function

        Public Overrides Sub PrintOutput(iteration%, Optional force As Boolean = False)

            If force OrElse ShowThisIteration(iteration) Then

                If Not Me.vectorizedLearningMode Then
                    'Dim nbTargets = Me.targetArray.GetLength(1)
                    TestAllSamples(Me.inputArray) ', nbOutputs:=nbTargets)
                Else
                    ComputeAverageError()
                End If
                PrintSuccess(iteration)

                'For i = 0 To Me.LayerCount - 1
                '    msg &= "Error(" & i & ")=" & Me.error_(i).ToString() & vbLf
                '    msg &= "A(" & i & ")=" & A(i).ToString() & vbLf
                'Next

            End If

        End Sub

    End Class

End Namespace