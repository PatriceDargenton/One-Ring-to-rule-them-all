
Imports System.Runtime.InteropServices ' OutAttribute <Out>
Imports Perceptron.Utility ' Matrix

Namespace VectorizedMatrixMLP

    Public Class clsVectorizedMatrixMLP : Inherits clsVectorizedMLPGeneric

        Public target As Matrix

        Private neuronCountWithBias%()
        Private input, Zlast As Matrix
        Private w, error_, Z, A, delta As Matrix()

        Public Overrides Sub InitializeStruct(neuronCount%(), addBiasColumn As Boolean)

            Me.input = Me.inputArray
            Me.target = Me.targetArray

            Me.useBias = addBiasColumn
            Me.layerCount = neuronCount.Length
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

        Public Overrides Sub Randomize(Optional minValue! = 0, Optional maxValue! = 1)

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

            Me.vectorizedLearningMode = True
            For iteration = 0 To Me.nbIterations - 1
                TrainAllSamplesInternal()
                If Me.printOutput_ Then PrintOutput(iteration)
            Next

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

            'Me.outputMatrix = Me.A(maxIndex)
            Me.output = Me.A(maxIndex)
            ' Cut first column for last index of result matrix
            Dim ar = Me.A(maxIndex).r
            Dim ac = Me.A(maxIndex).c
            'If Me.useBias Then Me.outputMatrix = Me.outputMatrix.Slice(0, 1, ar, ac)
            If Me.useBias Then Me.output = Me.output.Slice(0, 1, ar, ac)

        End Sub

        Public Sub ComputeErrorInternal()
            Me.error_ = New Matrix(Me.layerCount - 1) {}
            'Me.error_(Me.layerCount - 1) = Me.outputMatrix - Me.target
            Me.error_(Me.layerCount - 1) = Me.output - Me.target
        End Sub

        Public Sub BackwardPropagateError()
            ComputeErrorInternal()
            BackwardPropagateErrorInternal()
            ComputeGradientAndAdjustWeights()
        End Sub

        Public Sub TrainAllSamplesInternal()
            ForwardPropagateSignal()
            BackwardPropagateError()
        End Sub

        Private Sub ForwardPropagateSignalInternal()

            Me.Z = New Matrix(layerCount - 1) {}
            Me.A = New Matrix(layerCount - 1) {}

            Me.Z(0) = Me.input
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
                If Me.useBias Then Me.delta(i) = Me.delta(i).Slice(0, 1, Me.delta(i).r, Me.delta(i).c)

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

        Public Overrides Sub ComputeError()
            ' Calculate the error: ERROR = TARGETS - OUTPUTS
            Dim m As Matrix = Me.targetArray
            'Me.lastError = m - Me.outputMatrix
            Me.lastError = m - Me.output
        End Sub

        Public Sub SetLastError()
            Me.lastError = Me.error_(Me.layerCount - 1)
        End Sub

        Public Overrides Sub ComputeAverageErrorFromLastError()
            ' Compute first abs then average:
            Me.averageError = CSng(Me.lastError.Abs.Average * Me.exampleCount)
        End Sub

        Public Overrides Function ComputeAverageError!()
            MyBase.ComputeAverageError()
            Return Me.averageError
        End Function

        Public Sub ComputeErrorOneSample()
            ' Calculate the error: ERROR = TARGETS - OUTPUTS
            Dim m As Matrix = Me.targetArray
            Me.lastError = (m - Me.output).GetRow(0)
        End Sub

        Public Sub ComputeAverageErrorFromLastErrorOneSample()
            ' Compute first abs then average:
            Me.averageError = CSng(Me.lastError.Abs.Average)
        End Sub

        Public Function ComputeAverageErrorOneSample!()
            Me.ComputeErrorOneSample()
            Me.ComputeAverageErrorFromLastErrorOneSample()
            Return Me.averageError
        End Function

        Public Overrides Sub TrainOneSample(input!(), target!())
            SetInputOneSample(input)
            SetTargetOneSample(target)
            Me.exampleCount = 1
            TrainAllSamplesInternal()
        End Sub

        Private Sub SetInputOneSample(input!())
            Dim inputDble#(0, input.Length - 1)
            inputDble = clsMLPHelper.FillArray2(inputDble, input, 0)
            Dim matrixInput As Matrix = inputDble
            Me.input = matrixInput
        End Sub

        Private Sub SetTargetOneSample(target!())
            Dim targetsDble#(0, target.Length - 1)
            targetsDble = clsMLPHelper.FillArray2(targetsDble, target, 0)
            Me.target = targetsDble
        End Sub

        Public Sub SetOuput1D()
            'Me.lastOutputArray1DSingle = Me.outputMatrix.ToArraySingle()
            Me.lastOutputArray1DSingle = Me.output.ToArraySingle()
        End Sub

        Public Overrides Sub TestOneSample(input!())
            SetInputOneSample(input)
            ForwardPropagateSignal()
            SetOuput1D()
        End Sub

        Public Overrides Sub PrintWeights()

            Me.PrintParameters()

            For i = 0 To Me.layerCount - 1
                ShowMessage("Neuron count(" & i & ")=" & Me.neuronCount(i))
            Next

            ShowMessage("")

            For i = 0 To Me.w.Length - 1
                ShowMessage("W(" & i + 1 & ")=" & Me.w(i).ToString & vbLf)
            Next

        End Sub

        Public Overrides Sub PrintOutput(iteration%)

            If ShowThisIteration(iteration) Then

                If Not Me.vectorizedLearningMode Then
                    Dim nbTargets = Me.targetArray.GetLength(1)
                    TestAllSamples(Me.inputArray, nbOutputs:=nbTargets)
                End If
                Dim outputArrayDble#(,) = Me.output
                ComputeAverageError()
                Dim sMsg$ = vbLf & "Iteration n°" & iteration + 1 & "/" & nbIterations & vbLf &
                    "Output: " & Me.output.ToString() & vbLf &
                    "Average error: " & Me.averageError.ToString(format6Dec)
                'For i = 0 To Me.LayerCount - 1
                '    sMsg &= "Error(" & i & ")=" & Me.error_(i).ToString() & vbLf
                '    sMsg &= "A(" & i & ")=" & A(i).ToString() & vbLf
                'Next

                ShowMessage(sMsg)

            End If

        End Sub

    End Class

End Namespace