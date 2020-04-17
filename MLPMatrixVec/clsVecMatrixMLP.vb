
Option Infer On ' Lambda function

Imports System.Runtime.InteropServices ' OutAttribute <Out>

Namespace VectorizedMatrixMLP

    Public Class clsVectorizedMatrixMLP : Inherits clsMLPGeneric

        ' Note: Me.weightAdjustment: not used there

        Public vectorizedLearningMode As Boolean = True

        Public input As Matrix
        Public target As Matrix

        ''' <summary>
        ''' Output matrix (returned to compute average error, and discrete error)
        ''' </summary>
        Public outputMatrix As Matrix

        Public exampleCount%

        Private w As Matrix()

        Private error_ As Matrix()
        Private lastError As Matrix

        Private neuronCount%()

        Public Overrides Sub InitStruct(neuronCount%(), addBiasColumn As Boolean)

            Me.useBias = addBiasColumn
            Me.layerCount = neuronCount.Length
            ReDim Me.neuronCount(0 To Me.layerCount - 1)
            For i As Integer = 0 To Me.layerCount - 1
                Me.neuronCount(i) = neuronCount(i)
                If Me.useBias AndAlso
                    i > 0 AndAlso i < Me.layerCount - 1 Then Me.neuronCount(i) += 1 ' Bias
            Next
            Me.exampleCount = Me.target.x
            Me.w = New Matrix(Me.layerCount - 1 - 1) {}

        End Sub

        Public Overrides Sub Randomize(Optional minValue! = 0, Optional maxValue! = 1)

            'Me.rnd = New Random(Seed:=1)
            Me.rnd = New Random()

            For i = 0 To Me.w.Length - 1
                Dim iNbNeur% = Me.neuronCount(i)
                If Me.useBias Then iNbNeur += 1
                Dim iNbNeurCP1% = Me.neuronCount(i + 1)
                Me.w(i) = Matrix.Random(
                    iNbNeur, iNbNeurCP1, Me.rnd, minValue, maxValue) * 2 - 1
            Next

        End Sub

        Public Overrides Sub WeightInit(layer%, weights#(,))
            Me.w(layer) = weights
        End Sub

        Public Overrides Sub PrintWeights()

            Me.PrintParameters()

            For i As Integer = 0 To Me.layerCount - 1
                ShowMessage("Neuron count(" & i & ")=" & Me.neuronCount(i))
            Next

            ShowMessage("")

            For i = 0 To Me.w.Length - 1
                ShowMessage("W(" & i & ")=" & Me.w(i).ToString)
            Next

        End Sub

        Public Sub TrainVector()

            Me.vectorizedLearningMode = True
            For iteration = 0 To Me.nbIterations - 1

                OneIteration(Me.input, testOnly:=False, computeError:=True,
                    TargetValue:=Me.target)

                If Me.printOutput_ Then PrintOutput(iteration)

            Next

        End Sub

        Public Overrides Sub PrintOutput(iteration%)

            If ShowThisIteration(iteration) Then

                If Not Me.vectorizedLearningMode Then
                    Dim nbTargets% = Me.targetArray.GetLength(1)
                    TestAllSamples(Me.inputArray, nbOutputs:=nbTargets)
                    Me.outputMatrix = Me.outputArray
                End If
                ComputeAverageError()
                Dim sMsg$ = vbLf & "Iteration n°" & iteration + 1 & "/" & nbIterations & vbLf &
                    "Output: " & Me.outputMatrix.ToString() & vbLf &
                    "Average error: " & Me.averageError.ToString("0.000000")
                'For i = 0 To Me.LayerCount - 1
                '    sMsg &= "Error(" & i & ")=" & Me.error_(i).ToString() & vbLf
                '    sMsg &= "A(" & i & ")=" & A(i).ToString() & vbLf
                'Next

                ShowMessage(sMsg)

            End If

        End Sub

        Public Sub OneIteration(
            InputValue As Matrix, testOnly As Boolean,
            Optional computeError As Boolean = False,
            Optional TargetValue As Matrix = Nothing)

            Me.error_ = Nothing
            Me.outputMatrix = Nothing

            Dim Z As Matrix() = Nothing
            Dim A As Matrix() = Nothing
            ForwardPropagation(InputValue, Z, A)

            Dim maxLayer% = layerCount - 1
            Dim maxIndex% = A.Length - 1
            Dim Zlast As Matrix = Z(maxLayer)
            ' Cut first column for last layer
            Dim zx = Z(maxLayer).x
            Dim zy = Z(maxLayer).y
            If Me.useBias Then Zlast = Zlast.Slice(0, 1, zx, zy)

            Me.outputMatrix = A(maxIndex)
            ' Cut first column for last index of result matrix
            Dim ax = A(maxIndex).x
            Dim ay = A(maxIndex).y
            If Me.useBias Then Me.outputMatrix = Me.outputMatrix.Slice(0, 1, ax, ay)

            Me.error_ = Nothing
            If computeError Then
                Me.error_ = New Matrix(Me.layerCount - 1) {}
                Me.error_(Me.layerCount - 1) = Me.outputMatrix - TargetValue
            End If

            If testOnly Then Exit Sub

            Dim delta As Matrix() = Nothing
            BackPropagation(delta, Me.error_, Zlast, Z)

            GradientDescend(A, delta)

        End Sub

        Private Sub ForwardPropagation(InputValue As Matrix,
            <Out> ByRef Z As Matrix(), <Out> ByRef A As Matrix())

            Z = New Matrix(layerCount - 1) {}
            A = New Matrix(layerCount - 1) {}

            Z(0) = InputValue
            ' Column added with 1 for all examples
            If Me.useBias Then Z(0) = Z(0).AddColumn(Matrix.Ones(Me.exampleCount, 1))
            A(0) = Z(0)

            For i = 1 To layerCount - 1

                Dim AW = A(i - 1) * w(i - 1)

                Z(i) = AW
                ' Column added with 1 for all examples
                If Me.useBias Then Z(i) = Z(i).AddColumn(Matrix.Ones(Me.exampleCount, 1))

                A(i) = Matrix.Map(Z(i), Me.lambdaFnc)

            Next

            ' How use Relu
            ' Change all sigmoid function, for relu function
            ' Last A must have no Nonlinear function Matrix, Last A must be Equal To Last Z;
            '  because of that Last Delta has not derivated Matrix "Last Delta = Last error Error * 1";
            ' The learning rate must be smaller, like 0.001
            ' Optionaly you can use a Softmax layer to make a classifier
            ' Use if Relu OR iregularized Values
            If Me.activFnc.IsNonLinear Then A(A.Length - 1) = Z(Z.Length - 1)

        End Sub

        Private Sub BackPropagation(
            <Out> ByRef delta As Matrix(), error_ As Matrix(),
            Zlast As Matrix, Z As Matrix())

            delta = New Matrix(Me.layerCount - 1) {}
            delta(Me.layerCount - 1) = error_(Me.layerCount - 1) * Matrix.Map(Zlast, Me.lambdaFncD)

            For i = Me.layerCount - 2 To 0 Step -1

                Dim d = delta(i + 1)
                Dim t = Me.w(i).T
                Me.error_(i) = d * t

                delta(i) = Me.error_(i) * Matrix.Map(Z(i), Me.lambdaFncD)

                ' Cut first column
                If Me.useBias Then delta(i) = delta(i).Slice(0, 1, delta(i).x, delta(i).y)

            Next

        End Sub

        Private Sub GradientDescend(A As Matrix(), delta As Matrix())

            For i = 0 To w.Length - 1
                Me.w(i) -= A(i).T * delta(i + 1) * Me.learningRate
            Next

        End Sub

        Public Overrides Sub ComputeError()
            ' Calculate the error: ERROR = TARGETS - OUTPUTS
            Dim m As Matrix = Me.targetArray
            Me.lastError = m - Me.outputMatrix
        End Sub

        Public Sub SetLastError()
            Me.lastError = Me.error_(Me.layerCount - 1)
        End Sub

        Public Overrides Sub ComputeAverageErrorFromLastError()
            ' Compute first abs then average:
            Me.averageError = CSng(Me.lastError.abs.average * Me.exampleCount)
        End Sub

        Public Overrides Function ComputeAverageError!()

            Me.ComputeError()
            Me.ComputeAverageErrorFromLastError()
            Return Me.averageError

        End Function

        Public Sub ComputeErrorOneSample()
            ' Calculate the error: ERROR = TARGETS - OUTPUTS
            Dim m As Matrix = Me.targetArray
            Me.lastError = (m - Me.outputMatrix).GetRow(0)
        End Sub

        Public Sub ComputeAverageErrorFromLastErrorOneSample()
            ' Compute first abs then average:
            Me.averageError = CSng(Me.lastError.abs.average)
        End Sub

        Public Function ComputeAverageErrorOneSample!()

            Me.ComputeErrorOneSample()
            Me.ComputeAverageErrorFromLastErrorOneSample()
            Return Me.averageError

        End Function

        Public Overrides Sub TrainSystematic(inputs!(,), targets!(,),
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

            If learningMode = enumLearningMode.Vectoriel Then
                TrainVector()
                Exit Sub
            End If

            Me.vectorizedLearningMode = False
            Me.exampleCount = 1
            MyBase.TrainSystematic(inputs, targets, learningMode)

        End Sub

        Public Overrides Sub TrainOneSample(input!(), target!())

            Dim inputsDble#(0, input.Length - 1)
            inputsDble = clsMLPHelper.FillArray2(inputsDble, input, 0)
            Dim matrixInput As Matrix = inputsDble

            Dim targetsDble#(0, target.Length - 1)
            targetsDble = clsMLPHelper.FillArray2(targetsDble, target, 0)
            Dim TargetValue As Matrix = targetsDble

            Me.exampleCount = 1
            OneIteration(matrixInput, testOnly:=False, computeError:=True,
                TargetValue:=TargetValue)

        End Sub

        Public Overrides Sub TestOneSample(input!())

            Dim inputDble#(0, input.Length - 1)
            inputDble = clsMLPHelper.FillArray2(inputDble, input, 0)
            Dim matrixInput As Matrix = inputDble

            OneIteration(matrixInput, testOnly:=True, computeError:=False)

            ' Get first row
            Dim outputMatrix1D = Me.outputMatrix.GetRow(0)
            Me.lastOutputArray = outputMatrix1D.ToArray
            Me.lastOutputArraySingle = clsMLPHelper.ConvertDoubleToSingle(Me.lastOutputArray)

        End Sub

    End Class

End Namespace