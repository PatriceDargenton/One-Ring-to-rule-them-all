
' From:

' *********************************************************************
' * File  : mlp.h (+ mlp.cpp)
' * Author: Sylvain BARTHELEMY
' *         https://github.com/sylbarth/mlp
' * Date  : 2000-08
' *********************************************************************

' http://patrice.dargenton.free.fr/ia/ialab/perceptron.html (french)

Imports System.Text ' StringBuilder
Imports Perceptron.Utility ' Matrix

Friend Class clsMLPClassic : Inherits clsMLPGeneric

#Region "Declarations"

    Public Const floatMax! = 99999
    Public Const floatMin! = -99999

    Structure TNeuron
        Dim signal!
        Dim err!
        Dim w!()
        Dim dw!()  ' Weight adjustment
        Dim wCopy!() ' Weight copy, if the average error decreases
        Dim signalCopy! ' Signal copy used for some derivate
    End Structure

    Structure TLayer
        Dim nbNeurons%
        Dim nbWeights%
        Dim Neurons() As TNeuron
    End Structure

    Public Enum TBias ' Bias Type
        Disabled = 0
        WeightAdded = 1
        NeuronAdded = 2
        NeuronAddedSpecial = 3
    End Enum

    Public biasType As TBias = TBias.WeightAdded

    Private Layers() As TLayer = Nothing

    Private minValue! = 0
    Private maxValue! = 1

#End Region

#Region "Init"

    Public Overrides Sub InitializeStruct(neuronCount%(), addBiasColumn As Boolean)

        Me.layerCount = neuronCount.GetLength(0)
        Me.useBias = addBiasColumn

        ReDim Me.Layers(Me.layerCount - 1)

        Dim nbWeights%

        For i = 0 To Me.layerCount - 1
            With Me.Layers(i)
                .nbNeurons = neuronCount(i)

                If Me.biasType >= TBias.NeuronAdded AndAlso
                    i > 0 AndAlso i < Me.layerCount - 1 Then .nbNeurons -= 1

                .nbWeights = .nbNeurons
                If addBiasColumn AndAlso i < Me.layerCount - 1 Then .nbWeights += 1

                ReDim .Neurons(.nbNeurons - 1)
                For j = 0 To .nbNeurons - 1
                    .Neurons(j).signal = Me.minValue
                    .Neurons(j).err = 0.0!
                    If Me.biasType >= TBias.NeuronAdded AndAlso
                       j = .nbNeurons - 1 Then .Neurons(j).signal = Me.maxValue
                    If i > 0 Then
                        nbWeights = Me.Layers(i - 1).nbWeights
                        ReDim .Neurons(j).w(nbWeights - 1)
                        ReDim .Neurons(j).dw(nbWeights - 1)
                        ReDim .Neurons(j).wCopy(nbWeights - 1)
                    End If
                Next j
            End With
        Next i

        Me.nbInputNeurons = Me.Layers(0).nbNeurons
        Me.nbOutputNeurons = Me.Layers(Me.layerCount - 1).nbNeurons

    End Sub

    Public Overrides Sub Randomize(Optional minValue! = 0, Optional maxValue! = 1)

        For i = 1 To Me.layerCount - 1
            For j = 0 To Me.Layers(i).nbNeurons - 1
                For k = 0 To Me.Layers(i - 1).nbWeights - 1
                    With Me.Layers(i).Neurons(j)
                        Dim r# = rndShared.NextDouble(minValue, maxValue)
                        Dim rounded# = Math.Round(r, clsMLPGeneric.roundWeights)
                        .w(k) = CSng(rounded)
                        .dw(k) = 0
                        .wCopy(k) = 0
                    End With
                Next k
            Next j
        Next i

    End Sub

    Public Overrides Sub InitializeWeights(layer%, weights#(,))

        For j = 0 To Me.Layers(layer).nbNeurons - 1
            For k = 0 To Me.Layers(layer - 1).nbWeights - 1
                Me.Layers(layer).Neurons(j).w(k) = CSng(weights(j, k))
            Next k
        Next j

    End Sub

#End Region

#Region "Compute"

    Public Overrides Sub TestOneSample(input!())

        Dim ouput!(Me.nbOutputNeurons - 1)
        TestOneSample(input, ouput)
        Me.lastOutputArray1DSingle = ouput

    End Sub

    Public Overrides Sub TestOneSample(input!(), ByRef ouput!())

        SetInputSignal(input)
        ForwardPropogateSignal()
        GetOutputSignal(ouput)

    End Sub

    Public Overrides Sub TrainOneSample(input!(), target!())

        Dim output!(Me.nbOutputNeurons - 1)
        TestOneSample(input, output)
        ComputeOutputError(target)
        BackwardPropagateError()
        ComputeGradientAndAdjustWeights()

    End Sub

    Private Sub SetInputSignal(input!())

        For i = 0 To Me.nbInputNeurons - 1
            Me.Layers(0).Neurons(i).signal = input(i)
        Next i

    End Sub

    Private Sub ForwardPropogateSignal()

        ' Calculate and feedforward outputs from the first layer to the last

        For i = 1 To Me.layerCount - 1
            Dim nbNeurons = Me.Layers(i).nbNeurons
            For j = 0 To nbNeurons - 1

                Dim rSum! = 0
                For k = 0 To Me.Layers(i - 1).nbWeights - 1
                    Dim signal! = Me.maxValue
                    If k < Me.Layers(i - 1).nbNeurons Then _
                        signal = Me.Layers(i - 1).Neurons(k).signal
                    rSum += Me.Layers(i).Neurons(j).w(k) * signal
                Next k

                Me.Layers(i).Neurons(j).signalCopy = rSum

                If Not (Me.biasType = TBias.NeuronAdded AndAlso
                    i < Me.layerCount - 1 AndAlso j = nbNeurons - 1) Then

                    Dim r# = Me.lambdaFnc.Invoke(rSum)
                    Me.Layers(i).Neurons(j).signal = CSng(r)

                End If

            Next j
        Next i

    End Sub

    Private Sub GetOutputSignal(ByRef ouput!())

        For i = 0 To Me.nbOutputNeurons - 1
            ouput(i) = Me.Layers(Me.layerCount - 1).Neurons(i).signal
            If ouput(i) > floatMax Then ouput(i) = Me.maxValue
            If ouput(i) < floatMin Then ouput(i) = Me.minValue

            If Single.IsNaN(ouput(i)) Then ' -1.#IND
                ouput(i) = Me.minValue
            End If
            If Single.IsPositiveInfinity(ouput(i)) Then
                ouput(i) = Me.maxValue
            End If
            If Single.IsNegativeInfinity(ouput(i)) Then
                ouput(i) = Me.minValue
            End If

        Next i

    End Sub

    Public Sub BackwardPropagateError()

        ' Backward propagate error from the output layer through to the first layer

        Dim sumError!
        For i = Me.layerCount - 2 To 0 Step -1
            For j = 0 To Me.Layers(i).nbNeurons - 1

                sumError = 0
                For k = 0 To Me.Layers(i + 1).nbNeurons - 1
                    sumError += Me.Layers(i + 1).Neurons(k).w(j) *
                        Me.Layers(i + 1).Neurons(k).err
                Next k

                Dim signalCopy! = Layers(i).Neurons(j).signalCopy
                Dim deriv# = Me.lambdaFncD.Invoke(signalCopy)
                Me.Layers(i).Neurons(j).err = sumError * CSng(deriv)

            Next j
        Next i

    End Sub

    Public Sub ComputeGradientAndAdjustWeights()

        ' Gradient descend: Compute gradient and adjust weights
        ' Update weights for all of the neurons from the first to the last layer

        For i = 1 To Me.layerCount - 1
            For j = 0 To Me.Layers(i).nbNeurons - 1
                For k = 0 To Me.Layers(i - 1).nbWeights - 1
                    With Me.Layers(i).Neurons(j)

                        Dim signal! = Me.maxValue
                        If k < Me.Layers(i - 1).nbNeurons Then _
                            signal = Me.Layers(i - 1).Neurons(k).signal

                        If .err > floatMax OrElse .err < floatMin Then
                            .w(k) = 0
                            .dw(k) = 0
                        Else
                            If signal = 0 AndAlso .dw(k) = 0 Then
                                .dw(k) = 0
                            Else
                                Dim adjust! = Me.learningRate * signal * .err
                                .w(k) += adjust '+ Me.weightAdjustment * .dw(k)
                                If Me.weightAdjustment <> 0 Then .w(k) += Me.weightAdjustment * .dw(k)
                                .dw(k) = adjust
                            End If
                        End If
                    End With
                Next k
            Next j
        Next i

    End Sub

#End Region

#Region "Error"

    Public Function ComputeOutputError!(target!())

        Dim averageAbsErr! = 0
        Dim outputLayerIndex = Me.layerCount - 1

        For i = 0 To Me.nbOutputNeurons - 1

            Dim signal = Me.Layers(outputLayerIndex).Neurons(i).signal
            Dim delta = target(i) - signal

            Dim deriv#
            If Me.activFnc.DoesDerivativeDependOnOriginalFunction Then
                ' Optimization is possible in this case
                deriv = Me.lambdaFncDFOF.Invoke(signal)
            Else
                Dim signalCopy = Me.Layers(outputLayerIndex).Neurons(i).signalCopy
                deriv = Me.lambdaFncD.Invoke(signalCopy)
            End If

            Me.Layers(outputLayerIndex).Neurons(i).err = delta * CSng(deriv)

            averageAbsErr += Math.Abs(delta)

        Next i

        If Me.nbOutputNeurons <> 1 Then averageAbsErr /= Me.nbOutputNeurons

        Return averageAbsErr

    End Function

    Public Overrides Sub ComputeError()
        ' Calculate the error: ERROR = TARGETS - OUTPUTS
        Dim m As Matrix = Me.targetArray
        Me.lastError = m - Me.output
    End Sub

    Public Overrides Sub ComputeAverageErrorFromLastError()
        ' Compute first abs then average:
        Me.averageError = CSng(Me.lastError.Abs.Average)
    End Sub

#End Region

#Region "Print"

    Public Overrides Sub PrintWeights()

        Me.PrintParameters()

        Dim sb As New StringBuilder

        For i = 0 To Me.layerCount - 1
            sb.AppendLine("Neuron count(" & i & ")=" & Me.Layers(i).nbNeurons)
        Next

        sb.AppendLine()

        For i = 1 To Me.layerCount - 1

            sb.AppendLine("W(" & i & ")={")

            Dim nbNeurons = Me.Layers(i).nbNeurons
            For j = 0 To nbNeurons - 1
                sb.Append(" {")
                Dim nbWeights = Me.Layers(i - 1).nbWeights
                For k = 0 To nbWeights - 1
                    Dim weight = Me.Layers(i).Neurons(j).w(k)
                    Dim sVal$ = weight.ToString(format2Dec).ReplaceCommaByDot()
                    sb.Append(sVal)
                    If k < nbWeights - 1 Then sb.Append(", ")
                Next k
                sb.Append("}")
                If j < nbNeurons - 1 Then sb.Append("," & vbLf)
            Next j
            sb.Append("}" & vbLf)

            If i < Me.layerCount - 1 Then sb.AppendLine()

        Next i

        ShowMessage(sb.ToString())

    End Sub

    Public Overrides Sub PrintOutput(iteration%)

        If ShowThisIteration(iteration) Then

            Dim nbTargets = Me.targetArray.GetLength(1)
            TestAllSamples(Me.inputArray, nbTargets)
            Dim avErr = ComputeAverageError()
            Dim msg$ = vbLf & "Iteration n°" & iteration + 1 & "/" & nbIterations & vbLf &
                "Output: " & Me.output.ToString() & vbLf &
                "Average error: " & avErr.ToString(format6Dec)
            ShowMessage(msg)

        End If

    End Sub

#End Region

End Class
