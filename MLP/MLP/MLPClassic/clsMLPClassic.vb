﻿
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

Public Class clsMLPClassic : Inherits clsMLPGeneric

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

    Public biasType As TBias = TBias.WeightAdded

    Private Layers() As TLayer = Nothing

    Private minValue! = 0
    Private maxValue! = 1

#End Region

#Region "Initialization"

    Public Overrides Function GetMLPType$()
        Return System.Reflection.MethodBase.GetCurrentMethod().DeclaringType.Name
    End Function

    Public Overrides Function GetActivationFunctionType() As enumActivationFunctionType
        Return enumActivationFunctionType.Normal
    End Function

    Public Overrides Sub InitializeStruct(neuronCount%(), addBiasColumn As Boolean)

        MyBase.InitializeStruct(neuronCount, addBiasColumn)

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

        'Me.nbInputNeurons = Me.Layers(0).nbNeurons
        'Me.nbOutputNeurons = Me.Layers(Me.layerCount - 1).nbNeurons

    End Sub

    Public Overrides Sub Randomize(Optional minValue! = -0.5!, Optional maxValue! = 0.5!)

        For i = 1 To Me.layerCount - 1
            For j = 0 To Me.Layers(i).nbNeurons - 1
                For k = 0 To Me.Layers(i - 1).nbWeights - 1
                    With Me.Layers(i).Neurons(j)
                        Dim r# = rndShared.NextDouble(minValue, maxValue)
                        Dim rounded# = Math.Round(r, clsMLPGeneric.nbRoundingDigits)
                        .w(k) = CSng(rounded)
                        .dw(k) = 0
                        .wCopy(k) = 0
                    End With
                Next k
            Next j
        Next i

    End Sub

    Public Overrides Sub RoundWeights()

        ' Round the weights (to reproduce all tests exactly)

        For i As Integer = 1 To Me.layerCount - 1
            For j As Integer = 0 To Me.Layers(i).nbNeurons - 1
                For k As Integer = 0 To Me.Layers(i - 1).nbWeights - 1
                    With Me.Layers(i).Neurons(j)
                        Dim r! = .w(k)
                        Dim rounded# = Math.Round(r, clsMLPGeneric.nbRoundingDigits)
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

        Dim output1D!(Me.nbOutputNeurons - 1)
        TestOneSample(input, output1D)
        Me.lastOutputArray1DSingle = output1D

        Me.lastOutputArray1D = clsMLPHelper.Convert1DArrayOfSingleToDouble(
            Me.lastOutputArray1DSingle)
        Dim outputs2D#(0, Me.nbOutputNeurons - 1)
        clsMLPHelper.Fill2DArrayOfDouble(outputs2D, Me.lastOutputArray1D, 0)
        Me.output = outputs2D

    End Sub

    Public Overrides Sub TestOneSample(input!(), ByRef output1D!())

        SetInputSignal(input)
        ForwardPropogateSignal()
        GetOutputSignal(output1D)
        Me.lastOutputArray1DSingle = output1D

        Me.lastOutputArray1D = clsMLPHelper.Convert1DArrayOfSingleToDouble(
            Me.lastOutputArray1DSingle)
        Dim outputs2D#(0, Me.nbOutputNeurons - 1)
        clsMLPHelper.Fill2DArrayOfDouble(outputs2D, Me.lastOutputArray1D, 0)
        Me.output = outputs2D

    End Sub

    Public Overrides Sub TrainOneSample(input!(), target!())

        Dim output!(Me.nbOutputNeurons - 1)
        TestOneSample(input, output)
        Me.averageErrorOneSample = ComputeOutputError(target) ' 21/05/2021 Keep average error
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

        For i = Me.layerCount - 2 To 0 Step -1
            For j = 0 To Me.Layers(i).nbNeurons - 1

                Dim sumError! = 0
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

        Dim totalErr! = 0
        Dim totalAbsErr! = 0
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

            totalAbsErr += Math.Abs(delta)
            totalErr += delta

        Next i

        Me.averageErrorOneSample = totalAbsErr / Me.nbOutputNeurons
        Me.averageErrorOneSampleSigned = totalErr / Me.nbOutputNeurons

        Return totalAbsErr

    End Function

#End Region

#Region "Get/Set weights"

    Public Overrides Function GetWeight#(layer%, neuron%, weight%)
        Dim ws! = Me.GetWeightSingle(layer, neuron, weight)
        Dim wd# = ws
        Return wd
    End Function

    Public Overrides Function GetWeightSingle!(layer%, neuron%, weight%)
        Return Me.Layers(layer).Neurons(neuron).w(weight)
    End Function

    Public Overrides Sub SetWeight(layer%, neuron%, weight%, weightWalue#)
        Dim ws! = CSng(weightWalue)
        SetWeightSingle(layer, neuron, weight, ws)
    End Sub

    Public Overrides Sub SetWeightSingle(layer%, neuron%, weight%, weightWalue!)
        Me.Layers(layer).Neurons(neuron).w(weight) = weightWalue
    End Sub

#End Region

End Class
