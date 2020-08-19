
' From  : https://github.com/encog/encog-dotnet-core : C# -> VB .NET conversion
' https://www.nuget.org/packages/encog-dotnet-core
' <package id="encog-dotnet-core" version="3.4.0" targetFramework="net472" />
' https://www.heatonresearch.com/encog

Imports Encog.Engine.Network.Activation
Imports Encog.ML.Data
Imports Encog.ML.Data.Basic
Imports Encog.ML.Train
Imports Encog.Neural.Networks
Imports Encog.Neural.Networks.Layers
Imports Encog.Neural.Networks.Training.Propagation.Resilient

Imports System.Text

Public Class clsMLPEncog : Inherits clsVectorizedMLPGeneric

    Private network As BasicNetwork
    Private trainingSet As IMLDataSet
    Private imlTrain As IMLTrain

    Public inputJaggedDblArray#()()
    Public targetJaggedDblArray#()()

    Public Overrides Sub InitializeStruct(neuronCount%(), addBiasColumn As Boolean)

        Dim inputArrayDbl = clsMLPHelper.Convert2DArrayOfSingleToDouble(Me.inputArray)
        Me.inputJaggedDblArray = clsMLPHelper.Transform2DArrayToJaggedArray(inputArrayDbl)
        Dim targetArrayDbl = clsMLPHelper.Convert2DArrayOfSingleToDouble(Me.targetArray)
        Me.targetJaggedDblArray = clsMLPHelper.Transform2DArrayToJaggedArray(targetArrayDbl)

        Me.layerCount = neuronCount.Length
        Me.useBias = addBiasColumn
        Me.neuronCount = neuronCount
        Me.nbInputNeurons = Me.neuronCount(0)
        Me.nbHiddenNeurons = Me.neuronCount(1)
        Me.nbOutputNeurons = Me.neuronCount(Me.layerCount - 1)

    End Sub

    Public Overrides Sub SetActivationFunction(
        actFnc As enumActivationFunction, gain!, center!)

        ' gain can only be 1 for Encog MLP
        gain = 1
        center = 0
        Me.weightAdjustment = 0 ' Not used

        MyBase.SetActivationFunction(actFnc, gain, center)

        Me.network = New BasicNetwork()

        ' Input layer
        Me.network.AddLayer(New BasicLayer(Nothing, Me.useBias,
            neuronCount:=Me.neuronCount(0)))

        Dim act As Encog.Engine.Network.Activation.IActivationFunction
        Select Case actFnc
            Case enumActivationFunction.Sigmoid : act = New ActivationSigmoid()
            Case enumActivationFunction.HyperbolicTangent : act = New ActivationTANH()
            Case enumActivationFunction.Gaussian : act = New ActivationGaussian() ' Does not work?
            Case enumActivationFunction.Sinus : act = New ActivationSIN()
            Case Else
                Throw New NotImplementedException(
                    "This activation function is not available!")
        End Select

        For i = 0 To Me.layerCount - 3 ' Hidden layers
            Me.network.AddLayer(New BasicLayer(act, Me.useBias, Me.neuronCount(i + 1)))
        Next
        ' Output layer : no bias
        Me.network.AddLayer(New BasicLayer(act, hasBias:=False,
            neuronCount:=Me.neuronCount(Me.layerCount - 1)))

        Me.network.Structure.FinalizeStructure()

        Me.trainingSet = New BasicMLDataSet(
            Me.inputJaggedDblArray, Me.targetJaggedDblArray)
        Me.imlTrain = New ResilientPropagation(Me.network, Me.trainingSet)

    End Sub

    Public Overrides Sub InitializeWeights(numLayer%, weights#(,))

        Dim i = numLayer - 1
        Dim nbNeuronsLayer = Me.network.GetLayerNeuronCount(i + 1)
        Dim nbNeuronsPreviousLayer = Me.network.GetLayerNeuronCount(i)
        For j = 0 To nbNeuronsLayer - 1
            Dim nbWeights = nbNeuronsPreviousLayer
            For k = 0 To nbWeights - 1
                Dim r = weights(j, k)
                Me.network.SetWeight(i, k, j, r)
            Next k
            If Me.useBias Then
                Dim r = weights(j, nbWeights)
                Me.network.SetWeight(i, nbWeights, j, r)
            End If
        Next j

    End Sub

    Public Overrides Sub Randomize(Optional minValue! = 0, Optional maxValue! = 1)

        Me.network.Reset()

        ' Round the weights (to reproduce all tests exactly)
        For i = 1 To Me.layerCount - 1
            Dim nbNeuronsLayer = Me.network.GetLayerNeuronCount(i)
            Dim nbNeuronsPreviousLayer = Me.network.GetLayerNeuronCount(i - 1)
            For j = 0 To nbNeuronsLayer - 1
                Dim nbWeights = nbNeuronsPreviousLayer
                For k = 0 To nbWeights - 1
                    Dim weight = Me.network.GetWeight(i - 1, k, j)
                    Dim rounded = Math.Round(weight, clsMLPGeneric.roundWeights)
                    Me.network.SetWeight(i - 1, k, j, rounded)
                Next k
                If Me.useBias Then
                    Dim weightT = Me.network.GetWeight(i - 1, nbWeights, j)
                    Dim rounded = Math.Round(weightT, clsMLPGeneric.roundWeights)
                    Me.network.SetWeight(i - 1, nbWeights, j, rounded)
                End If
            Next j
        Next i

    End Sub

    Public Overrides Sub TrainVector()

        Me.vectorizedLearningMode = True
        For iteration = 0 To Me.nbIterations - 1
            TrainVectorOneIteration()
            If Me.printOutput_ Then PrintOutput(iteration)
        Next
        SetOuput1D()

    End Sub

    Public Sub TrainVectorOneIteration()

        Me.imlTrain.Iteration()
        Me.averageError = CSng(Me.imlTrain.Error)

    End Sub

    Public Sub SetOuput1D()

        Dim nbInputs = Me.inputArray.GetLength(0)
        Dim nbTargets = Me.targetArray.GetLength(0)
        Dim lengthTot = nbTargets * Me.nbOutputNeurons
        Dim outputs1D#(lengthTot - 1)
        Dim outputs2D#(nbTargets - 1, Me.nbOutputNeurons - 1)
        Dim k = 0
        For i = 0 To nbInputs - 1
            Dim outputs#(Me.nbOutputNeurons - 1)
            Me.network.Compute(Me.inputJaggedDblArray(i), outputs)
            For j = 0 To Me.nbOutputNeurons - 1
                outputs2D(i, j) = outputs(j)
                outputs1D(k) = outputs(j)
                k += 1
            Next
        Next
        Me.lastOutputArray1DSingle = clsMLPHelper.Convert1DArrayOfDoubleToSingle(outputs1D)
        Me.output = outputs2D

    End Sub

    Public Overrides Sub TrainSystematic(inputs!(,), targets!(,),
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        'If learningMode = enumLearningMode.Vectorial Then
             ' This is the unique learning mode for this MLP
             TrainVector()
        '    Exit Sub
        'End If

        'Me.vectorizedLearningMode = False
        'TrainAllSamples(inputs, targets)

    End Sub

    Public Overrides Sub TrainStochastic(inputs!(,), targets!(,))
        Throw New NotImplementedException("There is no TrainOneSample() function!")
    End Sub

    Public Overrides Sub TrainSemiStochastic(inputs!(,), targets!(,))
        Throw New NotImplementedException("There is no TrainOneSample() function!")
    End Sub

    Public Overrides Sub TrainOneSample(input!(), target!())
        Throw New NotImplementedException("There is no TrainOneSample() function!")
    End Sub

    Public Overrides Sub TestOneSample(input!())

        Dim inputArrayDbl = clsMLPHelper.Convert1DArrayOfSingleToDouble(input)
        Dim outputs#(Me.nbOutputNeurons - 1)
        Me.network.Compute(inputArrayDbl, outputs)
        Me.lastOutputArray1DSingle = clsMLPHelper.Convert1DArrayOfDoubleToSingle(outputs)

        Dim outputs2D#(0, Me.nbOutputNeurons - 1)
        clsMLPHelper.Fill2DArrayOfDouble(outputs2D, outputs, 0)
        Me.output = outputs2D

    End Sub

    Public Overrides Sub PrintWeights()

        Me.PrintParameters()

        For i = 0 To Me.layerCount - 1
            ShowMessage("Neuron count(" & i & ")=" & Me.neuronCount(i))
        Next

        ShowMessage("")

        Dim sb As New StringBuilder

        For i = 1 To Me.layerCount - 1

            sb.AppendLine("W(" & i & ")={")

            Dim nbNeuronsLayer = Me.network.GetLayerNeuronCount(i)
            Dim nbNeuronsPreviousLayer = Me.network.GetLayerNeuronCount(i - 1)

            For j = 0 To nbNeuronsLayer - 1
                sb.Append(" {")

                Dim nbWeights = nbNeuronsPreviousLayer
                For k = 0 To nbWeights - 1
                    Dim weight = Me.network.GetWeight(i - 1, k, j)
                    Dim sVal$ = weight.ToString(format2Dec).ReplaceCommaByDot()
                    sb.Append(sVal)
                    If Me.useBias OrElse k < nbWeights - 1 Then sb.Append(", ")
                Next k

                If Me.useBias Then
                    Dim weightT = Me.network.GetWeight(i - 1, nbWeights, j)
                    Dim sValT$ = weightT.ToString(format2Dec).ReplaceCommaByDot()
                    sb.Append(sValT)
                End If

                sb.Append("}")
                If j < nbNeuronsLayer - 1 Then sb.Append("," & vbLf)
            Next j
            sb.Append("}" & vbLf)

            If i < Me.layerCount - 1 Then sb.AppendLine()

        Next i

        ShowMessage(sb.ToString())

    End Sub

    Public Overrides Sub PrintOutput(iteration%)

        If ShowThisIteration(iteration) Then
            If Not Me.vectorizedLearningMode Then
                Dim nbTargets = Me.targetArray.GetLength(1)
                TestAllSamples(Me.inputArray, nbOutputs:=nbTargets)
            End If
            SetOuput1D()
            ComputeAverageError()
            Dim sMsg$ = vbLf & "Iteration n°" & iteration + 1 & "/" & nbIterations & vbLf &
                "Output: " & Me.output.ToString() & vbLf &
                "Average error: " & Me.averageError.ToString(format6Dec)
            ShowMessage(sMsg)
        End If

    End Sub

End Class