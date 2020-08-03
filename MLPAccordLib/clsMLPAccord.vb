
' From http://accord-framework.net/docs/html/T_Accord_Neuro_Learning_BackPropagationLearning.htm : C# -> VB .NET conversion

' See also :
' https://github.com/accord-net/framework
' https://github.com/accord-net/framework/releases/download/v3.8.0/Accord.NET-3.8.0-archive.rar
' https://www.nuget.org/packages/Accord : 3.8.0
' http://accord-framework.net

Imports Accord.Neuro
Imports Accord.Neuro.Learning

Imports System.IO
Imports Perceptron.Utility ' Matrix
Imports System.Text

Public Class clsMLPAccord : Inherits clsVectorizedMLPGeneric

    Private network As ActivationNetwork

    Public PRBPLAlgo As Boolean = False
    Private teacherBPL As BackPropagationLearning ' Reliable
    Private teacherPRBPL As ParallelResilientBackpropagationLearning ' Less reliable?

    Private neuronCountWithoutInputLayer%()

    Public inputJaggedDblArray#()()
    Public targetJaggedDblArray#()()

    Public Overrides Sub InitializeStruct(neuronCount%(), addBiasColumn As Boolean)

        Dim inputNodes = neuronCount(0)
        Dim hiddenNodes = neuronCount(1)
        Me.layerCount = neuronCount.Length
        Me.useBias = addBiasColumn
        Me.neuronCount = neuronCount

        If Not Me.useBias Then
            Throw New NotImplementedException(
                "useBias=False is not implemented for clsAccordMLP!")
        End If

        Dim inputArrayDbl = clsMLPHelper.ConvertSingleToDouble(Me.inputArray)
        Me.inputJaggedDblArray = clsMLPHelper.TransformDoubleArrayToJaggedArray(inputArrayDbl)
        Dim targetArrayDbl = clsMLPHelper.ConvertSingleToDouble(Me.targetArray)
        Me.targetJaggedDblArray = clsMLPHelper.TransformDoubleArrayToJaggedArray(targetArrayDbl)

        Dim sigmoidAlphaValue! = Me.m_gain

        Me.layerCount = Me.neuronCount.Length
        Me.nbInputNeurons = Me.neuronCount(0)
        Me.nbHiddenNeurons = Me.neuronCount(1)
        Me.nbOutputNeurons = Me.neuronCount(Me.layerCount - 1)

        ' 2, 2, 1 :
        ' 2 : two inputs  in the network
        ' 2 : two neurons in the first layer
        ' 1 : one neuron  in the second layer
        'Dim neuronCountWithoutInputLayer%()
        ReDim Me.neuronCountWithoutInputLayer(0 To Me.layerCount - 2)
        For i = 1 To Me.layerCount - 1
            Me.neuronCountWithoutInputLayer(i - 1) = neuronCount(i)
        Next

    End Sub

    Public Overrides Sub SetActivationFunction(
        actFnc As enumActivationFunction, gain!, center!)

        Select Case actFnc
            Case enumActivationFunction.Sigmoid _
              : SetActivationFunctionForMatrix(
                    enumActivationFunctionForMatrix.Sigmoid, gain, center)
            Case enumActivationFunction.HyperbolicTangent _
              : SetActivationFunctionForMatrix(
                    enumActivationFunctionForMatrix.HyperbolicTangent, gain, center)
            Case Else
                Throw New NotImplementedException(
                    "This activation function is not available!")
        End Select

    End Sub

    Public Overrides Sub SetActivationFunctionForMatrix(
        fctAct As enumActivationFunctionForMatrix, gain!, center!)

        MyBase.SetActivationFunctionForMatrix(fctAct, gain, center)

        Dim sigmoidAlphaValue! = Me.m_gain
        Select Case fctAct
            Case enumActivationFunctionForMatrix.Sigmoid
                Me.m_actFunc = enumActivationFunction.Sigmoid
                Me.network = New ActivationNetwork(
                    New SigmoidFunction(sigmoidAlphaValue),
                        Me.nbInputNeurons, Me.neuronCountWithoutInputLayer)
            Case enumActivationFunctionForMatrix.HyperbolicTangent
                Me.m_actFunc = enumActivationFunction.HyperbolicTangent
                Me.network = New ActivationNetwork(
                    New BipolarSigmoidFunction(sigmoidAlphaValue),
                        Me.nbInputNeurons, Me.neuronCountWithoutInputLayer)
            Case enumActivationFunctionForMatrix.ELU
                Me.m_actFunc = Nothing
                Throw New NotImplementedException(
                    "ELU activation function is not available!")
            Case Else
                Me.activFnc = Nothing
        End Select

        If PRBPLAlgo Then
            Me.teacherPRBPL = New ParallelResilientBackpropagationLearning(Me.network)
            'Me.teacherPRBPL.Reset(Me.learningRate)
            'Me.teacherPRBPL.DecreaseFactor = 0.5 ' eta minus
            'Me.teacherPRBPL.IncreaseFactor = 1.2 ' eta plus
            'Me.teacherPRBPL.UpdateLowerBound = 0.000001 ' delta max
            'Me.teacherPRBPL.UpdateUpperBound = 50 ' delta min
        Else
            Me.teacherBPL = New BackPropagationLearning(Me.network)
            Me.teacherBPL.LearningRate = Me.learningRate
            Me.teacherBPL.Momentum = Me.weightAdjustment
        End If

    End Sub

    Public Overrides Sub InitializeWeights(numLayer%, weights#(,))

        Dim i = numLayer - 1
        Dim layer = Me.network.Layers(i)
        Dim nbNeurons = layer.Neurons.Count
        For j = 0 To nbNeurons - 1
            ' Neuron: no Threshold property!
            'Dim neuron = layer.Neurons(j)
            ' ActivationNeuron: Threshold property
            Dim neuron = CType(layer.Neurons(j), ActivationNeuron)
            Dim nbWeights = neuron.Weights.Count
            For k = 0 To nbWeights - 1
                neuron.Weights(k) = weights(j, k)
            Next k
            If Me.useBias Then
                neuron.Threshold = weights(j, nbWeights)
            Else
                neuron.Threshold = 0
            End If
        Next j

    End Sub

    Public Overrides Sub Randomize(Optional minValue! = 0, Optional maxValue! = 1)

        If PRBPLAlgo Then Me.teacherPRBPL.Reset(Me.learningRate)

        ' Randomly initialize the network
        ' Random between -0.5 and +0.5 with statistical normalization 
        '  according to the network structure
        Dim x As New NguyenWidrow(Me.network)
        x.Randomize()

        ' Alternative: initialize the network with Gaussian weights
        ' Dim x As New GaussianWeights(Me.network, 0.1)
        ' x.Randomize()

        ' Round the weights (to reproduce all tests exactly)
        For i = 0 To Me.layerCount - 2
            Dim layer = Me.network.Layers(i)
            Dim nbNeurons = layer.Neurons.Count
            For j = 0 To nbNeurons - 1
                Dim neuron = CType(layer.Neurons(j), ActivationNeuron)
                Dim nbWeights = neuron.Weights.Count
                For k = 0 To nbWeights - 1
                    Dim r# = neuron.Weights(k)
                    Dim rounded# = Math.Round(r, clsMLPGeneric.roundWeights)
                    neuron.Weights(k) = rounded
                Next k
                If Me.useBias Then
                    Dim r# = neuron.Threshold
                    Dim rounded# = Math.Round(r, clsMLPGeneric.roundWeights)
                    neuron.Threshold = rounded
                Else
                    neuron.Threshold = 0
                End If
            Next j
        Next i

    End Sub

    Public Overrides Sub ComputeError()
        ' Calculate the error: ERROR = TARGETS - OUTPUTS
        Dim m As Matrix = Me.targetArray
        Me.lastError = m - Me.output
    End Sub

    Public Overrides Sub ComputeAverageErrorFromLastError()
        ' Compute first abs then average:
        Me.averageError = CSng(Me.lastError.Abs.Average)
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

        Dim avgError#
        If PRBPLAlgo Then
            avgError = Me.teacherPRBPL.RunEpoch(Me.inputJaggedDblArray, Me.targetJaggedDblArray)
        Else
            avgError = Me.teacherBPL.RunEpoch(Me.inputJaggedDblArray, Me.targetJaggedDblArray)
        End If
        Me.averageError = CSng(avgError)

    End Sub

    Public Sub SetOuput1D()

        Dim nbInputs = Me.inputArray.GetLength(0)
        Dim nbTargets = Me.targetArray.GetLength(0)
        Dim lengthTot = nbTargets * Me.nbOutputNeurons
        Dim outputs1D#(lengthTot - 1)
        Dim outputs2D#(nbTargets - 1, Me.nbOutputNeurons - 1)
        Dim k = 0
        For i = 0 To nbInputs - 1
            Dim inputs#(1)
            inputs = Me.inputJaggedDblArray(i)
            Dim outputs#() = Me.network.Compute(inputs)
            For j = 0 To Me.nbOutputNeurons - 1
                outputs2D(i, j) = outputs(j)
                outputs1D(k) = outputs(j)
                k += 1
            Next
        Next

        Me.lastOutputArray1DSingle = clsMLPHelper.ConvertDoubleToSingle(outputs1D)
        Me.output = outputs2D

    End Sub

    Public Overrides Sub TrainOneSample(input!(), target!())

        Dim inputArrayDbl = clsMLPHelper.ConvertSingleToDouble1D(input)
        Dim targetArrayDbl = clsMLPHelper.ConvertSingleToDouble1D(target)

        Dim avgError#
        If PRBPLAlgo Then
            avgError = Me.teacherPRBPL.Run(inputArrayDbl, targetArrayDbl)
        Else
            avgError = Me.teacherBPL.Run(inputArrayDbl, targetArrayDbl)
        End If
        'Me.averageError = CSng(avgError)

        Dim outputs#() = Me.network.Compute(inputArrayDbl)
        Me.lastOutputArray1DSingle = clsMLPHelper.ConvertDoubleToSingle(outputs)

        Dim sum# = 0
        For i = 0 To Me.nbOutputNeurons - 1
            Dim ouput = outputs(i)
            Dim target0 = target(i)
            sum += Math.Abs(ouput - target0)
        Next
        Me.averageError = CSng(sum / Me.nbOutputNeurons)

    End Sub

    Public Overrides Sub TestOneSample(input!())

        Dim inputsDbl#() = clsMLPHelper.ConvertSingleToDouble1D(input)
        Dim outputs#() = Me.network.Compute(inputsDbl)
        Dim outputSng!() = clsMLPHelper.ConvertDoubleToSingle(outputs)
        Me.lastOutputArray1DSingle = outputSng

    End Sub

    Public Overrides Sub TestOneSample(input!(), ByRef ouput!())

        Dim inputsDbl#() = clsMLPHelper.ConvertSingleToDouble1D(input)
        Dim outputs#() = Me.network.Compute(inputsDbl)
        Dim outputSng!() = clsMLPHelper.ConvertDoubleToSingle(outputs)
        Me.lastOutputArray1DSingle = ouput

    End Sub

    Public Overrides Sub PrintWeights()

        Me.PrintParameters()

        For i = 0 To Me.network.Layers.Count - 1
            ShowMessage("Neuron count(" & i & ")=" & Me.network.Layers(i).Neurons.Count)
        Next

        ShowMessage("")

        Dim sb As New StringBuilder

        For i = 0 To Me.network.Layers.Count - 1

            sb.AppendLine("W(" & i + 1 & ")={")

            Dim nbNeurons = Me.network.Layers(i).Neurons.Count
            For j = 0 To nbNeurons - 1
                sb.Append(" {")
                Dim neuron = CType(Me.network.Layers(i).Neurons(j), ActivationNeuron)
                Dim nbWeights = neuron.Weights.Count
                For k = 0 To nbWeights - 1
                    Dim weight = neuron.Weights(k)
                    Dim sVal$ = weight.ToString(format2Dec).ReplaceCommaByDot()
                    sb.Append(sVal)
                    If Me.useBias OrElse k < nbWeights - 1 Then sb.Append(", ")
                Next k

                If Me.useBias Then
                    Dim weightT = neuron.Threshold
                    Dim sValT$ = weightT.ToString(format2Dec).ReplaceCommaByDot()
                    sb.Append(sValT)
                End If

                sb.Append("}")
                If j < nbNeurons - 1 Then sb.Append("," & vbLf)
            Next j
            sb.Append("}" & vbLf)

            If i < Me.network.Layers.Count - 1 Then sb.AppendLine()

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
