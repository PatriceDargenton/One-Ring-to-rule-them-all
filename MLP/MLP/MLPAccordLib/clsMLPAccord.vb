
' From http://accord-framework.net/docs/html/T_Accord_Neuro_Learning_BackPropagationLearning.htm : C# -> VB .NET conversion

' See also :
' https://github.com/accord-net/framework
' https://github.com/accord-net/framework/releases/download/v3.8.0/Accord.NET-3.8.0-archive.rar
' https://www.nuget.org/packages/Accord : 3.8.0
' http://accord-framework.net

Imports Accord.Neuro
Imports Accord.Neuro.Learning

Imports Perceptron.Utility ' Matrix
Imports System.Text

Public Class clsMLPAccord : Inherits clsVectorizedMLPGeneric

    Private network As ActivationNetwork

    Private teacherBPL As BackPropagationLearning ' Reliable

    ''' <summary>
    ''' Resilient Backpropagation Learning
    ''' </summary>
    Private teacherRBPL As ResilientBackpropagationLearning

    ''' <summary>
    ''' Parallel Resilient Backpropagation Learning
    ''' </summary>
    Public PRBPLAlgo As Boolean = False
    Private teacherPRBPL As ParallelResilientBackpropagationLearning ' Less reliable?

    Private neuronCountWithoutInputLayer%()

    Public inputJaggedDblArray#()()
    Public targetJaggedDblArray#()()

    Public Overrides Function GetMLPType$()
        Return System.Reflection.MethodBase.GetCurrentMethod().DeclaringType.Name
    End Function

    Public Overrides Function GetActivationFunctionType() As enumActivationFunctionType
        Return enumActivationFunctionType.LibraryOptimized
    End Function

    Public Overrides Sub InitializeStruct(neuronCount%(), addBiasColumn As Boolean)

        MyBase.InitializeStruct(neuronCount, addBiasColumn)

        If Not Me.useBias Then
            Throw New NotImplementedException(
                "useBias=False is not implemented for clsAccordMLP!")
        End If

        Dim sigmoidAlphaValue! = Me.m_gain

        ' 2, 2, 1 :
        ' 2 : two inputs  in the network
        ' 2 : two neurons in the first layer
        ' 1 : one neuron  in the second layer
        'Dim neuronCountWithoutInputLayer%()
        ReDim Me.neuronCountWithoutInputLayer(0 To Me.layerCount - 2)
        For i = 1 To Me.layerCount - 1
            Me.neuronCountWithoutInputLayer(i - 1) = neuronCount(i)
        Next

        If IsNothing(Me.inputArray) Then Exit Sub
        Dim inputArrayDbl = clsMLPHelper.Convert2DArrayOfSingleToDouble(Me.inputArray)
        Me.inputJaggedDblArray = clsMLPHelper.Transform2DArrayToJaggedArray(inputArrayDbl)
        If IsNothing(Me.targetArray) Then Exit Sub
        Dim targetArrayDbl = clsMLPHelper.Convert2DArrayOfSingleToDouble(Me.targetArray)
        Me.targetJaggedDblArray = clsMLPHelper.Transform2DArrayToJaggedArray(targetArrayDbl)

    End Sub

    Public Overrides Sub SetActivationFunction(
        actFnc As enumActivationFunction, Optional gain! = 1, Optional center! = 0)

        Select Case actFnc
            Case enumActivationFunction.Sigmoid
                SetActivationFunctionOptimized(
                    enumActivationFunctionOptimized.Sigmoid, gain, center)
            Case enumActivationFunction.HyperbolicTangent
                SetActivationFunctionOptimized(
                    enumActivationFunctionOptimized.HyperbolicTangent, gain, center)
            Case Else
                Throw New NotImplementedException(
                    "This activation function is not available!")
        End Select

    End Sub

    Public Overrides Sub SetActivationFunctionOptimized(
        actFnc As enumActivationFunctionOptimized, Optional gain! = 1, Optional center! = 0)

        MyBase.SetActivationFunctionOptimized(actFnc, gain, center)

        Dim sigmoidAlphaValue! = Me.m_gain
        Select Case actFnc
            Case enumActivationFunctionOptimized.Sigmoid
                Me.m_actFunc = enumActivationFunction.Sigmoid
                Me.network = New ActivationNetwork(
                    New SigmoidFunction(sigmoidAlphaValue),
                        Me.nbInputNeurons, Me.neuronCountWithoutInputLayer)
            Case enumActivationFunctionOptimized.HyperbolicTangent
                Me.m_actFunc = enumActivationFunction.HyperbolicTangent
                Me.network = New ActivationNetwork(
                    New BipolarSigmoidFunction(sigmoidAlphaValue),
                        Me.nbInputNeurons, Me.neuronCountWithoutInputLayer)
            Case enumActivationFunctionOptimized.ELU
                Me.m_actFunc = Nothing
                Throw New NotImplementedException(
                    "ELU activation function is not available!")
            Case Else
                Me.activFnc = Nothing
        End Select

        If Me.trainingAlgorithm = enumTrainingAlgorithm.RProp AndAlso Not PRBPLAlgo Then
            Me.teacherRBPL = New ResilientBackpropagationLearning(Me.network)
            Me.teacherRBPL.LearningRate = Me.learningRate ' default value: 0.0125
            Me.weightAdjustment = 0
        ElseIf Me.trainingAlgorithm = enumTrainingAlgorithm.RProp AndAlso PRBPLAlgo Then
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
                ' The Threshold value is added to inputs weighted sum before
                '  it is passed to activation function
                neuron.Threshold = weights(j, nbWeights)
            Else
                neuron.Threshold = 0
            End If
        Next j

    End Sub

    Public Overrides Sub Randomize(Optional minValue! = -0.5!, Optional maxValue! = 0.5!)

        ' No Reset function for RBPL algo.
        If PRBPLAlgo Then Me.teacherPRBPL.Reset(Me.learningRate)

        ' Randomly initialize the network
        ' Random between -0.5 and +0.5 with statistical normalization 
        '  according to the network structure
        Dim x As New NguyenWidrow(Me.network)
        x.Randomize()

        ' Alternative: initialize the network with Gaussian weights
        ' Dim x As New GaussianWeights(Me.network, 0.1)
        ' x.Randomize()

        RoundWeights()

    End Sub

    Public Overrides Sub RoundWeights()

        ' Round the weights (to reproduce all tests exactly)

        For i = 0 To Me.layerCount - 2
            Dim layer = Me.network.Layers(i)
            Dim nbNeurons = layer.Neurons.Count
            For j = 0 To nbNeurons - 1
                Dim neuron = CType(layer.Neurons(j), ActivationNeuron)
                Dim nbWeights = neuron.Weights.Count
                For k = 0 To nbWeights - 1
                    Dim r = neuron.Weights(k)
                    Dim rounded = Math.Round(r, clsMLPGeneric.nbRoundingDigits)
                    neuron.Weights(k) = rounded
                Next k
                If Me.useBias Then
                    Dim r = neuron.Threshold
                    Dim rounded = Math.Round(r, clsMLPGeneric.nbRoundingDigits)
                    neuron.Threshold = rounded
                Else
                    neuron.Threshold = 0
                End If
            Next j
        Next i

    End Sub

    Public Overrides Sub TrainVector()

        Me.learningMode = enumLearningMode.Vectorial
        Me.vectorizedLearningMode = True
        For iteration = 0 To Me.nbIterations - 1
            Me.numIteration = iteration
            TrainVectorOneIteration()
            If Me.printOutput_ Then PrintOutput(iteration)
        Next
        SetOuput1D()
        ComputeAverageError()

    End Sub

    Public Overrides Sub TrainVectorOneIteration()

        Dim avgError#
        If Me.trainingAlgorithm = enumTrainingAlgorithm.RProp AndAlso Not PRBPLAlgo Then
            avgError = Me.teacherRBPL.RunEpoch(Me.inputJaggedDblArray, Me.targetJaggedDblArray)
        ElseIf Me.trainingAlgorithm = enumTrainingAlgorithm.RProp AndAlso PRBPLAlgo Then
            avgError = Me.teacherPRBPL.RunEpoch(Me.inputJaggedDblArray, Me.targetJaggedDblArray)
        Else
            avgError = Me.teacherBPL.RunEpoch(Me.inputJaggedDblArray, Me.targetJaggedDblArray)
        End If
        ' Does not work fine, too high!?
        'Me.averageError = avgError

    End Sub

    Public Overrides Sub SetOuput1D()

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

        Me.lastOutputArray1DSingle = clsMLPHelper.Convert1DArrayOfDoubleToSingle(outputs1D)
        Me.output = outputs2D

    End Sub

    Public Overrides Sub TrainOneSample(input!(), target!())

        Dim inputArrayDbl = clsMLPHelper.Convert1DArrayOfSingleToDouble(input)
        Dim targetArrayDbl = clsMLPHelper.Convert1DArrayOfSingleToDouble(target)

        Dim avgError#
        If Me.trainingAlgorithm = enumTrainingAlgorithm.RProp AndAlso Not PRBPLAlgo Then
            avgError = Me.teacherRBPL.Run(inputArrayDbl, targetArrayDbl)
        ElseIf Me.trainingAlgorithm = enumTrainingAlgorithm.RProp AndAlso PRBPLAlgo Then
            avgError = Me.teacherPRBPL.Run(inputArrayDbl, targetArrayDbl)
        Else
            avgError = Me.teacherBPL.Run(inputArrayDbl, targetArrayDbl)
        End If
        'Me.averageErrorOneSample = avgError

        Dim outputs#() = Me.network.Compute(inputArrayDbl)
        Me.lastOutputArray1DSingle = clsMLPHelper.Convert1DArrayOfDoubleToSingle(outputs)

    End Sub

    Public Overrides Sub TestOneSample(input!())

        Dim inputsDbl#() = clsMLPHelper.Convert1DArrayOfSingleToDouble(input)
        Dim outputs#() = Me.network.Compute(inputsDbl)
        Me.lastOutputArray1DSingle = clsMLPHelper.Convert1DArrayOfDoubleToSingle(outputs)

        ' 20/11/2020
        Dim outputs2D#(0, Me.nbOutputNeurons - 1)
        clsMLPHelper.Fill2DArrayOfDouble(outputs2D, outputs, 0)
        Me.output = outputs2D

    End Sub

    Public Overrides Function GetWeight!(layer%, neuron%, weight%)

        Dim layer_ = Me.network.Layers(layer - 1)
        Dim neuron_ = CType(layer_.Neurons(neuron), ActivationNeuron)
        Dim nbWeights = neuron_.Weights.Count
        Dim w#
        If weight < nbWeights Then
            w = neuron_.Weights(weight)
        Else
            w = neuron_.Threshold
        End If
        Dim wSng = CSng(w)
        Return wSng

    End Function

End Class
