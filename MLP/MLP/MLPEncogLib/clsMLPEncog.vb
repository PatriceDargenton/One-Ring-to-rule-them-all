
' From https://github.com/encog/encog-dotnet-core : C# -> VB .NET conversion
' https://www.nuget.org/packages/encog-dotnet-core
' <package id="encog-dotnet-core" version="3.4.0" targetFramework="net472" />
' https://www.heatonresearch.com/encog

' https://github.com/jeffheaton/encog-dotnet-core/issues/108 Porting to .Net Core #108
' The reason it is named "core" is because the library in Java, C#, C, and JavaScript that contain 
'  the core functions is named "core". (there is no relation to .Net core)
' Solution: disable NU1701 warning for the .Net core version of the project using .Net4.7 encog:
' (there is no .Net core version of encog, and the version 3.4.0 works fine with .Net core anyway)
' <PackageReference Include="encog-dotnet-core" Version="3.4.0">
'   <NoWarn>NU1701</NoWarn>
' </PackageReference>

Imports Encog.Engine.Network.Activation
Imports Encog.ML.Data
Imports Encog.ML.Data.Basic
Imports Encog.ML.Train
Imports Encog.Neural.Networks
Imports Encog.Neural.Networks.Layers
Imports Encog.Neural.Networks.Training.Propagation.Resilient ' ResilientPropagation
Imports Encog.Neural.Networks.Training.Propagation.Back ' Backpropagation

Imports System.Text

Public Class clsMLPEncog : Inherits clsVectorizedMLPGeneric

    Private network As BasicNetwork
    Private trainingSet As IMLDataSet
    Private imlTrain As IMLTrain

    Public inputJaggedDblArray#()()
    Public targetJaggedDblArray#()()

    Public Overrides Function GetMLPType$()
        Return System.Reflection.MethodBase.GetCurrentMethod().DeclaringType.Name
    End Function

    Public Overrides Function GetActivationFunctionType() As enumActivationFunctionType
        Return enumActivationFunctionType.Library
    End Function

    Public Overrides Sub InitializeStruct(neuronCount%(), addBiasColumn As Boolean)

        MyBase.InitializeStruct(neuronCount, addBiasColumn)
        Me.trainingAlgorithm = enumTrainingAlgorithm.RProp

        If IsNothing(Me.inputArray) Then Exit Sub
        Dim inputArrayDbl = clsMLPHelper.Convert2DArrayOfSingleToDouble(Me.inputArray)
        Me.inputJaggedDblArray = clsMLPHelper.Transform2DArrayToJaggedArray(inputArrayDbl)
        Dim targetArrayDbl = clsMLPHelper.Convert2DArrayOfSingleToDouble(Me.targetArray)
        Me.targetJaggedDblArray = clsMLPHelper.Transform2DArrayToJaggedArray(targetArrayDbl)

    End Sub

    Public Overrides Sub SetActivationFunction(
        actFnc As enumActivationFunction, Optional gain! = 1, Optional center! = 0)

        ' gain can only be 1 for Encog MLP
        gain = 1
        If actFnc = enumActivationFunction.HyperbolicTangent Then gain = 2
        center = 0

        If Me.trainingAlgorithm = enumTrainingAlgorithm.RProp Then
            Me.weightAdjustment = 0 ' Not used
            Me.learningRate = 0 ' Learning rate is not use with ResilientPropagation:
            ' http://heatonresearch-site.s3-website-us-east-1.amazonaws.com/javadoc/encog-3.3/org/encog/neural/networks/training/propagation/resilient/ResilientPropagation.html
            ' One problem with the backpropagation algorithm is that the magnitude of the 
            '  partial derivative is usually too large or too small. Further, the learning
            '  rate is a single value for the entire neural network. The resilient propagation
            '  learning algorithm uses a special update value (similar to the learning rate)
            '  for every neuron connection. Further these update values are automatically
            '  determined, unlike the learning rate of the backpropagation algorithm.
        End If

        MyBase.SetActivationFunction(actFnc, gain, center)

        If IsNothing(Me.inputJaggedDblArray) Then Exit Sub
        If Me.inputJaggedDblArray.Length = 0 Then Exit Sub

        Me.network = New BasicNetwork()

        ' Input layer
        Me.network.AddLayer(New BasicLayer(Nothing, Me.useBias,
            neuronCount:=Me.neuronCount(0)))

        'Dim act As Encog.Engine.Network.Activation.IActivationFunction
        Dim act As IActivationFunction
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

        ' Reset the weight matrix and the bias values: random weights to start
        Me.network.Reset()

        Me.trainingSet = New BasicMLDataSet(Me.inputJaggedDblArray, Me.targetJaggedDblArray)

        If Me.trainingAlgorithm = enumTrainingAlgorithm.RProp Then
            ' maxStep: The maximum that a delta can reach
            Me.imlTrain = New ResilientPropagation(Me.network, Me.trainingSet,
                initialUpdate:=0.1#, maxStep:=50.0#)
        Else
            Me.imlTrain = New Backpropagation(Me.network, Me.trainingSet,
                Me.learningRate, momentum:=Me.weightAdjustment)
        End If

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

    Public Overrides Sub Randomize(Optional minValue! = -0.5!, Optional maxValue! = 0.5!)

        ' Reset the weight matrix and the bias values: random weights to start
        Me.network.Reset()

        RoundWeights()

    End Sub

    Public Overrides Sub RoundWeights()

        ' Round the weights (to reproduce all tests exactly)
        For i = 1 To Me.layerCount - 1
            Dim nbNeuronsLayer = Me.network.GetLayerNeuronCount(i)
            Dim nbNeuronsPreviousLayer = Me.network.GetLayerNeuronCount(i - 1)
            For j = 0 To nbNeuronsLayer - 1
                Dim nbWeights = nbNeuronsPreviousLayer
                For k = 0 To nbWeights - 1
                    Dim weight = Me.network.GetWeight(i - 1, k, j)
                    Dim rounded = Math.Round(weight, clsMLPGeneric.nbRoundingDigits)
                    Me.network.SetWeight(i - 1, k, j, rounded)
                Next k
                If Me.useBias Then
                    Dim weightT = Me.network.GetWeight(i - 1, nbWeights, j)
                    Dim rounded = Math.Round(weightT, clsMLPGeneric.nbRoundingDigits)
                    Me.network.SetWeight(i - 1, nbWeights, j, rounded)
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

        CloseTrainingSession() ' 21/11/2020

        SetOuput1D()
        'ComputeError()
        ComputeAverageError() ' 14/11/2020

    End Sub

    Public Overrides Sub CloseTrainingSession()

        ' Should be called once training is complete and no more iterations are
        ' needed. Calling iteration again will simply begin the training again, and
        ' require finishTraining to be called once the new training session is
        ' complete.
        ' It is particularly important to call finishTraining for multithreaded
        ' training techniques.
        Me.imlTrain.FinishTraining()

    End Sub

    Public Overrides Sub TrainVectorOneIteration()

        Me.imlTrain.Iteration()
        Me.averageError = Me.imlTrain.Error

    End Sub

    Public Overrides Sub SetOuput1D()

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

    Public Overrides Function GetWeight!(layer%, neuron%, weight%)

        Dim weightDbl = Me.network.GetWeight(layer - 1, weight, neuron)
        Dim weightSng = CSng(weightDbl)
        Return weightSng

    End Function

End Class