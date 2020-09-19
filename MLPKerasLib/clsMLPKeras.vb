﻿
' From  : https://github.com/SciSharp/Keras.NET : C# -> VB .NET conversion
' https://www.nuget.org/packages/Keras.NET Nuget install
' https://scisharp.github.io/SciSharp Other .NET Machine Learning projects
' https://scisharp.github.io/Keras.NET Documentation
' https://keras.io Documentation (not only for Keras.NET but Keras)

' Keras.NET -> packages added:
' <package id = "Keras.NET" version="3.7.4.2" targetFramework="net472" />
' <package id = "Microsoft.CSharp" version="4.5.0" targetFramework="net472" />
' <package id = "Numpy.Bare" version="3.7.1.11" targetFramework="net472" />
' <package id = "Python.Runtime.NETStandard" version="3.7.1" targetFramework="net472" />
' <package id = "System.Reflection.Emit" version="4.3.0" targetFramework="net472" />

' Python 3.7 is required at runtime:
' https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe
' https://www.python.org/downloads
' For PowerShell installations, type:
' python -mpip install numpy      : fix "No module named 'numpy'"
' python -mpip install keras      : fix "No module named 'keras'"
' python -mpip install tensorflow : fix "Keras requires TensorFlow 2.2 or higher"

Imports Keras.Layers
Imports Keras.Models
Imports Keras.Optimizers
Imports Numpy

Imports System.Text

Public Class clsMLPKeras : Inherits clsVectorizedMLPGeneric

    Private inputNDA, targetNDA As NDarray
    Private model As Model
    Dim outputs1D#()
    Dim weightsNDA As List(Of NDarray)
    Private Const batch_size% = 2

    Public inputJaggedDblArray#()()
    Public targetJaggedDblArray#()()

    Public Overrides Sub InitializeStruct(neuronCount%(), addBiasColumn As Boolean)

        Dim inputArrayDbl = clsMLPHelper.Convert2DArrayOfSingleToDouble(Me.inputArray)
        Me.inputJaggedDblArray = clsMLPHelper.Transform2DArrayToJaggedArray(inputArrayDbl)
        Dim targetArrayDbl = clsMLPHelper.Convert2DArrayOfSingleToDouble(Me.targetArray)
        Me.targetJaggedDblArray = clsMLPHelper.Transform2DArrayToJaggedArray(targetArrayDbl)

        Me.layerCount = neuronCount.Length
        Me.useBias = False 'addBiasColumn
        Me.neuronCount = neuronCount
        Me.nbInputNeurons = Me.neuronCount(0)
        Me.nbHiddenNeurons = Me.neuronCount(1)
        Me.nbOutputNeurons = Me.neuronCount(Me.layerCount - 1)

    End Sub

    Public Overrides Sub SetActivationFunction(
        actFnc As enumActivationFunction, Optional gain! = 1, Optional center! = 0)

        gain = 1 ' gain can only be 1 for Keras MLP (sigmoid)
        If actFnc = enumActivationFunction.HyperbolicTangent Then gain = 2
        center = 0
        'Me.weightAdjustment = 0 ' Not used

        MyBase.SetActivationFunction(actFnc, gain, center)

        Me.ShowMessage(Now() & " Initializing python engine...")

        ' Load train data
        Me.inputNDA = Me.inputArray
        Me.targetNDA = np.array(Me.targetArray, dtype:=np.float32)
        Me.exampleCount = Me.inputArray.GetLength(0)
        'batch_size = Me.exampleCount ' Does not work?

        ' Build functional model

        ' Minimal XOR example:
        'Dim hidden1 = New Dense(Me.nbHiddenNeurons, activation:="relu").Set(input)
        'Dim hidden2 = New Dense(Me.nbHiddenNeurons, activation:="relu").Set(hidden1)
        'Dim outputLayer = New Dense(Me.nbOutputNeurons, activation:="sigmoid").Set(hidden2)

        Dim input = New Input(shape:=New Keras.Shape(Me.nbInputNeurons))
        Dim prevHidden As BaseLayer = Nothing
        Dim lastHidden As BaseLayer = Nothing
        Dim nbNeurons = Me.nbHiddenNeurons
        For i = 0 To Me.layerCount - 3
            If i = 0 Then prevHidden = input

            lastHidden = New Dense(Me.nbHiddenNeurons, activation:="relu").Set(prevHidden)

            ' relu works better than sigmoid or tanh in hidden layer(s):
            'lastHidden = New Dense(Me.nbHiddenNeurons, activation:="sigmoid").Set(prevHidden)
            'Select Case Me.m_actFunc
            '    Case enumActivationFunction.Sigmoid
            '        lastHidden = New Dense(Me.nbHiddenNeurons,
            '            activation:="sigmoid").Set(prevHidden)
            '    Case enumActivationFunction.HyperbolicTangent
            '        lastHidden = New Dense(Me.nbHiddenNeurons,
            '            activation:="tanh").Set(prevHidden)
            '    Case Else
            '        Throw New NotImplementedException(
            '            "This activation function is not available!")
            'End Select

            prevHidden = lastHidden

        Next

        Dim outputLayer As BaseLayer = Nothing
        Select Case Me.m_actFunc
            Case enumActivationFunction.Sigmoid
                outputLayer = New Dense(Me.nbOutputNeurons,
                    activation:="sigmoid").Set(lastHidden)
            Case enumActivationFunction.HyperbolicTangent
                outputLayer = New Dense(Me.nbOutputNeurons,
                    activation:="tanh").Set(lastHidden)
            Case Else
                Throw New NotImplementedException(
                    "This activation function is not available!")
        End Select

        Me.model = New Keras.Models.Model(
            New Input() {input},
            New BaseLayer() {outputLayer})

        ' Other optimizer:
        'Me.model.Compile(
        '    optimizer:=New Adam(lr:=Me.learningRate),
        '    loss:="binary_crossentropy",
        '    metrics:=New String() {"accuracy"})

        Me.model.Compile(optimizer:=New SGD(
            lr:=Me.learningRate, momentum:=Me.weightAdjustment),
            loss:="binary_crossentropy",
            metrics:=New String() {"accuracy"})
        ' Other possible loss: mse

        Me.ShowMessage(Now() & " Initializing python engine: Done.")

    End Sub

    Public Overrides Sub InitializeWeights(numLayer%, weights#(,))

        If numLayer = 1 Then Me.weightsNDA = New List(Of NDarray)

        Dim nbWeightLayers = (Me.layerCount - 1) * 2
        Dim nda As NDarray

        If numLayer Mod 2 = 0 Then
            Dim asng = clsMLPHelper.Transform2DArrayToJaggedArray(weights)
            nda = asng(0)
        Else
            nda = weights
        End If

        Me.weightsNDA.Add(nda)

        If numLayer = nbWeightLayers Then Me.model.SetWeights(Me.weightsNDA)

    End Sub

    Public Overrides Sub Randomize(Optional minValue! = 0, Optional maxValue! = 1)

        ' Round the weights (to reproduce all tests exactly)

        Dim weights = Me.model.GetWeights()
        Dim nbWeightLayers = (Me.layerCount - 1) * 2
        For i = 0 To nbWeightLayers - 1
            Dim ws = weights(i).GetData(Of Single)()
            Dim nbItems = ws.GetUpperBound(0) + 1
            For j = 0 To nbItems - 1
                Dim weight = ws(j)
                Dim rounded = Math.Round(weight, clsMLPGeneric.roundWeights)
                ws(j) = CSng(rounded)
            Next
        Next

    End Sub

    Public Overrides Sub TrainVector()
        Throw New NotImplementedException(
            "Use TrainVectorBatch(nbIterationsBatch)!")
    End Sub

    Public Overrides Sub TrainVectorBatch(nbIterationsBatch%)

        Me.vectorizedLearningMode = True

        Dim history = model.Fit(
            Me.inputNDA, Me.targetNDA, batch_size:=batch_size,
            epochs:=nbIterationsBatch, verbose:=0)
        Dim err = history.HistoryLogs("loss").GetValue(0)
        Me.averageError = CSng(err)

    End Sub

    Public Overrides Sub TrainVectorOneIteration()
        Throw New NotImplementedException(
            "Use TrainVectorBatch(nbIterationsBatch)!")
    End Sub

    Public Overrides Sub SetOuput1D()

        ' ForwardPropogateSignal
        Dim score = model.Predict(Me.inputNDA, verbose:=0)
        Dim outputs!() = score.GetData(Of Single)()

        Dim nbInputs = Me.inputArray.GetLength(0)
        Dim nbTargets = Me.targetArray.GetLength(0)
        Dim lengthTot = nbTargets * Me.nbOutputNeurons
        Dim outputs2D#(nbTargets - 1, Me.nbOutputNeurons - 1)
        Dim k = 0
        For i = 0 To nbInputs - 1
            For j = 0 To Me.nbOutputNeurons - 1
                outputs2D(i, j) = outputs(i * Me.nbOutputNeurons + j)
                k += 1
            Next
        Next
        Me.lastOutputArray1DSingle = outputs
        Me.output = outputs2D

    End Sub

    Public Sub SetOuput1DOneSample()

        ' ForwardPropogateSignal
        Dim score = model.Predict(Me.inputNDA, verbose:=0)
        Dim outputs!() = score.GetData(Of Single)()

        Dim nbInputs = 1
        Dim nbTargets = 1
        Dim lengthTot = nbTargets * Me.nbOutputNeurons
        Dim outputs2D#(nbTargets - 1, Me.nbOutputNeurons - 1)
        Dim k = 0
        For i = 0 To nbInputs - 1
            For j = 0 To Me.nbOutputNeurons - 1
                outputs2D(i, j) = outputs(i * Me.nbOutputNeurons + j)
                k += 1
            Next
        Next
        Me.lastOutputArray1DSingle = outputs
        Me.output = outputs2D

    End Sub

    Public Overrides Sub TrainSystematic(inputs!(,), targets!(,),
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        ' This is the unique learning mode for this MLP
        TrainVectorBatch()

    End Sub

    Public Overrides Sub TrainStochastic(inputs!(,), targets!(,))
        ' TrainStochastic requires TrainOneSample
        ' Possibility: shuffle:= True in model.fit()
        Throw New NotImplementedException("There is no TrainOneSample() function!")
    End Sub

    Public Overrides Sub TrainSemiStochastic(inputs!(,), targets!(,))
        ' TrainSemiStochastic requires TrainOneSample
        Throw New NotImplementedException("There is no TrainOneSample() function!")
    End Sub

    Public Overrides Sub TrainOneSample(input!(), target!())
        Throw New NotImplementedException("There is no TrainOneSample() function!")
    End Sub

    Public Overrides Sub TestOneSample(input!())

        Dim inputsDble#(0, input.Length - 1)
        clsMLPHelper.Fill2DArrayOfDoubleByArrayOfSingle(inputsDble, input, 0)
        Me.inputNDA = inputsDble
        SetOuput1DOneSample()

    End Sub

    Public Overrides Sub PrintWeights()

        Me.PrintParameters()

        For i = 0 To Me.layerCount - 1
            ShowMessage("Neuron count(" & i & ")=" & Me.neuronCount(i))
        Next

        ShowMessage("")

        Dim weights = Me.model.GetWeights()
        Dim nbWeightLayers = (Me.layerCount - 1) * 2
        Dim ws(nbWeightLayers - 1)() As Single
        For i = 0 To nbWeightLayers - 1
            ws(i) = weights(i).GetData(Of Single)()
        Next

        Dim sb As New StringBuilder
        For i = 1 To nbWeightLayers

            sb.AppendLine("W(" & i & ")={")

            Dim wsi = ws(i - 1)
            Dim nbItems = wsi.GetUpperBound(0) + 1
            Dim nbNeuronsLayer = weights(i - 1).len
            Dim nbNeuronsPreviousLayer = nbItems \ nbNeuronsLayer
            Dim oneDim = (i Mod 2 = 0)
            Dim l = 0
            Dim lMax = wsi.GetUpperBound(0) + 1
            For j = 0 To nbNeuronsLayer - 1
                sb.Append(" ")
                If Not oneDim Then sb.Append("{")

                Dim nbWeights = nbNeuronsPreviousLayer
                For k = 0 To nbWeights - 1
                    Dim weight = wsi(l)
                    Dim sVal$ = weight.ToString(format2Dec).ReplaceCommaByDot()
                    sb.Append(sVal)
                    If Me.useBias OrElse k < nbWeights - 1 Then sb.Append(", ")
                    l += 1
                Next k

                If Not oneDim Then sb.Append("}")
                If j < nbNeuronsLayer - 1 Then
                    sb.Append(",")
                    If Not oneDim Then sb.Append(vbLf)
                End If
            Next j
            sb.Append("}" & vbLf)

            If i < nbWeightLayers Then sb.AppendLine()

        Next i

        ShowMessage(sb.ToString())

    End Sub

End Class