
' https://www.nuget.org/packages/NeuralNetwork.NET Nuget install
' https://github.com/Sergio0694/NeuralNetwork.NET
' https://scisharp.github.io/SciSharp Other .NET Machine Learning projects

' Install-Package NeuralNetwork.NET
' Fix SixLabors.ImageSharp FileLoadException (0x80131040) bug:
' Install-Package SixLabors.ImageSharp -Version 1.0.0-beta0007
' (bug starting from 1.0.0-rc0001 version)

Imports NeuralNetworkNET.APIs
Imports NeuralNetworkNET.APIs.Enums
Imports NeuralNetworkNET.APIs.Structs
Imports NeuralNetworkNET.Networks.Cost ' CostFunctionType

' BC40025: Type of this member is not CLS-compliant:
'Imports NeuralNetworkNET.SupervisedLearning.Algorithms ' TrainingAlgorithmType

Imports System.Text

' Without GetWeights, SetWeights, functional tests will fails
#Const NuGetImplementation = 1 ' 0: Off, 1: On

' GetWeights, SetWeights, see there:
' https://github.com/PatriceDargenton/NeuralNetwork.NET/tree/get_set_weights
#Const GetWeightsImplementation = 0 ' 0: Off, 1: On

' Tuples are not available for Visual Studio 2013:
' https://github.com/PatriceDargenton/NeuralNetwork.NET/tree/training_input_target
#Const GetWeightsImplementationVS2013 = 0 ' 0: Off, 1: On

' The stable branch contains the get_set_weights and training_input_target branches:
' https://github.com/PatriceDargenton/NeuralNetwork.NET/tree/stable

Public Class clsMLPNeuralNet : Inherits clsVectorizedMLPGeneric

    '#Disable Warning BC40025 ' Type of this member is not CLS-compliant
    'Public trainingAlgorithm As TrainingAlgorithmType = TrainingAlgorithmType.RMSProp
    '#Enable Warning BC40025
    Public Enum TrainingAlgorithmType 'As Integer

        ''' <summary>
        ''' Undefined
        ''' </summary>
        Undefined

        ''' <summary>
        ''' The plain stochastic gradient descent training algorithm
        ''' </summary>
        StochasticGradientDescent

        ''' <summary>
        ''' A variant of the stochastic gradient descent algorithm with momentum
        ''' </summary>
        Momentum

        ''' <summary>
        ''' The AdaGrad learning method, by John Duchi, Elad Hazan and Yoram Singer, see http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
        ''' </summary>
        AdaGrad

        ''' <summary>
        ''' The AdaDelta adaptive learning method, by Matthew D. Zeiler, see https://arxiv.org/abs/1212.5701
        ''' </summary>
        AdaDelta

        ''' <summary>
        ''' The Adam learning method, by Diederik P. Kingma and Jimmy Lei Ba, see https://arxiv.org/pdf/1412.6980v8.pdf
        ''' </summary>
        Adam

        ''' <summary>
        ''' The AdaMax learning method, by Diederik P. Kingma and Jimmy Lei Ba, see section 7.1 of https://arxiv.org/pdf/1412.6980v8.pdf
        ''' </summary>
        AdaMax

        ''' <summary>
        ''' The RMSProp learning method, see http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        ''' </summary>
        RMSProp

    End Enum
    Public trainingAlgorithm As TrainingAlgorithmType = TrainingAlgorithmType.RMSProp

    Private network As Interfaces.INeuralNetwork
    Private output2D!(,)

    Public inputJaggedArray!()()
    Public targetJaggedArray!()()

    Private m_weights!()()
    Private m_biases!()()

#If GetWeightsImplementationVS2013 Then
#ElseIf GetWeightsImplementation Or NuGetImplementation Then
    Private m_trainingData As (n As Single(), w As Single())()
#End If
    Private m_dataset As Interfaces.Data.ITrainingDataset
    Private m_nbIterationsBatchLast%

    Public Overrides Function GetActivationFunctionType() As enumActivationFunctionType
        Return enumActivationFunctionType.LibraryOptimized
    End Function

    Public Overrides Sub InitializeStruct(neuronCount%(), addBiasColumn As Boolean)

        Me.useBias = addBiasColumn
        If Not Me.useBias Then
            Throw New NotImplementedException(
                "useBias=False is not implemented for clsMLPNeuralNet!")
        End If

        MyBase.InitializeStruct(neuronCount, addBiasColumn)
        Me.learningRate = 0
        Me.weightAdjustment = 0

        Me.minBatchSize = 1
#If NuGetImplementation Then
        Me.minBatchSize = 10
#End If

        Me.nbIterationsBatch = 100

    End Sub

    Public Overrides Sub SetActivationFunction(
        actFnc As enumActivationFunction, Optional gain! = 1, Optional center! = 0)

        center = 0
        If actFnc = enumActivationFunction.Sigmoid Then gain = 1
        If actFnc = enumActivationFunction.HyperbolicTangent Then gain = 2
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

        center = 0
        If actFnc = enumActivationFunctionOptimized.Sigmoid Then gain = 1
        If actFnc = enumActivationFunctionOptimized.HyperbolicTangent Then gain = 2
        MyBase.SetActivationFunctionOptimized(actFnc, gain, center)

        Dim actFunc = ActivationType.Tanh
        Select Case Me.m_actFunc
            Case enumActivationFunction.Sigmoid : actFunc = ActivationType.Sigmoid
            Case enumActivationFunction.HyperbolicTangent : actFunc = ActivationType.Tanh
            Case Else : Throw New NotImplementedException("This activation function is not available!")
        End Select

        If IsNothing(Me.inputArray) Then Exit Sub
        Me.inputJaggedArray = clsMLPHelper.Transform2DArrayToJaggedArraySingle(Me.inputArray)
        Me.targetJaggedArray = clsMLPHelper.Transform2DArrayToJaggedArraySingle(Me.targetArray)

#If GetWeightsImplementationVS2013 Then
        ' Zip function compiled in .NET Standard 2.0 NeuralNetwork.NET.dll, available from VS2013
        Me.m_dataset = DatasetLoader.Training2(Me.inputJaggedArray, Me.targetJaggedArray, size:=Me.nbIterationsBatch)
#ElseIf GetWeightsImplementation Or NuGetImplementation Then
        ' Zip: just concatenate the input array with the target array like a zipper!
        ' var trainingData = Enumerable.Zip(input, target).ToArray();                   .NET Core 3.1
        ' var trainingData = Enumerable.Zip(input, target, (n, c) => (n, c)).ToArray(); .NET Framework 4.7.2
        ' var trainingData = input.Zip(target, (n, c) => (n, c)).ToArray();             .NET Standard 2.0
        ' https://docs.microsoft.com/fr-fr/dotnet/api/system.linq.enumerable.zip?view=net-5.0
        Dim zip = Me.inputJaggedArray.Zip(Me.targetJaggedArray, Function(n, w) (n, w))
        Me.m_trainingData = zip.ToArray()
        ' size: The desired dataset batch size
        Me.m_dataset = DatasetLoader.Training(Me.m_trainingData, size:=Me.nbIterationsBatch)
#End If
        Me.m_nbIterationsBatchLast = Me.nbIterationsBatch

        BuildGraph()

    End Sub

    Private Sub BuildGraph()

        Dim actFunc = ActivationType.Tanh
        Select Case Me.m_actFunc
            Case enumActivationFunction.Sigmoid : actFunc = ActivationType.Sigmoid
            Case enumActivationFunction.HyperbolicTangent : actFunc = ActivationType.Tanh
            Case Else : Throw New NotImplementedException("This activation function is not available!")
        End Select

        Select Case Me.layerCount
            Case 3
                Me.network = NetworkManager.NewSequential(
                    TensorInfo.Linear(Me.nbInputNeurons),
                    NetworkLayers.FullyConnected(Me.neuronCount(1), actFunc),
                    NetworkLayers.FullyConnected(Me.nbOutputNeurons, actFunc,
                        CostFunctionType.Quadratic))
            Case 4
                Me.network = NetworkManager.NewSequential(
                    TensorInfo.Linear(Me.nbInputNeurons),
                    NetworkLayers.FullyConnected(Me.neuronCount(1), actFunc),
                    NetworkLayers.FullyConnected(Me.neuronCount(2), actFunc),
                    NetworkLayers.FullyConnected(Me.nbOutputNeurons, actFunc,
                        CostFunctionType.Quadratic))
            Case 5
                Me.network = NetworkManager.NewSequential(
                    TensorInfo.Linear(Me.nbInputNeurons),
                    NetworkLayers.FullyConnected(Me.neuronCount(1), actFunc),
                    NetworkLayers.FullyConnected(Me.neuronCount(2), actFunc),
                    NetworkLayers.FullyConnected(Me.neuronCount(3), actFunc),
                    NetworkLayers.FullyConnected(Me.nbOutputNeurons, actFunc,
                        CostFunctionType.Quadratic))
            Case Else : Throw New NotImplementedException("Wrong number of layers!")
        End Select

    End Sub

#If GetWeightsImplementation Or GetWeightsImplementationVS2013 Then

    Public Overrides Sub InitializeWeights(layer%, weights#(,))

        If layer = 1 Then
            ReDim Me.m_weights(0 To Me.layerCount - 2)
            ReDim Me.m_biases(0 To Me.layerCount - 2)
        End If

        Dim i = layer - 1
        Dim nbNeuronsLayer = Me.neuronCount(i + 1)
        Dim nbBiases = nbNeuronsLayer
        Dim n = nbBiases * Me.neuronCount(i)
        ReDim Me.m_weights(i)(0 To n - 1)
        ReDim Me.m_biases(i)(0 To nbBiases - 1)
        For j = 0 To nbBiases - 1
            Me.m_biases(i)(j) = 0
        Next j
        Dim nbNeuronsPreviousLayer = Me.neuronCount(i)
        Dim l = 0
        For j = 0 To nbNeuronsLayer - 1
            Dim nbWeights = nbNeuronsPreviousLayer
            For k = 0 To nbWeights - 1
                Dim r = weights(j, k)
                Me.m_weights(i)(l) = CSng(r)
                l += 1
            Next k
            If Me.useBias Then
                Dim r = weights(j, nbWeights)
                Me.m_biases(i)(j) = CSng(r)
            End If
        Next j

        If layer = Me.layerCount - 1 Then
            BuildGraph()
            Me.network.SetWeights(Me.m_weights, Me.m_biases)
        End If

    End Sub

    Public Sub GetWeights()
        Me.network.GetWeights(Me.m_weights, Me.m_biases)
    End Sub

    Public Sub SetWeights()
        Me.network.SetWeights(Me.m_weights, Me.m_biases)
    End Sub

#Else

    Public Overrides Sub InitializeWeights(layer%, weights#(,))

        ShowMessage("GetWeights, SetWeights, see there:")
        ShowMessage("https://github.com/PatriceDargenton/NeuralNetwork.NET/tree/stable")

    End Sub

    Public Sub GetWeights()
    End Sub
    Public Sub SetWeights()
    End Sub

#End If

    Public Overrides Sub Randomize(Optional minValue! = -0.5!, Optional maxValue! = 0.5!)

        ' Re-build the graphe to randomize again the network!
        BuildGraph()

        ' Round the weights (to reproduce all tests exactly)
        RoundWeights()

    End Sub

    Public Sub ReDimWeights()
        'If IsNothing(Me.m_weights) OrElse IsNothing(Me.m_biases) Then
        ReDim Me.m_weights(0 To Me.layerCount - 2)
        ReDim Me.m_biases(0 To Me.layerCount - 2)
        'End If
        For i = 0 To Me.layerCount - 2
            Dim nbBiases = Me.neuronCount(i + 1)
            Dim n = nbBiases * Me.neuronCount(i)
            ReDim Me.m_weights(i)(0 To n - 1)
            ReDim Me.m_biases(i)(0 To nbBiases - 1)
        Next
    End Sub

    Public Overrides Sub RoundWeights()

        ReDimWeights()
        GetWeights()

        For i = 0 To Me.network.Layers.Count - 1

            Dim nbBiases = Me.neuronCount(i + 1)
            Dim nbWeights = nbBiases * Me.neuronCount(i)

            For k = 0 To nbWeights - 1
                Dim weight = Me.m_weights(i)(k)
                Dim rounded = Math.Round(weight, clsMLPGeneric.nbRoundingDigits)
                Me.m_weights(i)(k) = CSng(rounded)
            Next k

            If Me.useBias Then
                For k = 0 To nbBiases - 1
                    Dim weightT = Me.m_biases(i)(k)
                    Dim rounded = Math.Round(weightT, clsMLPGeneric.nbRoundingDigits)
                    Me.m_biases(i)(k) = CSng(rounded)
                Next k
            End If

        Next

        SetWeights()

    End Sub

#If NuGetImplementation Then

    Public Overrides Sub TrainSystematic(inputs!(,), targets!(,),
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        ' This is the unique learning mode for this MLP
        TrainVectorBatch()

    End Sub

    Public Overrides Sub TrainVector()
        Throw New NotImplementedException(
            "Use TrainVectorBatch(nbIterationsBatch)!")
    End Sub

    Public Overrides Sub TrainVectorOneIteration()
        Throw New NotImplementedException("Use TrainVectorBatch(nbIterationsBatch)!")
    End Sub

#Else

    Public Overrides Sub TrainSystematic(inputs!(,), targets!(,),
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        If learningMode = enumLearningMode.Vectorial Then TrainVector() : Exit Sub

        ' This is the defaut learning mode for this MLP
        TrainVectorBatch()

    End Sub

    Public Overrides Sub TrainVector()

        Me.learningMode = enumLearningMode.Vectorial
        Me.vectorizedLearningMode = True

        If nbIterationsBatch <> Me.m_nbIterationsBatchLast Then
            Me.m_dataset = DatasetLoader.Training2(Me.inputJaggedArray, Me.targetJaggedArray, size:=1)
        End If

        For iteration = 0 To Me.nbIterations - 1
            TrainVectorOneIteration()
            If Me.printOutput_ Then PrintOutput(iteration)
        Next
        SetOuput1D()
        ComputeAverageError()

    End Sub

    Public Overrides Sub TrainVectorOneIteration()

        TrainNetwork(Me.m_dataset)

    End Sub

#End If

#If GetWeightsImplementationVS2013 Then

    Public Overrides Sub TrainVectorBatch(nbIterationsBatch%)

        Me.vectorizedLearningMode = True

        Dim dataset As Interfaces.Data.ITrainingDataset
        If nbIterationsBatch = Me.m_nbIterationsBatchLast Then
            dataset = Me.m_dataset
        Else
            ' size: The desired dataset batch size
            dataset = DatasetLoader.Training2(Me.inputJaggedArray, Me.targetJaggedArray, size:=nbIterationsBatch)
        End If

        TrainNetwork(dataset)

    End Sub

#ElseIf GetWeightsImplementation Or NuGetImplementation Then

    Public Overrides Sub TrainVectorBatch(nbIterationsBatch%)

        Me.vectorizedLearningMode = True

        Dim dataset As Interfaces.Data.ITrainingDataset
        If nbIterationsBatch = Me.m_nbIterationsBatchLast Then
            dataset = Me.m_dataset
        Else
            ' size: The desired dataset batch size
            dataset = DatasetLoader.Training(Me.m_trainingData, size:=nbIterationsBatch)
        End If

        TrainNetwork(dataset)

    End Sub

#Else

    Public Overrides Sub TrainVectorBatch(nbIterationsBatch%)

        ' VS2013: You need DatasetLoader.Training2 function there:
        ' https://github.com/PatriceDargenton/NeuralNetwork.NET/tree/stable
        ' dataset = DatasetLoader.Training2(Me.inputJaggedArray, Me.targetJaggedArray, size:=nbIterationsBatch)

        Throw New NotImplementedException("A special NeuralNetwork.NET dll version is needed!")

    End Sub

#End If

    Private Sub TrainNetwork(dataset As Interfaces.Data.ITrainingDataset)

        Select Case Me.trainingAlgorithm
            Case TrainingAlgorithmType.AdaMax
                NetworkManager.TrainNetworkAsync(Me.network, dataset,
                    TrainingAlgorithms.AdaMax(), epochs:=nbIterationsBatch).Wait()
            Case TrainingAlgorithmType.AdaGrad
                NetworkManager.TrainNetworkAsync(Me.network, dataset,
                    TrainingAlgorithms.AdaGrad(), epochs:=nbIterationsBatch).Wait()
            Case TrainingAlgorithmType.Adam
                NetworkManager.TrainNetworkAsync(Me.network, dataset,
                    TrainingAlgorithms.Adam(), epochs:=nbIterationsBatch).Wait()
            Case TrainingAlgorithmType.Momentum
                NetworkManager.TrainNetworkAsync(Me.network, dataset,
                    TrainingAlgorithms.Momentum(), epochs:=nbIterationsBatch).Wait()
            Case TrainingAlgorithmType.RMSProp
                NetworkManager.TrainNetworkAsync(Me.network, dataset,
                    TrainingAlgorithms.RMSProp(), epochs:=nbIterationsBatch).Wait()
            Case TrainingAlgorithmType.AdaDelta
                NetworkManager.TrainNetworkAsync(Me.network, dataset,
                    TrainingAlgorithms.AdaDelta(), epochs:=nbIterationsBatch).Wait()
            Case TrainingAlgorithmType.StochasticGradientDescent
                NetworkManager.TrainNetworkAsync(Me.network, dataset,
                    TrainingAlgorithms.StochasticGradientDescent(), epochs:=nbIterationsBatch).Wait()
            Case Else : Throw New NotImplementedException(
                    "This training algorithm is not available!")
        End Select

    End Sub

    Private Sub Forward()

        Me.output2D = Me.network.Forward(Me.inputArray)
        'Dim r1 = Me.output2D(0, 0)
        'If Single.IsNaN(r1) Then Debug.WriteLine(Now & " : Error found: NaN!")

    End Sub

    Public Overrides Sub SetOuput1D()

        Forward()

        Dim nbInputs = Me.inputArray.GetLength(0)
        Dim nbTargets = Me.targetArray.GetLength(0)
        Dim lengthTot = nbTargets * Me.nbOutputNeurons
        Dim outputs1D#(lengthTot - 1)
        Dim outputs2D#(nbTargets - 1, Me.nbOutputNeurons - 1)
        Dim k = 0
        For i = 0 To nbInputs - 1
            For j = 0 To Me.nbOutputNeurons - 1
                outputs2D(i, j) = Me.output2D(i, j)
                outputs1D(k) = Me.output2D(i, j)
                k += 1
            Next
        Next
        Me.lastOutputArray1DSingle = clsMLPHelper.Convert1DArrayOfDoubleToSingle(outputs1D)
        Me.output = outputs2D

    End Sub

    Public Sub SetOuput1DOneSample()

        Dim nbInputs = 1
        Dim nbTargets = 1
        Dim lengthTot = nbTargets * Me.nbOutputNeurons
        Dim outputs!(lengthTot - 1)
        Dim k = 0
        For i = 0 To nbInputs - 1
            For j = 0 To Me.nbOutputNeurons - 1
                outputs(i * Me.nbOutputNeurons + j) = Me.output2D(i, j)
                k += 1
            Next
        Next
        Me.lastOutputArray1DSingle = outputs
        Me.output = Me.output2D

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

        Dim inputs2D!(0, input.Length - 1)
        clsMLPHelper.Fill2DArrayOfSingle(inputs2D, input, 0)
        Me.output2D = Me.network.Forward(inputs2D)

        SetOuput1DOneSample()

    End Sub

#If GetWeightsImplementation Or GetWeightsImplementationVS2013 Then

    Public Overrides Function ShowWeights$()

        Dim sb As New StringBuilder
        sb.AppendLine("Training algorithm=" & clsMLPHelper.ReadEnumDescription(Me.trainingAlgorithm))
        sb.AppendLine("nb iterations batch=" & Me.nbIterationsBatch)
        sb.Append(Me.ShowParameters())

        sb.AppendLine("Neuron count(" & 0 & ")=" & Me.nbInputNeurons)
        For i = 0 To Me.network.Layers.Count - 1
            sb.AppendLine("Neuron count(" & i + 1 & ")=" & Me.neuronCount(i + 1))
        Next

        ReDimWeights()
        GetWeights()

        sb.AppendLine("")

        For i = 1 To Me.layerCount - 1

            sb.AppendLine("W(" & i & ")={")

            'Dim layer = Me.network.Layers(i)
            ' Not available (but available inside Visual Studio Debug mode!)
            'Dim w = layer.Weights
            'Dim b = layer.Biases

            Dim nbNeuronsLayer = Me.neuronCount(i) ' Me.network.GetLayerNeuronCount(i)
            Dim nbNeuronsPreviousLayer = Me.neuronCount(i - 1) ' Me.network.GetLayerNeuronCount(i - 1)

            Dim l% = 0
            For j = 0 To nbNeuronsLayer - 1
                sb.Append(" {")

                Dim nbWeights = nbNeuronsPreviousLayer
                For k = 0 To nbWeights - 1
                    Dim weight = Me.m_weights(i - 1)(l)
                    Dim sVal$ = weight.ToString(format2Dec).ReplaceCommaByDot()
                    sb.Append(sVal)
                    If Me.useBias OrElse k < nbWeights - 1 Then sb.Append(", ")
                    l += 1
                Next k

                If Me.useBias Then
                    Dim weightT = Me.m_biases(i - 1)(j)
                    Dim sValT$ = weightT.ToString(format2Dec).ReplaceCommaByDot()
                    sb.Append(sValT)
                End If

                sb.Append("}")
                If j < nbNeuronsLayer - 1 Then sb.Append("," & vbLf)
            Next j
            sb.Append("}" & vbLf)

            If i < Me.layerCount - 1 Then sb.AppendLine()

        Next i

        Return sb.ToString()

    End Function

#Else

    Public Overrides Function ShowWeights$()

        Dim sb As New StringBuilder
        sb.Append(Me.ShowParameters())

        sb.AppendLine("Neuron count(" & 0 & ")=" & Me.nbInputNeurons)
        For i = 0 To Me.network.Layers.Count - 1
            sb.AppendLine("Neuron count(" & i + 1 & ")=" & Me.neuronCount(i + 1))
        Next

        sb.AppendLine("")
        sb.AppendLine("GetWeights, SetWeights, see there:")
        sb.AppendLine("https://github.com/PatriceDargenton/NeuralNetwork.NET/tree/stable")

        Return sb.ToString()

    End Function

#End If

End Class