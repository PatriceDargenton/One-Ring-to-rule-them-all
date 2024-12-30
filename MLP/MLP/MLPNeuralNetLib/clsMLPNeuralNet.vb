
' https://www.nuget.org/packages/NeuralNetwork.NET Nuget install
' https://github.com/Sergio0694/NeuralNetwork.NET
' https://scisharp.github.io/SciSharp Other .NET Machine Learning projects

' Install-Package NeuralNetwork.NET -Version 2.1.3
' Fix SixLabors.ImageSharp FileLoadException (0x80131040) bug:
' Install-Package SixLabors.ImageSharp -Version 1.0.0-beta0007
' (bug starting from 1.0.0-rc0001 version)

' To implement NeuralNetwork.NET set the conditional compilation constant "NeuralNetworkNETEngine" 
'  in the Directory.Build.props file at the solution level:
' <DefineConstants>NeuralNetworkNETEngine</DefineConstants>
' Not there:
'#Const NeuralNetworkNETEngine = 0 ' 0: Off, 1: On
' This engine has been disabled, because it uses a vulnerable version of the SixLabors.ImageSharp library,
'  which cannot be updated:
' https://github.com/advisories/GHSA-65x7-c272-7g7r Severity: high
' https://github.com/advisories/GHSA-63p8-c4ww-9cg7 Severity: high

#If NeuralNetworkNETEngine Then

Imports NeuralNetworkNET.APIs
Imports NeuralNetworkNET.APIs.Enums
Imports NeuralNetworkNET.APIs.Structs
Imports NeuralNetworkNET.Networks.Cost ' CostFunctionType

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

    Public Overrides Function GetMLPType$()
        Return System.Reflection.MethodBase.GetCurrentMethod().DeclaringType.Name
    End Function

    Public Overrides Function GetActivationFunctionType() As enumActivationFunctionType
        Return enumActivationFunctionType.Library
    End Function

    Public Overrides Sub InitializeStruct(neuronCount%(), addBiasColumn As Boolean)

        Me.useBias = addBiasColumn
        If Not Me.useBias Then
            Throw New NotImplementedException(
                "useBias=False is not implemented for clsMLPNeuralNet!")
        End If

        MyBase.InitializeStruct(neuronCount, addBiasColumn)
        Me.trainingAlgorithm = enumTrainingAlgorithm.RMSProp
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
        MyBase.SetActivationFunction(actFnc, gain, center)

        Select Case actFnc
            Case enumActivationFunction.Sigmoid
            Case enumActivationFunction.HyperbolicTangent
#If GetWeightsImplementation Or GetWeightsImplementationVS2013 Then
            Case enumActivationFunction.Mish
#End If
            Case Else
                Throw New NotImplementedException(
                    "This activation function is not available!")
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
#If GetWeightsImplementation Or GetWeightsImplementationVS2013 Then
            Case enumActivationFunction.Mish : actFunc = ActivationType.Mish 
#End If
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
        Dim nbWeights = nbNeuronsLayer * Me.neuronCount(i)
        ReDim Me.m_weights(i)(0 To nbWeights - 1)
        ReDim Me.m_biases(i)(0 To nbNeuronsLayer - 1)
        For j = 0 To nbNeuronsLayer - 1
            Me.m_biases(i)(j) = 0
        Next j
        Dim nbNeuronsPreviousLayer = Me.neuronCount(i)
        Dim l = 0
        For j = 0 To nbNeuronsLayer - 1
            For k = 0 To nbNeuronsPreviousLayer - 1
                Dim r = weights(j, k)
                Me.m_weights(i)(l) = CSng(r)
                l += 1
            Next k
            If Me.useBias Then
                Dim r = weights(j, nbNeuronsPreviousLayer)
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
        ReDim Me.m_weights(0 To Me.layerCount - 2)
        ReDim Me.m_biases(0 To Me.layerCount - 2)
        For i = 0 To Me.layerCount - 2
            Dim nbNeuronsLayer = Me.neuronCount(i + 1)
            Dim nbWeights = nbNeuronsLayer * Me.neuronCount(i)
            ReDim Me.m_weights(i)(0 To nbWeights - 1)
            ReDim Me.m_biases(i)(0 To nbNeuronsLayer - 1)
        Next
    End Sub

    Public Overrides Sub RoundWeights()

        ReDimWeights()
        GetWeights()

        For i = 0 To Me.network.Layers.Count - 1

            Dim nbNeuronsLayer = Me.neuronCount(i + 1)
            Dim nbWeights = nbNeuronsLayer * Me.neuronCount(i)

            For k = 0 To nbWeights - 1
                Dim weight = Me.m_weights(i)(k)
                Dim rounded = Math.Round(weight, clsMLPGeneric.nbRoundingDigits)
                Me.m_weights(i)(k) = CSng(rounded)
            Next k

            If Me.useBias Then
                For k = 0 To nbNeuronsLayer - 1
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

        If Me.nbIterationsBatch <> Me.m_nbIterationsBatchLast Then
#If GetWeightsImplementationVS2013 Then
            Me.m_dataset = DatasetLoader.Training2(Me.inputJaggedArray, Me.targetJaggedArray, size:=1)
#End If
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
            Case enumTrainingAlgorithm.AdaMax
                NetworkManager.TrainNetworkAsync(Me.network, dataset,
                    TrainingAlgorithms.AdaMax(), epochs:=Me.nbIterationsBatch).Wait()
            Case enumTrainingAlgorithm.AdaGrad
                NetworkManager.TrainNetworkAsync(Me.network, dataset,
                    TrainingAlgorithms.AdaGrad(), epochs:=Me.nbIterationsBatch).Wait()
            Case enumTrainingAlgorithm.Adam
                NetworkManager.TrainNetworkAsync(Me.network, dataset,
                    TrainingAlgorithms.Adam(), epochs:=Me.nbIterationsBatch).Wait()
            Case enumTrainingAlgorithm.Momentum
                NetworkManager.TrainNetworkAsync(Me.network, dataset,
                    TrainingAlgorithms.Momentum(), epochs:=Me.nbIterationsBatch).Wait()
            Case enumTrainingAlgorithm.RMSProp
                NetworkManager.TrainNetworkAsync(Me.network, dataset,
                    TrainingAlgorithms.RMSProp(), epochs:=Me.nbIterationsBatch).Wait()
            Case enumTrainingAlgorithm.AdaDelta
                NetworkManager.TrainNetworkAsync(Me.network, dataset,
                    TrainingAlgorithms.AdaDelta(), epochs:=Me.nbIterationsBatch).Wait()
            Case enumTrainingAlgorithm.StochasticGradientDescent
                NetworkManager.TrainNetworkAsync(Me.network, dataset,
                    TrainingAlgorithms.StochasticGradientDescent(), epochs:=Me.nbIterationsBatch).Wait()
            Case Else
                'Throw New NotImplementedException("This training algorithm is not available!")
                ' Default training algorithm: RMSProp
                NetworkManager.TrainNetworkAsync(Me.network, dataset,
                    TrainingAlgorithms.RMSProp(), epochs:=Me.nbIterationsBatch).Wait()
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

    Public Overrides Function ShowWeights$(Optional format$ = format2Dec)

        ReDimWeights()
        GetWeights()

        Dim sb As New StringBuilder
        If Me.learningMode = enumLearningMode.VectorialBatch Then _
            sb.AppendLine("nb iterations batch=" & Me.nbIterationsBatch)
        Dim weightsBase = MyBase.ShowWeights(format)
        sb.Append(weightsBase)
        Dim weights = sb.ToString
        Return weights

    End Function

    Public Overrides Function GetWeight#(layer%, neuron%, weight%)
        Dim ws! = Me.GetWeightSingle(layer, neuron, weight)
        Dim wd# = ws
        Return wd
    End Function

    Public Overrides Function GetWeightSingle!(layer%, neuron%, weight%)

        Dim nbNeuronsLayer = Me.neuronCount(layer - 1)
        If weight >= nbNeuronsLayer Then
            Dim l2% = weight - nbNeuronsLayer + neuron
            Dim biasValue = Me.m_biases(layer - 1)(l2)
            Return biasValue
        End If
        Dim l% = neuron * nbNeuronsLayer + weight
        Dim weightValue = Me.m_weights(layer - 1)(l)
        Return weightValue

    End Function

    Public Overrides Sub SetWeight(layer%, neuron%, weight%, weightWalue#)
        Dim ws! = CSng(weightWalue)
        SetWeightSingle(layer, neuron, weight, ws)
    End Sub

    Public Overrides Sub SetWeightSingle(layer%, neuron%, weight%, weightWalue!)

        Dim nbNeuronsLayer = Me.neuronCount(layer - 1)
        If weight >= nbNeuronsLayer Then
            Dim l2% = weight - nbNeuronsLayer + neuron
            Me.m_biases(layer - 1)(l2) = weightWalue
            Exit Sub
        End If
        Dim l% = neuron * nbNeuronsLayer + weight
        Me.m_weights(layer - 1)(l) = weightWalue

    End Sub

#Else

    Public Overrides Function ShowWeights$(Optional format$ = format2Dec)

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
    
#End If