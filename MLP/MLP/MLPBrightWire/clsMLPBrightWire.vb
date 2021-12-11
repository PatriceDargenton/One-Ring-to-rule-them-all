
#If NET4 Then

' From https://github.com/jdermody/brightwire-v2 .Net 4.6

#ElseIf NETCORE Then

' From https://github.com/jdermody/brightwire    .Net core (.Net 5 and .Net 6)

Imports BrightData
Imports BrightData.Numerics
Imports BrightWire.Models

#End If

Imports System.Text
Imports BrightWire ' CreateGraphFactory
Imports BrightWire.ExecutionGraph ' GraphFactory

Public Class clsMLPBrightWire : Inherits clsVectorizedMLPGeneric

    Const defaultBatchSize = 10

    Dim m_testData, m_trainingData As IDataSource
    Dim m_errorMetric As IErrorMetric
    Dim m_graph As ExecutionGraph.GraphFactory

    Private m_weights!()()
    Private m_biases!()()

#If NET4 Then
    Dim m_dataTableTest, m_dataTableTraining, m_dataTable As IDataTable
    Dim m_executionContext As IExecutionContext
    Dim m_bestGraph As Models.ExecutionGraph
    Dim m_engine As IGraphTrainingEngine
    Dim m_output As IReadOnlyList(Of Models.ExecutionResult)
#ElseIf NETCORE Then
    Dim m_context As BrightDataContext
    Dim m_training, m_test As IRowOrientedDataTable
    Dim m_model, m_bestGraph As ExecutionGraphModel
    Dim m_executionEngine As IGraphExecutionEngine
    Dim m_graphTrainingEngine As IGraphTrainingEngine
    Dim m_output As List(Of ExecutionResult)
#End If

    Public Overrides Function GetMLPType$()
        Return System.Reflection.MethodBase.GetCurrentMethod().DeclaringType.Name
    End Function

    Public Overrides Function GetActivationFunctionType() As enumActivationFunctionType
        Return enumActivationFunctionType.LibraryOptimized
    End Function

    Public Overrides Sub InitializeStruct(neuronCount%(), addBiasColumn As Boolean)

        Me.useBias = addBiasColumn
        If Not Me.useBias Then
            Throw New NotImplementedException(
                "useBias=False is not implemented for clsMLPBrightWire!")
        End If

        MyBase.InitializeStruct(neuronCount, addBiasColumn)
        Me.trainingAlgorithm = enumTrainingAlgorithm.RMSProp
        Me.weightAdjustment = 0
        Me.minBatchSize = defaultBatchSize
        Me.nbIterationsBatch = defaultBatchSize

        ' Useful for testing other MLP tests not using learning rate
        If Me.learningRate = 0 Then Me.learningRate = 0.1

        If IsNothing(Me.inputArray) Then Exit Sub
        If IsNothing(Me.m_actFunc) Then Exit Sub
        If Me.m_actFunc = enumActivationFunction.Undefined Then Exit Sub

    End Sub

    Public Overrides Sub SetActivationFunction(
            actFnc As enumActivationFunction, Optional gain! = 1, Optional center! = 0)

        center = 0
        If actFnc = enumActivationFunction.Sigmoid Then gain = 1
        If actFnc = enumActivationFunction.HyperbolicTangent Then gain = 2
        MyBase.SetActivationFunction(actFnc, gain, center)

        BuildGraph()

    End Sub

#If NET4 Then

    Private Sub BuildSamples(inputArray0(,) As Single, targetArray0(,) As Single)

        Dim builder = BrightWireProvider.CreateDataTableBuilder()
        builder.AddVectorColumn(Me.nbInputNeurons, "Input")
        builder.AddVectorColumn(Me.nbOutputNeurons, "Output", isTarget:=True)
        Me.nbSamples = Me.targetArray.GetUpperBound(0) + 1
        For j = 0 To Me.nbSamples - 1
            Dim fvi As New Models.FloatVector
            fvi.Data = clsMLPHelper.GetVector(inputArray0, j)
            Dim fvo As New Models.FloatVector
            fvo.Data = clsMLPHelper.GetVector(targetArray0, j)
            builder.Add(fvi, fvo)
        Next
        m_dataTable = builder.Build()
        Dim data0 = m_graph.CreateDataSource(m_dataTable)

        m_engine.Test(data0, m_errorMetric)

        Dim networkGraph = m_engine.Graph
        Dim executionEngine = m_graph.CreateEngine(networkGraph)

        m_output = executionEngine.Execute(data0)
        Me.averageError = m_output.Average(Function(o) o.CalculateError(m_errorMetric))

    End Sub

    Private Sub BuildGraph()

        Dim lap = BrightWireProvider.CreateLinearAlgebra

        Dim builder = BrightWireProvider.CreateDataTableBuilder()
        builder.AddVectorColumn(Me.nbInputNeurons, "Input")
        builder.AddVectorColumn(Me.nbOutputNeurons, "Output", isTarget:=True)
        Me.nbSamples = Me.targetArray.GetUpperBound(0) + 1
        For j = 0 To Me.nbSamples - 1
            Dim fvi As New Models.FloatVector
            fvi.Data = clsMLPHelper.GetVector(Me.inputArray, j)
            Dim fvo As New Models.FloatVector
            fvo.Data = clsMLPHelper.GetVector(Me.targetArray, j)
            builder.Add(fvi, fvo)
        Next
        m_dataTableTraining = builder.Build()

        m_dataTableTest = Nothing
        If Not IsNothing(Me.targetArrayTest) Then
            Dim nbTestSamples = Me.targetArrayTest.GetUpperBound(0) + 1
            For j = 0 To nbTestSamples - 1
                Dim fvi As New Models.FloatVector
                fvi.Data = clsMLPHelper.GetVector(Me.inputArrayTest, j)
                Dim fvo As New Models.FloatVector
                fvo.Data = clsMLPHelper.GetVector(Me.targetArrayTest, j)
                builder.Add(fvi, fvo)
            Next
            m_dataTableTest = builder.Build()
        End If

        ' Create the graph
        m_graph = New GraphFactory(lap)
        m_errorMetric = m_graph.ErrorMetric.Quadratic

        ' BinaryClassification: Binary classification rounds outputs to 1 or 0
        '  and compares them against the target
        ' OneHotEncoding: Finds the single index of the highest activation
        '  and compares it to the target index
        ' Quadratic: Quadratic error
        ' CrossEntropy : Cross entropy error https://en.wikipedia.org/wiki/Cross_entropy

        Select Case Me.trainingAlgorithm
            Case enumTrainingAlgorithm.NesterovMomentum
                m_graph.CurrentPropertySet.Use(m_graph.GradientDescent.NesterovMomentum)
            Case enumTrainingAlgorithm.AdaGrad
                m_graph.CurrentPropertySet.Use(m_graph.GradientDescent.AdaGrad)
            Case enumTrainingAlgorithm.Adam
                m_graph.CurrentPropertySet.Use(m_graph.GradientDescent.Adam)
            Case enumTrainingAlgorithm.Momentum
                m_graph.CurrentPropertySet.Use(m_graph.GradientDescent.Momentum)
            Case enumTrainingAlgorithm.RMSProp
                m_graph.CurrentPropertySet.Use(m_graph.GradientDescent.RmsProp)
            Case Else
                'Throw New NotImplementedException("This training algorithm is not available!")
                ' Default training algorithm: RMSProp
                m_graph.CurrentPropertySet.Use(m_graph.GradientDescent.RmsProp)
        End Select

        m_graph.CurrentPropertySet.Use(m_graph.WeightInitialisation.Gaussian)

        ' Create the engine
        m_trainingData = m_graph.CreateDataSource(m_dataTableTraining)
        If Not IsNothing(Me.targetArrayTest) Then
            m_testData = m_graph.CreateDataSource(m_dataTableTest)
        End If

        Dim batchSize = defaultBatchSize
        If Me.learningMode = enumLearningMode.VectorialBatch Then batchSize = Me.nbIterationsBatch
        m_engine = m_graph.CreateTrainingEngine(m_trainingData,
            learningRate:=Me.learningRate, batchSize:=batchSize)

        ' Create the network

        With m_graph.Connect(m_engine)

            Dim actFctOut As INode = Nothing ' Must be distinct from actFct
            For l = 1 To Me.layerCount - 2
                Dim actFct As INode
                Select Case m_actFunc
                    Case enumActivationFunction.Sigmoid
                        actFct = m_graph.SigmoidActivation
                        actFctOut = m_graph.SigmoidActivation
                    Case enumActivationFunction.HyperbolicTangent
                        actFct = m_graph.TanhActivation
                        actFctOut = m_graph.TanhActivation
                    Case enumActivationFunction.ReLu
                        actFct = m_graph.ReluActivation
                        actFctOut = m_graph.ReluActivation
                    Case Else
                        Throw New NotImplementedException(
                            "This activation function is not available!")
                End Select
                ' Create a feed forward layer with the activation function
                .AddFeedForward(neuronCount(l)).Add(actFct)
            Next

            ' Create a second feed forward layer with the activation function
            Dim outputSize = m_engine.DataSource.OutputSize
            .AddFeedForward(outputSize).Add(actFctOut)

            ' Calculate the error and backpropagate the error signal
            .AddBackpropagation(m_errorMetric)
        End With

        ' Train the network
        m_executionContext = m_graph.CreateExecutionContext

    End Sub

#ElseIf NETCORE Then

    Private Sub BuildSamples(inputArray0(,) As Single, targetArray0(,) As Single)

        Dim builder = m_context.BuildTable
        builder = m_context.BuildTable
        builder.AddColumn(BrightDataType.Vector, "Input")
        builder.AddColumn(BrightDataType.Vector, "Output").SetTarget(True)
        Dim nbTestSamples = targetArray0.GetUpperBound(0) + 1
        For j = 0 To nbTestSamples - 1
            Dim v_in0 = clsMLPHelper.GetVector(inputArray0, j)
            Dim v_out0 = clsMLPHelper.GetVector(targetArray0, j)
            Dim v_in = m_context.CreateVector(v_in0)
            Dim v_out = m_context.CreateVector(v_out0)
            builder.AddRow(v_in, v_out)
        Next
        m_test = builder.BuildRowOriented
        m_testData = m_graph.CreateDataSource(m_test)

        If Not (m_bestGraph Is Nothing) Then
            ' Create a new network to execute the learned network
            Dim executionEngine = m_graph.CreateExecutionEngine(m_bestGraph)
            m_output = executionEngine.Execute(m_testData).ToList
        End If

    End Sub

    Private Sub BuildGraph()

        m_context = New BrightDataContext(randomSeed:=Nothing)
        m_context.ResetRandom(seed:=Nothing)
        m_context.UseNumericsLinearAlgebra()
        m_context.UserNotifications = Nothing

        m_graph = m_context.CreateGraphFactory

        Dim hiddenLayerSize% = Me.neuronCount(1)

        Dim builder = m_context.BuildTable
        Me.nbSamples = Me.targetArray.GetUpperBound(0) + 1
        builder.AddColumn(BrightDataType.Vector, "Input")
        builder.AddColumn(BrightDataType.Vector, "Output").SetTarget(True)
        For j = 0 To Me.nbSamples - 1
            Dim v_in0 = clsMLPHelper.GetVector(Me.inputArray, j)
            Dim v_out0 = clsMLPHelper.GetVector(Me.targetArray, j)
            Dim v_in = m_context.CreateVector(v_in0)
            Dim v_out = m_context.CreateVector(v_out0)
            builder.AddRow(v_in, v_out)
        Next
        m_training = builder.BuildRowOriented

        m_test = Nothing
        If Not IsNothing(Me.targetArrayTest) Then
            builder = m_context.BuildTable
            builder.AddColumn(BrightDataType.Vector, "Input")
            builder.AddColumn(BrightDataType.Vector, "Output").SetTarget(True)
            Dim nbTestSamples = Me.targetArrayTest.GetUpperBound(0) + 1
            For j = 0 To nbTestSamples - 1
                Dim v_in0 = clsMLPHelper.GetVector(Me.inputArrayTest, j)
                Dim v_out0 = clsMLPHelper.GetVector(Me.targetArrayTest, j)
                Dim v_in = m_context.CreateVector(v_in0)
                Dim v_out = m_context.CreateVector(v_out0)
                builder.AddRow(v_in, v_out)
            Next
            m_test = builder.BuildRowOriented
        End If

        Dim batchSize = defaultBatchSize
        If Me.learningMode = enumLearningMode.VectorialBatch Then batchSize = Me.nbIterationsBatch

        ' BinaryClassification: Binary classification rounds outputs to 1 or 0
        '  and compares them against the target
        ' OneHotEncoding: Finds the single index of the highest activation
        '  and compares it to the target index
        ' Quadratic: Quadratic error
        ' CrossEntropy : Cross entropy error https://en.wikipedia.org/wiki/Cross_entropy

        m_errorMetric = m_graph.ErrorMetric.Quadratic

        ' Create the property set
        m_graph.CurrentPropertySet.
            Use(m_graph.GaussianWeightInitialisation(
                zeroBias:=True, stdDev:=0.1F,
                varianceCalibration:=GaussianVarianceCalibration.SquareRoot2N))

        Select Case Me.trainingAlgorithm
            Case enumTrainingAlgorithm.NesterovMomentum
                m_graph.CurrentPropertySet.Use(m_graph.GradientDescent.NesterovMomentum)
            Case enumTrainingAlgorithm.AdaGrad
                m_graph.CurrentPropertySet.Use(m_graph.GradientDescent.AdaGrad)
            Case enumTrainingAlgorithm.Adam
                m_graph.CurrentPropertySet.Use(m_graph.GradientDescent.Adam)
            Case enumTrainingAlgorithm.Momentum
                m_graph.CurrentPropertySet.Use(m_graph.GradientDescent.Momentum)
            Case enumTrainingAlgorithm.RMSProp
                m_graph.CurrentPropertySet.Use(m_graph.GradientDescent.RmsProp)
            Case Else
                'Throw New NotImplementedException("This training algorithm is not available!")
                ' Default training algorithm: RMSProp
                m_graph.CurrentPropertySet.Use(m_graph.GradientDescent.RmsProp)
        End Select

        m_trainingData = m_graph.CreateDataSource(m_training)
        m_testData = Nothing
        If Not IsNothing(Me.targetArrayTest) Then m_testData = m_graph.CreateDataSource(m_test)

        m_graphTrainingEngine = m_graph.CreateTrainingEngine(m_trainingData, m_errorMetric,
            learningRate:=Me.learningRate, batchSize:=CType(Me.nbIterationsBatch, UInt32))

        With m_graph.Connect(m_graphTrainingEngine)
            Dim actFct As Node.NodeBase = Nothing
            Dim actFctOut As Node.NodeBase = Nothing
            For l = 1 To Me.layerCount - 2
                Select Case m_actFunc
                    Case enumActivationFunction.Sigmoid
                        actFct = m_graph.SigmoidActivation
                        actFctOut = m_graph.SigmoidActivation
                    Case enumActivationFunction.HyperbolicTangent
                        actFct = m_graph.TanhActivation
                        actFctOut = m_graph.TanhActivation
                    Case enumActivationFunction.ReLu
                        actFct = m_graph.ReluActivation
                        actFctOut = m_graph.ReluActivation
                    Case Else
                        Throw New NotImplementedException(
                            "This activation function is not available!")
                End Select
                ' Create a feed forward layer with the activation function
                Dim nc = CType(neuronCount(l), UInteger)
                .AddFeedForward(nc).Add(actFct)
            Next
            ' Create a second feed forward layer with the activation function
            Dim outputSize = CType(m_trainingData.OutputSize, UInteger)
            .AddFeedForward(outputSize).Add(actFctOut)
            .AddBackpropagation()
        End With

        m_bestGraph = Nothing

    End Sub

    Public Overrides Sub TrainVector()

        TrainVectorBatch()

    End Sub

    Public Overrides Sub TrainVectorBatch()

        TrainVectorBatch(Me.nbIterationsBatch)

        ComputeAverageError()

        If Me.printOutput_ Then PrintOutput(Me.nbIterations - 1, force:=True)

    End Sub

    Public Overrides Sub TrainVectorBatch(nbIterationsBatch%)

        ' Default implementation: call TrainVectorOneIteration()
        Me.learningMode = enumLearningMode.VectorialBatch
        Me.vectorizedLearningMode = True

        ' Works only with testCadence = 1, but this is very slow
        'For iteration = 0 To nbIterationsBatch - 1
        '    TrainVectorOneIteration()
        'Next

        ShowMessage("Training...")
        Dim nbIter = CType(Me.nbIterations, UInteger)
        Dim testCadence = Me.nbIterations
        m_graphTrainingEngine.Train(
            numIterations:=nbIter,
            testData:=m_trainingData,
            onImprovement:=
                Function(model)
                    m_bestGraph = model.Graph
                    Return m_bestGraph
                End Function,
            testCadence:=testCadence)

        If m_bestGraph Is Nothing Then m_bestGraph = m_graphTrainingEngine.Graph
        ShowMessage("Training: Done.")

        SetOuput1D()

    End Sub

    Public Overrides Sub TrainVectorOneIteration()

        Throw New NotImplementedException()

        ' Too slow:
        'm_graphTrainingEngine.Train(numIterations:=1, testData:=m_trainingData,
        '    onImprovement:=
        '        Function(model)
        '            m_bestGraph = model.Graph
        '            Return m_bestGraph
        '        End Function,
        '    testCadence:=1)

    End Sub

    Public Overrides Sub SetOuput1D()

        If m_trainingData Is Nothing Then Exit Sub

        Dim nbTargets = Me.targetArray.GetLength(0)
        Dim lengthTot = nbTargets * Me.nbOutputNeurons
        Dim outputs!(lengthTot - 1)
        Dim outputs1D#(lengthTot - 1)
        Dim outputsDbl#(lengthTot - 1)
        Dim outputs2D#(nbTargets - 1, Me.nbOutputNeurons - 1)

        If m_bestGraph Is Nothing Then m_bestGraph = m_graphTrainingEngine.Graph
        m_executionEngine = m_graph.CreateExecutionEngine(m_bestGraph)
        Dim output = m_executionEngine.Execute(m_trainingData).ToList

        Dim i% = 0
        For Each item In output
            For Each index In item.MiniBatchSequence.MiniBatch.Rows
                Dim indexInt = CType(index, Integer)
                Dim result = item.Output(indexInt)
                For j = 0 To Me.nbOutputNeurons - 1
                    Dim rOutput! = result(j)
                    outputs(i * Me.nbOutputNeurons + j) = rOutput
                    outputs1D(i * Me.nbOutputNeurons + j) = rOutput
                    outputsDbl(i * Me.nbOutputNeurons + j) = rOutput
                    outputs2D(i, j) = rOutput
                Next
                i += 1
            Next
        Next

        Me.lastOutputArray1DSingle = clsMLPHelper.Convert1DArrayOfDoubleToSingle(outputs1D)
        Me.output = outputs2D

    End Sub

    Public Overrides Sub TrainOneSample(input() As Single, target() As Single)
        Throw New NotImplementedException()
    End Sub

    Public Overrides Sub TestAllSamples(inputs!(,), nbOutputs%)

        BuildSamples(inputs, Me.targetArray)

        InitializeTraining()
        Me.nbSamples = inputs.GetLength(0)
        Dim nbInputs = inputs.GetLength(1)
        Dim outputs!(0 To Me.nbSamples - 1, 0 To nbOutputs - 1)
        For i = 0 To Me.nbSamples - 1
            Me.numSample = i
            TestOneSampleByIndex(i)
            Dim output!() = Me.lastOutputArray1DSingle
            For j = 0 To output.GetLength(0) - 1
                outputs(i, j) = output(j)
            Next
        Next
        Me.output = outputs
        ComputeAverageError()

    End Sub

    Public Sub TestOneSampleByIndex(index%)

        Dim nbTargets = 1
        Dim lengthTot = nbTargets * Me.nbOutputNeurons
        Dim outputs!(lengthTot - 1)
        Dim outputsDbl#(lengthTot - 1)

        Dim i% = 0
        For Each item In m_output
            Dim result = item.Output(index)
            For j% = 0 To Me.nbOutputNeurons - 1
                Dim rOutput! = result(j)
                outputs(i * Me.nbOutputNeurons + j) = rOutput
                outputsDbl(i * Me.nbOutputNeurons + j) = rOutput
            Next
        Next

        Me.lastOutputArray1DSingle = outputs
        Dim outputs2D#(0, Me.nbOutputNeurons - 1)
        clsMLPHelper.Fill2DArrayOfDouble(outputs2D, outputsDbl, 0)
        Me.output = outputs2D

    End Sub

    Public Overrides Sub TestOneSample(input() As Single)

        If m_testData Is Nothing Then Exit Sub

        Dim output = m_executionEngine.Execute(m_testData).ToList

        Dim nbTargets = 1
        Dim lengthTot = nbTargets * Me.nbOutputNeurons
        Dim outputs!(lengthTot - 1)
        Dim outputsDbl#(lengthTot - 1)

        Dim samplesFound% = 0
        Dim i% = 0
        For Each item In output
            For Each index In item.MiniBatchSequence.MiniBatch.Rows

                Dim indexInt = CType(index, Integer)
                Dim result = item.Output(indexInt)

                Dim row = m_test.Row(index)
                Dim fv = CType(row.Item(0), LinearAlgebra.Vector(Of Single))
                Dim sampleFound = True
                For k = 0 To input.Length - 1
                    Dim xi! = CType(fv(k), Single)
                    If xi <> input(k) Then sampleFound = False : Exit For
                Next
                If Not sampleFound Then Continue For

                samplesFound += 1
                For j% = 0 To Me.nbOutputNeurons - 1
                    Dim rOutput! = result(j)
                    outputs(i * Me.nbOutputNeurons + j) = rOutput
                    outputsDbl(i * Me.nbOutputNeurons + j) = rOutput
                Next

            Next
        Next

        If samplesFound <> 1 Then Stop

        Me.lastOutputArray1DSingle = outputs
        Dim outputs2D#(0, Me.nbOutputNeurons - 1)
        clsMLPHelper.Fill2DArrayOfDouble(outputs2D, outputsDbl, 0)
        Me.output = outputs2D

    End Sub

    Public Sub GetWeights()

        m_model = m_bestGraph
        If IsNothing(m_model) Then m_model = m_graphTrainingEngine.Graph

        Dim layer% = Me.layerCount - 2
        For Each node0 In m_model.OtherNodes
            If node0.TypeName.Contains("FeedForward") Then
                Dim node1 = m_graphTrainingEngine.Start.FindById(node0.Id)
                node1.LoadParameters(m_graph, node0)
                Dim ff = CType(node1, IFeedForward)
                Dim l% = 0
                Dim nbRows% = CType(ff.Weight.RowCount, Integer)
                Dim nbCols% = CType(ff.Weight.ColumnCount, Integer)
                Dim nbNeuronsPreviousLayer = Me.neuronCount(layer)
                Dim nbNeuronsLayer = Me.neuronCount(layer + 1)
                If nbRows <> nbNeuronsPreviousLayer Then Stop
                If nbCols <> nbNeuronsLayer Then Stop
                For j = 0 To nbRows - 1
                    Dim fm = ff.Weight.Data
                    Dim fv = fm.Row(CType(j, UInteger))
                    For k% = 0 To nbCols - 1
                        Me.m_weights(layer)(l + k) = fv.Values(k)
                    Next
                    l += CType(ff.Weight.ColumnCount, Integer)
                Next
                nbRows = 1
                For j = 0 To nbRows - 1
                    Dim fv = ff.Bias.Data
                    For k% = 0 To CType(fv.Size, Integer) - 1
                        Me.m_biases(layer)(k) = fv.Values(k)
                    Next
                Next
                layer -= 1
            End If
        Next

    End Sub

    Public Sub SetWeights()

        m_model = m_bestGraph
        If IsNothing(m_model) Then m_model = m_graphTrainingEngine.Graph

        Dim layer% = Me.layerCount - 2
        For Each node0 In m_model.OtherNodes
            If node0.TypeName.Contains("FeedForward") Then
                Dim node1 = m_graphTrainingEngine.Start.FindById(node0.Id)
                node1.LoadParameters(m_graph, node0)
                Dim ff = CType(node1, IFeedForward)
                Dim nbRowsUI = ff.Weight.RowCount
                Dim nbColsUI = ff.Weight.ColumnCount
                Dim nbRows% = CType(nbRowsUI, Integer)
                Dim nbCols% = CType(nbColsUI, Integer)
                Dim nbNeuronsPreviousLayer = Me.neuronCount(layer)
                Dim nbNeuronsLayer = Me.neuronCount(layer + 1)
                Dim nbBiases = nbNeuronsLayer
                Dim nbWeights = nbNeuronsPreviousLayer
                If nbRows <> nbNeuronsPreviousLayer Then Stop
                If nbCols <> nbNeuronsLayer Then Stop
                Dim m As Utility.Matrix = clsMLPHelper.TransformArrayTo2DArray(Me.m_weights(layer), nbRows, nbCols)
                Dim v2 = m_context.CreateMatrix(Of Single)(nbRowsUI, nbColsUI,
                    Function(m1, m2) m.ItemUIntSng(m1, m2))
                ff.Weight.Data = v2
                Dim mb As Utility.Matrix = clsMLPHelper.TransformArrayTo2DArray(Me.m_biases(layer), 1, nbCols)
                Dim vb = m_context.CreateVector(Of Single)(nbColsUI, Function(m1) mb.ItemUIntSng(0, m1))
                ff.Bias.Data = vb
                layer -= 1
            End If
        Next

    End Sub

#End If

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
            SetWeights()
        End If

    End Sub

    Public Overrides Sub Randomize(Optional minValue! = -0.5, Optional maxValue! = 0.5)

        ' Re-build the graphe to randomize again the network!
        BuildGraph()

        ' Round the weights (to reproduce all tests exactly)
        RoundWeights()

    End Sub

    Public Sub ReDimWeights()

        ReDim Me.m_weights(0 To Me.layerCount - 2)
        ReDim Me.m_biases(0 To Me.layerCount - 2)
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

        For i = 0 To Me.layerCount - 2

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

    Public Overrides Function GetWeight!(layer%, neuron%, weight%)

        Dim nbNeuronsPreviousLayer = Me.neuronCount(layer - 1)
        If weight >= nbNeuronsPreviousLayer Then
            Dim l2% = weight - nbNeuronsPreviousLayer + neuron
            Dim biasSng = Me.m_biases(layer - 1)(l2)
            Return biasSng
        End If
        Dim l% = neuron * nbNeuronsPreviousLayer + weight
        Dim weightSng = Me.m_weights(layer - 1)(l)
        Return weightSng

    End Function

#If NET4 Then

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

        m_engine.Train(m_executionContext)

    End Sub

    Public Overrides Sub SetOuput1D()

        ' Create a new network to execute the learned network
        Dim networkGraph = m_engine.Graph
        Dim executionEngine = m_graph.CreateEngine(networkGraph)
        Dim output = executionEngine.Execute(m_trainingData)

        Dim nbInputs = Me.inputArray.GetLength(0)
        Dim nbTargets = Me.targetArray.GetLength(0)
        Dim lengthTot = nbTargets * Me.nbOutputNeurons
        Dim outputs!(lengthTot - 1)
        Dim outputs1D#(lengthTot - 1)
        Dim outputsDbl#(lengthTot - 1)
        Dim outputs2D#(nbTargets - 1, Me.nbOutputNeurons - 1)

        Dim i% = 0
        For Each item In output
            For Each index In item.MiniBatchSequence.MiniBatch.Rows
                Dim result = item.Output(index)
                For j = 0 To Me.nbOutputNeurons - 1
                    Dim rOutput! = result.Data(j)
                    outputs(i * Me.nbOutputNeurons + j) = rOutput
                    outputs1D(i * Me.nbOutputNeurons + j) = rOutput
                    outputsDbl(i * Me.nbOutputNeurons + j) = rOutput
                    outputs2D(i, j) = rOutput
                Next
                i += 1
            Next
        Next

        Me.lastOutputArray1DSingle = clsMLPHelper.Convert1DArrayOfDoubleToSingle(outputs1D)
        Me.output = outputs2D

    End Sub

    Public Overrides Sub TrainOneSample(input() As Single, target() As Single)

        m_engine.Train(m_executionContext)

    End Sub

    Public Overrides Sub TestOneSampleAndComputeError(input!(), target!())

        TestOneSampleStatic(Me.numSample)

        Dim targetArray2D!(0, target.GetUpperBound(0))
        clsMLPHelper.Fill2DArrayOfSingle(targetArray2D, target, 0)
        ComputeAverageErrorOneSample(targetArray2D)

    End Sub

    Public Sub TestOneSampleAndComputeErrorTest(input!(), target!())

        TestOneSampleStaticTest(Me.numSample)

        Dim targetArray2D!(0, target.GetUpperBound(0))
        clsMLPHelper.Fill2DArrayOfSingle(targetArray2D, target, 0)
        ComputeAverageErrorOneSample(targetArray2D)

    End Sub

    Private Sub TestOneSampleStatic(index0%)

        Static networkGraph As Models.ExecutionGraph
        Static executionEngine As IGraphEngine
        Static output As IReadOnlyList(Of Models.ExecutionResult)
        If Me.numSample = 0 Then
            m_engine.Test(m_trainingData, m_errorMetric)
            ' Create a new network to execute the learned network
            networkGraph = m_engine.Graph
            executionEngine = m_graph.CreateEngine(networkGraph)
            output = executionEngine.Execute(m_trainingData)
            Me.averageError = output.Average(Function(o) o.CalculateError(m_errorMetric))
        End If

        Dim nbInputs = 1
        Dim nbTargets = 1
        Dim lengthTot = nbTargets * Me.nbOutputNeurons
        Dim outputs!(lengthTot - 1)
        Dim outputsDbl#(lengthTot - 1)

        Dim i% = 0, j% = 0

        For Each item In output

            Dim size = item.Output.Count
            If index0 >= size Then
                Stop
                Exit For
            End If
            Dim result = item.Output(index0)

            Const verify = False
            If verify Then
                ' Seek the right sample into the graph
                Dim row = m_dataTableTraining.GetRow(index0)
                Dim fvobj = row.Data(0)
                Dim fv As Models.FloatVector = CType(fvobj, Models.FloatVector)
                Dim sampleFound = True
                For k = 0 To Me.nbInputNeurons - 1
                    Dim xi! = CType(fv.Data(k), Single)
                    If xi <> Me.inputArray(index0, k) Then sampleFound = False : Exit For
                Next
                If Not sampleFound Then Stop
            End If

            For j = 0 To Me.nbOutputNeurons - 1
                Dim rOutput! = result.Data(j)
                outputs(i * Me.nbOutputNeurons + j) = rOutput
                outputsDbl(i * Me.nbOutputNeurons + j) = rOutput
            Next

        Next

        Me.lastOutputArray1DSingle = outputs
        Dim outputs2D#(0, Me.nbOutputNeurons - 1)
        clsMLPHelper.Fill2DArrayOfDouble(outputs2D, outputsDbl, 0)
        Me.output = outputs2D

    End Sub

    Private Sub TestOneSampleStaticTest(index0%)

        Static networkGraph As Models.ExecutionGraph
        Static executionEngine As IGraphEngine
        Static output As IReadOnlyList(Of Models.ExecutionResult)
        If Me.numSample = 0 Then
            m_engine.Test(m_testData, m_errorMetric)
            ' Create a new network to execute the learned network
            networkGraph = m_engine.Graph
            executionEngine = m_graph.CreateEngine(networkGraph)
            output = executionEngine.Execute(m_testData)
            Me.averageError = output.Average(Function(o) o.CalculateError(m_errorMetric))
        End If

        Dim nbInputs = 1
        Dim nbTargets = 1
        Dim lengthTot = nbTargets * Me.nbOutputNeurons
        Dim outputs!(lengthTot - 1)
        Dim outputsDbl#(lengthTot - 1)

        Dim i% = 0, j% = 0

        For Each item In output

            Dim size = item.Output.Count
            If index0 >= size Then
                Stop
                Exit For
            End If
            Dim result = item.Output(index0)

            Const verify = False
            If verify Then
                ' Seek the right sample into the graph
                Dim row = m_dataTableTest.GetRow(index0)
                Dim fvobj = row.Data(0)
                Dim fv As Models.FloatVector = CType(fvobj, Models.FloatVector)
                Dim sampleFound = True
                For k = 0 To Me.nbInputNeurons - 1
                    Dim xi! = CType(fv.Data(k), Single)
                    If xi <> Me.inputArrayTest(index0, k) Then sampleFound = False : Exit For
                Next
                If Not sampleFound Then Stop
            End If

            For j = 0 To Me.nbOutputNeurons - 1
                Dim rOutput! = result.Data(j)
                outputs(i * Me.nbOutputNeurons + j) = rOutput
                outputsDbl(i * Me.nbOutputNeurons + j) = rOutput
            Next

        Next

        Me.lastOutputArray1DSingle = outputs
        Dim outputs2D#(0, Me.nbOutputNeurons - 1)
        clsMLPHelper.Fill2DArrayOfDouble(outputs2D, outputsDbl, 0)
        Me.output = outputs2D

    End Sub

    Public Overrides Sub TrainSystematic(inputs!(,), targets!(,),
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        BuildSamples(inputs, targets)
        MyBase.TrainSystematic(inputs, targets, learningMode)

    End Sub

    Public Overrides Sub TestAllSamples(inputs!(,), nbOutputs%)

        BuildSamples(inputs, Me.targetArray)

        InitializeTraining()
        Me.nbSamples = inputs.GetLength(0)
        Dim nbInputs = inputs.GetLength(1)
        Dim outputs!(0 To Me.nbSamples - 1, 0 To nbOutputs - 1)
        For i = 0 To Me.nbSamples - 1
            Me.numSample = i
            TestOneSampleByIndex(i)
            Dim output!() = Me.lastOutputArray1DSingle
            For j = 0 To output.GetLength(0) - 1
                outputs(i, j) = output(j)
            Next
        Next
        Me.output = outputs
        ComputeAverageError()

    End Sub

    Public Overrides Sub TestOneSample(input() As Single)

        Dim nbTargets = 1
        Dim lengthTot = nbTargets * Me.nbOutputNeurons
        Dim outputs!(lengthTot - 1)
        Dim outputsDbl#(lengthTot - 1)

        Dim i% = 0, j% = 0

        Dim samplesFound% = 0
        For Each item In m_output
            For Each index In item.MiniBatchSequence.MiniBatch.Rows

                Dim indexMax = item.Output.Count
                If index >= indexMax Then Exit For
                Dim result = item.Output(index)

                ' Seek the right sample into the graph
                Dim row = m_dataTable.GetRow(index)
                Dim fvobj = row.Data(0)
                Dim fv As Models.FloatVector = CType(fvobj, Models.FloatVector)
                Dim sampleFound = True
                For k = 0 To input.Length - 1
                    Dim xi! = CType(fv.Data(k), Single)
                    If xi <> input(k) Then sampleFound = False : Exit For
                Next
                If Not sampleFound Then Continue For

                samplesFound += 1
                For j = 0 To Me.nbOutputNeurons - 1
                    Dim rOutput! = result.Data(j)
                    outputs(i * Me.nbOutputNeurons + j) = rOutput
                    outputsDbl(i * Me.nbOutputNeurons + j) = rOutput
                Next
                Exit For

            Next
            If samplesFound > 0 Then Exit For
        Next

        If samplesFound <> 1 Then Stop

        Me.lastOutputArray1DSingle = outputs
        Dim outputs2D#(0, Me.nbOutputNeurons - 1)
        clsMLPHelper.Fill2DArrayOfDouble(outputs2D, outputsDbl, 0)
        Me.output = outputs2D

    End Sub

    Public Sub TestOneSampleByIndex(index%)

        Dim nbTargets = 1
        Dim lengthTot = nbTargets * Me.nbOutputNeurons
        Dim outputs!(lengthTot - 1)
        Dim outputsDbl#(lengthTot - 1)

        Dim i% = 0
        For Each item In m_output
            Dim result = item.Output(index)
            For j% = 0 To Me.nbOutputNeurons - 1
                Dim rOutput! = result.Data(j)
                outputs(i * Me.nbOutputNeurons + j) = rOutput
                outputsDbl(i * Me.nbOutputNeurons + j) = rOutput
            Next
        Next

        Me.lastOutputArray1DSingle = outputs
        Dim outputs2D#(0, Me.nbOutputNeurons - 1)
        clsMLPHelper.Fill2DArrayOfDouble(outputs2D, outputsDbl, 0)
        Me.output = outputs2D

    End Sub

    Public Sub GetWeights()

        m_bestGraph = m_engine.Graph
        If m_bestGraph Is Nothing Then Exit Sub
        If m_bestGraph.OtherNodes Is Nothing Then Exit Sub
        Dim layer% = Me.layerCount - 2
        For Each node0 In m_bestGraph.OtherNodes

            If node0.TypeName.Contains("FeedForward") Then
                Dim node1 = m_engine.Start.FindById(node0.Id)
                node1.LoadParameters(node0)
                Dim ff = CType(node1, IFeedForward)
                Dim nbRows = ff.Weight.RowCount
                Dim nbCols = ff.Weight.ColumnCount
                Dim nbNeuronsPreviousLayer = Me.neuronCount(layer)
                Dim nbNeuronsLayer = Me.neuronCount(layer + 1)
                If nbRows <> nbNeuronsPreviousLayer Then Stop
                If nbCols <> nbNeuronsLayer Then Stop
                Dim l = 0
                For j = 0 To nbRows - 1
                    Dim fm = ff.Weight.Data
                    Dim fv = fm.Row(j)
                    For k = 0 To nbCols - 1
                        Me.m_weights(layer)(l + k) = fv.Data(k)
                    Next
                    l += ff.Weight.ColumnCount
                Next
                nbRows = 1
                For j = 0 To nbRows - 1
                    Dim fv = ff.Bias.Data
                    For k = 0 To fv.Count - 1
                        Me.m_biases(layer)(k) = fv.Data(k)
                    Next
                Next
                layer -= 1
            End If
        Next

    End Sub

    Public Sub SetWeights()

        If m_engine.Graph Is Nothing Then Exit Sub
        If m_engine.Graph.OtherNodes Is Nothing Then Exit Sub
        Dim layer% = Me.layerCount - 2
        For Each node0 In m_engine.Graph.OtherNodes
            If node0.TypeName.Contains("FeedForward") Then
                Dim node1 = m_engine.Start.FindById(node0.Id)
                node1.LoadParameters(node0)
                Dim ff = CType(node1, IFeedForward)
                Dim nbRows = ff.Weight.RowCount
                Dim nbCols = ff.Weight.ColumnCount
                If layer < 0 Then layer = Me.layerCount - 2 : Stop ' ???
                Dim nbNeuronsPreviousLayer = Me.neuronCount(layer)
                Dim nbNeuronsLayer = Me.neuronCount(layer + 1)
                If nbRows <> nbNeuronsPreviousLayer Then Stop
                If nbCols <> nbNeuronsLayer Then Stop
                Dim fm = ff.Weight.Data
                Dim l = 0
                For j = 0 To nbRows - 1
                    Dim fv = fm.Row(j)
                    For k = 0 To nbCols - 1
                        Dim w = Me.m_weights(layer)(l + k)
                        fv.Data(k) = w
                    Next
                    fm.Row(j) = fv
                    l += ff.Weight.ColumnCount
                Next
                nbRows = 1
                Dim fvb = ff.Bias.Data
                For j = 0 To nbRows - 1
                    For k = 0 To fvb.Count - 1
                        Dim b = Me.m_biases(layer)(k)
                        fvb.Data(k) = b
                    Next
                Next
                ff.Weight.Data = fm
                ff.Bias.Data = fvb
                layer -= 1
            End If
        Next
        m_engine = m_graph.CreateTrainingEngine(m_trainingData, m_engine.Graph,
            Me.learningRate, Me.nbIterationsBatch)

    End Sub

#End If

End Class
