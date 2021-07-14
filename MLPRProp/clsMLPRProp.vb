
' From https://github.com/nokitakaze/ResilientBackProp : C# -> VB .NET conversion

Imports System.IO
Imports System.Text ' StringBuilder
Imports System.Threading ' Interlocked
Imports System.Threading.Tasks ' TaskFactory

''' <summary>
''' Resilient Back Propagation (RPROP)
''' </summary>
Friend Class clsMLPRProp : Inherits clsVectorizedMLPGeneric

    Public multiThread As Boolean = False
    Public inputJaggedDblArray#()()
    Public targetJaggedDblArray#()()
    Dim m_trainData#()()
    Dim m_nbWeights%
    Dim m_gnn As NeuralNetwork

    Private m_weights#()

    Public Overrides Function GetMLPType$()
        Return System.Reflection.MethodBase.GetCurrentMethod().DeclaringType.Name
    End Function

    Public Overrides Function GetActivationFunctionType() As enumActivationFunctionType
        Return enumActivationFunctionType.SpecificCodeOptimized
    End Function

    Public Overrides Sub InitializeStruct(neuronCount%(), addBiasColumn As Boolean)

        MyBase.InitializeStruct(neuronCount, addBiasColumn)
        Me.trainingAlgorithm = enumTrainingAlgorithm.RProp

        ' Randomize weights between [0 - 1] instead of [-0.5 - 0.5] ?
        Me.useNguyenWidrowWeightsInitialization = False

        Me.learningRate = 0
        Me.weightAdjustment = 0
        Me.useBias = addBiasColumn
        Me.nbIterationsBatch = 10

        If Not Me.useBias Then
            Throw New NotImplementedException(
                "useBias=False is not implemented for clsMLPRProp!")
        End If

        If IsNothing(Me.inputArray) Then Exit Sub
        Dim inputArrayDbl = clsMLPHelper.Convert2DArrayOfSingleToDouble(Me.inputArray)
        Me.inputJaggedDblArray = clsMLPHelper.Transform2DArrayToJaggedArray(inputArrayDbl)
        Dim targetArrayDbl = clsMLPHelper.Convert2DArrayOfSingleToDouble(Me.targetArray)
        Me.targetJaggedDblArray = clsMLPHelper.Transform2DArrayToJaggedArray(targetArrayDbl)
        Me.exampleCount = Me.inputArray.GetLength(0)
        SetTrainData()

        m_gnn = New NeuralNetwork(neuronCount)
        m_nbWeights = m_gnn.GetWeightsCount()
        m_gnn.multiThread = multiThread
        m_gnn.softMaxForLastLayer = Me.classificationObjective

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

            Case enumActivationFunction.ELU
                SetActivationFunctionOptimized(
                    enumActivationFunctionOptimized.ELU, gain, center)

            Case Else
                Throw New NotImplementedException(
                    "This activation function is not available!")
        End Select

    End Sub

    Public Overrides Sub SetActivationFunctionOptimized(
        actFnc As enumActivationFunctionOptimized, Optional gain! = 1, Optional center! = 0)

        MyBase.SetActivationFunctionOptimized(actFnc, gain, center)

        m_gnn.LambdaFnc = Me.lambdaFnc
        m_gnn.LambdaFncDFOF = Me.lambdaFncDFOF
        Select Case actFnc
            Case enumActivationFunctionOptimized.Sigmoid
                Me.m_actFunc = enumActivationFunction.Sigmoid
            Case enumActivationFunctionOptimized.HyperbolicTangent
                Me.m_actFunc = enumActivationFunction.HyperbolicTangent
            Case enumActivationFunctionOptimized.ELU
                Me.m_actFunc = enumActivationFunction.ELU
            Case Else
                Throw New NotImplementedException(
                    "This activation function is not available!")
                Me.activFnc = Nothing
                m_gnn.LambdaFnc = Nothing
                m_gnn.LambdaFncDFOF = Nothing
        End Select

    End Sub

    Public Overrides Sub Randomize(Optional minValue! = -0.5!, Optional maxValue! = 0.5!)

        If Me.useNguyenWidrowWeightsInitialization Then

            InitializeWeightsNguyenWidrow()

        Else

            Dim rnd As Random = New Random()

            Dim weights = New Double(Me.m_nbWeights - 1) {} ' actually weights & biases

            For i = 0 To Me.m_nbWeights - 1
                'Dim r# = 20.0 * rnd.NextDouble() - 10.0
                'Dim r# = rnd.NextDouble(minValue, maxValue)
                Dim r# = rnd.NextDoubleGreaterThanZero(minValue, maxValue, clsMLPGeneric.minRandomValue)
                Dim rounded# = Math.Round(r, clsMLPGeneric.nbRoundingDigits)
                weights(i) = rounded
            Next

            Me.m_gnn.SetWeights(weights)

        End If

    End Sub

    Private Sub InitializeWeightsNguyenWidrow()

        Const SmallStep! = 0.001
        Const VerySmallStep! = clsMLPGeneric.minRandomValue ' 0.0001

        Me.rnd = New Random()
        Dim weights = New Double(Me.m_nbWeights - 1) {} ' actually weights & biases

        Dim numWeight% = 0
        For layer = 1 To Me.layerCount - 1

            Dim size = Me.neuronCount(layer)
            Dim prev_size = Me.neuronCount(layer - 1)
            Dim layerNumWeights = (Me.neuronCount(layer - 1) + 1) * Me.neuronCount(layer)

            For node = 0 To size - 1

                Dim biasIndice = numWeight + prev_size ' Me.Neurons(layer).Biases(node)
                Dim r# = Me.rnd.NextDouble()
                'r = 1 ' Show the weights distribution
                weights(biasIndice) = (SmallStep - VerySmallStep) * r + VerySmallStep

                Dim vj# = 0
                For i = 0 To prev_size - 1
                    Dim wIndice = numWeight + i ' Me.Neurons(layer).Weights(node)(i)
                    r = Me.rnd.NextDouble()
                    'r = 1 ' Show the weights distribution
                    weights(wIndice) = (SmallStep - VerySmallStep) * r + VerySmallStep
                    Dim x = weights(wIndice)
                    vj += x * x
                Next

                ' Nguyen-Widrow (1990) algorithm
                ' https://www.rdocumentation.org/packages/brnn/versions/0.8/topics/initnw
                ' p : Number of predictors
                ' n : Number of cases
                ' Scaling factor : teta = 0.7 * p ^ (1/n)
                ' (see the distribution examples at the end of this file)
                vj = 0.7 * Math.Pow(size, 1.0 / prev_size) / Math.Sqrt(vj)
                For i = 0 To prev_size - 1
                    Dim wIndice = numWeight + i ' Me.Neurons(layer).Weights(node)(i)
                    weights(wIndice) *= vj
                Next

                numWeight += prev_size + 1

            Next node

        Next layer

        For i = 0 To Me.m_nbWeights - 1
            Dim r# = weights(i)
            Dim rounded# = Math.Round(r, 4) ' clsMLPGeneric.nbRoundingDigits = 2
            weights(i) = rounded
        Next

        m_gnn.SetWeights(weights)

    End Sub

    Public Overrides Sub InitializeWeights(layer%, weights#(,))

        Static s_weights#()
        Static l% = 0
        If layer = 1 Then
            l = 0
            s_weights = New Double(Me.m_nbWeights - 1) {} ' actually weights & biases
        End If

        Dim nbNeuronsLayer = Me.neuronCount(layer)
        Dim nbNeuronsPreviousLayer = Me.neuronCount(layer - 1)

        ' Bias weigths are included within main weights in this array: weights#(,)
        If Me.useBias Then nbNeuronsPreviousLayer += 1

        For j = 0 To nbNeuronsLayer - 1
            For k = 0 To nbNeuronsPreviousLayer - 1
                s_weights(l) = weights(j, k)
                l += 1
            Next k
        Next j

        If layer = Me.layerCount - 1 Then Me.m_gnn.SetWeights(s_weights)

    End Sub

    Public Sub SetTrainData()

        Dim numInput = Me.nbInputNeurons
        Dim numOutput = Me.nbOutputNeurons
        Dim numRows = Me.exampleCount

        Dim result = New Double(numRows - 1)() {} ' allocate return-result matrix
        For i = 0 To numRows - 1
            result(i) = New Double(numInput + numOutput - 1) {} ' 1-of-N Y in last column
        Next

        Dim numTrainRows% = Me.exampleCount
        m_trainData = New Double(numTrainRows - 1)() {}

        For r = 0 To numRows - 1

            Dim inputs#() = Me.inputJaggedDblArray(r)

            Dim c% = 0 ' column into result[][]
            For i = 0 To numInput - 1
                result(r)(c) = inputs(i)
                c += 1
            Next

            For i = 0 To numOutput - 1
                result(r)(c) = Me.targetJaggedDblArray(r)(i)
                c += 1
            Next

            m_trainData(r) = result(r)

        Next r ' each row

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

        Dim maxEpochs% = 1
        Dim finalErr# = 0
        m_gnn.TrainRPROP(m_trainData, maxEpochs, finalErr)
        Me.averageError = finalErr

    End Sub

    Public Overrides Sub TrainVectorBatch(nbIterationsBatch%)

        Me.learningMode = enumLearningMode.VectorialBatch
        Me.vectorizedLearningMode = True

        Dim maxEpochs% = nbIterationsBatch
        Dim finalErr# = 0
        m_gnn.TrainRPROP(m_trainData, maxEpochs, finalErr)
        Me.averageError = finalErr

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
            outputs = m_gnn.ComputeOutputs(Me.inputJaggedDblArray(i))
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

        If learningMode = enumLearningMode.Vectorial Then
            TrainVector() ' Does not work fine
        Else
            TrainVectorBatch() ' This is the main learning mode for this MLP
        End If

    End Sub

    Public Overrides Sub TrainStochastic(inputs!(,), targets!(,))
        ' TrainStochastic requires TrainOneSample
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

        Dim inputDble = clsMLPHelper.Convert1DArrayOfSingleToDouble(input)
        Dim outputs = m_gnn.ComputeOutputs(inputDble)
        Me.lastOutputArray1DSingle = clsMLPHelper.Convert1DArrayOfDoubleToSingle(outputs)

        Dim outputs2D#(0, Me.nbOutputNeurons - 1)
        clsMLPHelper.Fill2DArrayOfDouble(outputs2D, outputs, 0)
        Me.output = outputs2D

    End Sub

    Public Overrides Function ShowWeights$(Optional format$ = format2Dec)

        Me.m_weights = Me.m_gnn.GetWeights()

        Dim sb As New StringBuilder
        If Me.learningMode = enumLearningMode.VectorialBatch Then _
            sb.AppendLine("nb iterations batch=" & Me.nbIterationsBatch)
        If Me.useNguyenWidrowWeightsInitialization Then format = format4Dec
        Dim weightsBase = MyBase.ShowWeights(format)
        sb.Append(weightsBase)
        Dim weights = sb.ToString
        Return weights

    End Function

    Public Overrides Function GetWeight!(layer%, neuron%, weight%)

        If IsNothing(Me.m_weights) Then Return 0.0!
        Dim l% = weight
        For i% = 1 To layer
            Dim nbNeuronsLayer = Me.neuronCount(i)
            Dim nbNeuronsPreviousLayer = Me.neuronCount(i - 1)
            If Me.useBias Then nbNeuronsPreviousLayer += 1
            Dim mult = neuron
            If i < layer Then mult = nbNeuronsLayer
            l += nbNeuronsPreviousLayer * mult
        Next
        Dim weightDbl = Me.m_weights(l)
        Dim weightSng = CSng(weightDbl)
        Return weightSng

    End Function

#Region "Console demo"

    Const lineLenGlob% = 10

    Public Sub ConsoleDemo(ByRef trainAcc#, ByRef testAcc#, multiThread As Boolean,
            Optional trainAndTest As Boolean = False,
            Optional fastMode As Boolean = False)

        Console.WriteLine(vbLf &
            "Begin neural network with Resilient Back-Propagation (RPROP) training demo")

        Const numInput% = 4 ' number features
        Const numHidden% = 5
        Const numOutput% = 3 ' number of classes for Y
        Const numRows% = 5000 '10000

        Console.WriteLine(vbLf &
            "Generating " & numRows & " artificial data items with " & numInput & " features")
        Dim allData = MakeAllData(numInput, numHidden, numOutput, numRows)
        Console.WriteLine("Done")

        Console.WriteLine(vbLf & "Creating train (80%) and test (20%) matrices")
        Dim trainData As Double()() = Nothing
        Dim testData As Double()() = Nothing
        MakeTrainTest(allData, 0.8, trainData, testData)
        Console.WriteLine("Done")

        Console.WriteLine(vbLf & "Training data: " & vbLf)
        ShowData(trainData, 4, 2, True)

        Console.WriteLine("Test data: " & vbLf)
        ShowData(testData, 3, 2, True)

        Console.WriteLine("Creating a 4-5-3 neural network")
        Dim sizes%() = {numInput, numHidden, numOutput}
        Dim nn As NeuralNetwork = New NeuralNetwork(sizes)
        If nn.saveWeights Then nn.Save("before_test.dat")

        Const maxEpochs% = 1000
        Console.WriteLine(vbLf & "Setting maxEpochs = " & maxEpochs)
        Console.WriteLine(vbLf & "Starting RPROP training")

        nn.multiThread = multiThread 'False
        nn.consoleDemo = True
        nn.debugConsoleDemo = False 'True
        nn.fastMode = fastMode
        nn.softMaxForLastLayer = True

        Dim weights#()

        If trainAndTest Then
            weights = nn.TrainRPROPAndTest(trainData, maxEpochs, testData) ' RPROP
        Else
            Dim finalErr# = 0
            nn.TrainRPROP(trainData, maxEpochs, finalErr) ' RPROP
            nn.TestRPROP(testData)
            weights = nn.GetWeights
        End If

        If nn.saveWeights Then nn.Save("after_test.dat")
        nn.ShowMessage("Done")

        nn.ShowMessage(vbLf & "Final neural network model weights:" & vbLf)
        ShowVector(weights, 4, lineLenGlob, True)
        trainAcc = nn.Accuracy(trainData, weights)
        nn.ShowMessage(vbLf & "Accuracy on training data = " & trainAcc.ToString("F4"))
        testAcc = nn.Accuracy(testData, weights)
        nn.ShowMessage(vbLf & "Accuracy on test data = " & testAcc.ToString("F4"))
        nn.ShowMessage(vbLf & "End neural network with Resilient Propagation demo" & vbLf)

    End Sub

    ''' <summary>
    ''' Generate synthetic data
    ''' </summary>
    Private Function MakeAllData(numInput%, numHidden%, numOutput%, numRows%) As Double()()

        Dim rnd As Random = New Random()
        Dim numWeights = numInput * numHidden + numHidden + numHidden * numOutput + numOutput

        Const initWeights = True ' Note: There is allready a Nguyen-Widrow weights initialization
        Dim weights#() = New Double(numWeights - 1) {} ' actually weights & biases

        If initWeights Then
            For i = 0 To numWeights - 1
                weights(i) = 20.0 * rnd.NextDouble() - 10.0 ' [-10.0 to -10.0]
            Next
            Console.WriteLine("Generating weights:")
            ShowVector(weights, 4, lineLenGlob, True)
        End If

        Dim result = New Double(numRows - 1)() {} ' allocate return-result matrix
        For i = 0 To numRows - 1
            result(i) = New Double(numInput + numOutput - 1) {} ' 1-of-N Y in last column
        Next

        Dim sizes%() = {numInput, numHidden, numOutput}
        Dim gnn As NeuralNetwork = New NeuralNetwork(sizes) ' generating NN
        If initWeights Then gnn.SetWeights(weights)

        For r = 0 To numRows - 1

            ' generate random inputs
            Dim inputs#() = New Double(numInput - 1) {}

            For i = 0 To numInput - 1
                inputs(i) = 20.0 * rnd.NextDouble() - 10.0 ' [-10.0 to -10.0]
            Next

            ' compute outputs
            Dim outputs#() = gnn.ComputeOutputs(inputs)
            ' translate outputs to 1-of-N
            Dim oneOfN#() = New Double(numOutput - 1) {} ' all 0.0
            Dim maxIndex% = 0
            Dim maxValue# = outputs(0)

            For i = 0 To numOutput - 1
                If Not outputs(i) > maxValue Then Continue For
                maxIndex = i
                maxValue = outputs(i)
            Next

            oneOfN(maxIndex) = 1.0

            ' place inputs and 1-of-N output values into curr row
            Dim c% = 0 ' column into result[][]
            For i = 0 To numInput - 1
                result(r)(c) = inputs(i)
                c += 1
            Next

            For i = 0 To numOutput - 1
                result(r)(c) = oneOfN(i)
                c += 1
            Next

        Next r ' each row

        Return result

    End Function ' MakeAllData

    ''' <summary>
    ''' Put synthetic data to train and test
    ''' </summary>
    Private Sub MakeTrainTest(allData#()(), trainPct#, ByRef trainData#()(), ByRef testData#()())

        Dim rnd As Random = New Random()
        Dim totRows = allData.Length
        Dim numTrainRows = CInt(totRows * trainPct) ' usually 0.80
        Dim numTestRows = totRows - numTrainRows
        trainData = New Double(numTrainRows - 1)() {}
        testData = New Double(numTestRows - 1)() {}
        Dim copy = New Double(allData.Length - 1)() {} ' ref copy of all data

        For i = 0 To copy.Length - 1
            copy(i) = allData(i)
        Next

        For i = 0 To copy.Length - 1
            Dim r% = rnd.Next(i, copy.Length) ' use Fisher-Yates
            Dim tmp#() = copy(r)
            copy(r) = copy(i)
            copy(i) = tmp
        Next i

        For i = 0 To numTrainRows - 1
            trainData(i) = copy(i)
        Next

        For i = 0 To numTestRows - 1
            testData(i) = copy(i + numTrainRows)
        Next

    End Sub ' MakeTrainTest

    Private Shared Sub ShowData(data#()(), numRows%, decimals%, indices As Boolean)

        Dim len = data.Length.ToString().Length

        ' First rows to display
        For i = 0 To numRows - 1
            If indices Then Console.Write("[" & i.ToString().PadLeft(len) & "]  ")

            If IsNothing(data(i)) Then Console.WriteLine("") : Continue For
            For j = 0 To data(i).Length - 1
                Dim v# = data(i)(j)
                If v >= 0.0 Then Console.Write(" ") ' '+'
                Console.Write(v.ToString("F" & decimals) & "    ")
            Next

            Console.WriteLine("")
        Next

        Dim lastRow% = data.Length - 1
        If numRows = lastRow + 1 Then Console.WriteLine(vbLf) : Exit Sub

        Console.WriteLine(". . .")

        If indices Then Console.Write("[" & lastRow.ToString().PadLeft(len) & "]  ")

        ' Display last row
        For j = 0 To data(lastRow).Length - 1
            Dim v# = data(lastRow)(j)
            If v >= 0.0 Then Console.Write(" ") ' '+'
            Console.Write(v.ToString("F" & decimals) & "    ")
        Next

        Console.WriteLine(vbLf)

    End Sub

    Private Shared Sub ShowVector(vector#(), decimals%, lineLen%, newLine As Boolean)

        For i = 0 To vector.Length - 1
            If i > 0 AndAlso i Mod lineLen = 0 Then Console.WriteLine("")
            If vector(i) >= 0 Then Console.Write(" ")
            Console.Write(vector(i).ToString("F" & decimals) & " ")
        Next

        If newLine Then Console.WriteLine("")

    End Sub

#End Region

#Region "Structures"

    Public Structure WeightComposite
        Public Weights#()()
        Public Biases#()
    End Structure

    Public Structure ThreadInputDatum
        Public trainDatum#()()
        Public allGradsAcc As WeightComposite()
        Public field#()()
        Public xValues#()
        Public tValues#()
        ''' <summary>
        ''' Sum
        ''' </summary>
        Public delim1#
        ''' <summary>
        ''' Average
        ''' </summary>
        Public delim2#
        Public sumSquaredErrors#()
    End Structure

    Public Structure RMSEThreadInputDatum
        Public trainDatum#()()
        Public xValues#()
        Public tValues#()
        Public field#()()
        ''' <summary>
        ''' Sum
        ''' </summary>
        Public delim1#
        ''' <summary>
        ''' Average
        ''' </summary>
        Public delim2#
        Public sumSquaredErrors#()
    End Structure

#End Region

    Public Class NeuralNetwork : Inherits AbstractNeuralNetwork

        Public Sub New(sizes As IReadOnlyList(Of Integer))
            MyBase.New(sizes)
        End Sub

        Protected Overrides Function ActivateFunction(
                rawValues As IReadOnlyList(Of Double), layerId%) As Double()

            If Me.softMaxForLastLayer AndAlso
               layerId >= Me.LayerCount - 1 Then Return Softmax(rawValues)

            Dim actfctDefined = Not IsNothing(Me.LambdaFnc)

            Dim values#() = New Double(rawValues.Count - 1) {}

            For i = 0 To rawValues.Count - 1

                If actfctDefined Then
                    Dim r# = rawValues(i)
                    Dim v1 = Me.LambdaFnc.Invoke(r)
                    values(i) = v1

                    'If debugActivationFunction Then
                    '    Dim v2 = HyperTan(rawValues(i))
                    '    If Not clsMLPHelper.Compare(v1, v2, dec:=5) Then Stop
                    'End If

                    Continue For
                End If

                values(i) = HyperTan(rawValues(i))

            Next

            Return values

        End Function

        Protected Overrides Function CalculateGradTerms(rawValues#()(),
                tValues As IReadOnlyList(Of Double)) As Double()()

            Dim gradTerms = New Double(rawValues.Length - 1)() {}
            For layerId = Me.LayerCount - 1 To 0 + 1 Step -1
                gradTerms(layerId) = If(layerId < Me.LayerCount - 1,
                    CalculateGradTermsForNonLast(
                        rawValues(layerId), Me.Neurons(layerId + 1).Weights,
                        gradTerms(layerId + 1), Me.LambdaFncDFOF),
                    CalculateGradTermsForLast(rawValues(layerId), tValues,
                        Me.LambdaFncDFOF, Me.softMaxForLastLayer))
            Next
            Return gradTerms

        End Function

        Private Shared Function CalculateGradTermsForLast(rawValues As IReadOnlyList(Of Double),
                tValues As IReadOnlyList(Of Double),
                lambdaFncDFOF As Func(Of Double, Double),
                softMaxForLastLayer As Boolean) As Double()

            Dim actfctDefined = Not IsNothing(lambdaFncDFOF)

            Dim gradTerms = New Double(rawValues.Count - 1) {}
            For i = 0 To rawValues.Count - 1

                Dim value# = rawValues(i)
                Dim derivative#
                If softMaxForLastLayer OrElse Not actfctDefined Then
                    ' derivative of softmax = (1 - y) * y (same as log-sigmoid)
                    derivative = (1 - value) * value
                Else
                    derivative = lambdaFncDFOF.Invoke(value)
                End If

                ' careful with O-T vs. T-O, O-T is the most usual
                Dim delta# = value - tValues(i)
                gradTerms(i) = derivative * delta

            Next

            Return gradTerms

        End Function

        Private Shared Function CalculateGradTermsForNonLast(
                rawValues As IReadOnlyList(Of Double),
                nextNeuronLayerWeights As IReadOnlyList(Of Double()),
                nextGradTerms As IReadOnlyList(Of Double),
                lambdaFncDFOF As Func(Of Double, Double)) As Double()

            Dim actfctDefined = Not IsNothing(lambdaFncDFOF)

            Dim gradTerms = New Double(rawValues.Count - 1) {}
            Dim i% = 0
            For i = 0 To rawValues.Count - 1
                Dim value# = rawValues(i)

                Dim derivative#
                If actfctDefined Then
                    derivative = lambdaFncDFOF.Invoke(value)

                    'If debugActivationFunction Then
                    '    Dim d2 = (1 - value) * (1 + value)
                    '    If Not clsMLPHelper.Compare(derivative, d2, dec:=5) Then Stop
                    'End If

                Else
                    ' derivative of tanh = (1 - y) * (1 + y)
                    derivative = (1 - value) * (1 + value)
                End If

                ' double sum = nextGradTerms.Select((t, j) => t * nextNeuronLayerWeights[j][i]).Sum();
                Dim sum# = Enumerable.Select(nextGradTerms,
                    Function(t, j) t * nextNeuronLayerWeights(j)(i)).Sum()
                ' each hidden delta is the sum of this.sizes[2] terms
                gradTerms(i) = derivative * sum

            Next

            Return gradTerms

        End Function

        Protected Shared Function HyperTan#(x#)
            If x < -20.0 Then Return -1.0 ' approximation is correct to 30 decimals
            Return If(x > 20.0, 1.0, Math.Tanh(x))
        End Function

        Protected Shared Function Softmax(oSums As IReadOnlyList(Of Double)) As Double()

            ' does all output nodes at once so scale doesn't have to be re-computed each time
            ' determine max output-sum
            Dim max# = oSums(0)
            max = Enumerable.Concat(oSums, {max}).Max()

            ' determine scaling factor -- sum of exp(each val - max)
            Dim scale# = oSums.Sum(Function(t) Math.Exp(t - max))
            Dim result = New Double(oSums.Count - 1) {}

            For i = 0 To oSums.Count - 1
                result(i) = Math.Exp(oSums(i) - max) / scale
            Next

            Return result ' now scaled so that xi sum to 1.0

        End Function

    End Class

    Public MustInherit Class AbstractNeuralNetwork

        ''' <summary>
        ''' Lambda function for the activation function
        ''' </summary>
        Public LambdaFnc As Func(Of Double, Double)

        ''' <summary>
        ''' Lambda function for the derivative of the activation function,
        ''' from the original function
        ''' </summary>
        Public LambdaFncDFOF As Func(Of Double, Double)

        Public consoleDemo As Boolean = False

        Public debugConsoleDemo As Boolean = False

        ''' <summary>
        ''' softmax is an activation function allowing to estimate a probability at each
        '''  output (1 among N) in the objective of classification within a homogeneous group
        ''' </summary>
        Public softMaxForLastLayer As Boolean = False

        Public saveWeights As Boolean = False

        Public fastMode As Boolean = True

        Public multiThread As Boolean

        Const displayMod% = 100 ' 10
        Const showProgress = False
        Const displayMod2% = 5
        Const sumIndice% = 0
        Const averageIndice% = 1

        Protected ReadOnly Rnd As Random

        Protected LayerCount%

        Protected Sizes%()

        ''' <summary>
        ''' Values for layers
        ''' </summary>
        Protected Layers#()()
        Protected Neurons As WeightComposite()

        Const SmallErr! = 0.001
        Const VerySmallErr! = 0.0001

        Const SmallValue# = 0.01

        ''' <summary>
        ''' Variable etaPlus is the factor used to increase a weight or bias delta when
        '''  the algorithm is moving in the correct direction.
        ''' </summary>
        Const EtaPlus# = 1.2 ' values are from the paper

        ''' <summary>
        ''' Variable etaMinus is the factor used to decrease a weight or bias delta when
        '''  the algorithm has overshot a minimum error.
        ''' </summary>
        Const EtaMinus# = 0.5

        ''' <summary>
        ''' Variables deltaMax and deltaMin are used to prevent any weight or bias increase
        '''  or decrease factor from being too large or too small.
        ''' </summary>
        Const DeltaMax# = 50.0
        Const DeltaMin# = 0.000001

        Private ThreadCount%

        Protected Sub New(sizes As IReadOnlyList(Of Integer))

            Me.LayerCount = sizes.Count
            Me.Sizes = New Integer(sizes.Count - 1) {}

            For i = 0 To sizes.Count - 1
                Me.Sizes(i) = sizes(i)
            Next

            Me.Layers = New Double(Me.LayerCount - 1)() {}
            Me.Neurons = New WeightComposite(Me.LayerCount - 1) {}

            For i = 0 To Me.LayerCount - 1
                Me.Layers(i) = New Double(Me.Sizes(i) - 1) {}
            Next

            For i = 1 To Me.LayerCount - 1
                Me.Neurons(i).Biases = New Double(Me.Sizes(i) - 1) {}
                Me.Neurons(i).Weights = MakeMatrix(Me.Sizes(i), Me.Sizes(i - 1), 0.0)
            Next

            Me.Rnd = New Random()
            InitializeWeightsNguyenWidrow() ' all weights and biases

        End Sub

        Protected MustOverride Function ActivateFunction(
            rawValues As IReadOnlyList(Of Double), layerId%) As Double()

        Protected MustOverride Function CalculateGradTerms(
            rawValues#()(), tValues As IReadOnlyList(Of Double)) As Double()()

        ' helper for Train
        Protected Shared Function MakeMatrix(rows%, cols%, v#) As Double()()

            Dim result = New Double(rows - 1)() {}

            For r = 0 To result.Length - 1
                result(r) = New Double(cols - 1) {}
            Next

            For i = 0 To rows - 1
                For j = 0 To cols - 1
                    result(i)(j) = v
                Next
            Next

            Return result

        End Function

        Protected Shared Function MakeVector(len%, v#) As Double() ' helper for Train

            Dim result#() = New Double(len - 1) {}
            For i = 0 To len - 1
                result(i) = v
            Next
            Return result

        End Function

        Protected Sub InitializeWeightsNguyenWidrow()

            Const SmallStep! = 0.001
            Const VerySmallStep! = 0.0001

            For layer = 1 To Me.LayerCount - 1

                Dim size = Me.Sizes(layer)
                Dim prev_size = Me.Sizes(layer - 1)

                For node = 0 To size - 1

                    Me.Neurons(layer).Biases(node) =
                        (SmallStep - VerySmallStep) * Me.Rnd.NextDouble() + VerySmallStep
                    Me.Neurons(layer).Weights(node) = New Double(prev_size - 1) {}

                    Dim vj# = 0
                    For i = 0 To prev_size - 1
                        Me.Neurons(layer).Weights(node)(i) =
                            (SmallStep - VerySmallStep) * Me.Rnd.NextDouble() + VerySmallStep
                        'vj += Math.Pow(Me.Neurons(layer).Weights(node)(i), 2)
                        Dim x = Me.Neurons(layer).Weights(node)(i)
                        vj += x * x
                    Next

                    ' Nguyen-Widrow (1990) algorithm
                    ' https://www.rdocumentation.org/packages/brnn/versions/0.8/topics/initnw
                    ' p : Number of predictors
                    ' n : Number of cases
                    ' Scaling factor : teta = 0.7 * p ^ (1/n)
                    ' (see the distribution examples at the end of this file)
                    vj = 0.7 * Math.Pow(size, 1.0 / prev_size) / Math.Sqrt(vj)
                    For i = 0 To prev_size - 1
                        Me.Neurons(layer).Weights(node)(i) *= vj
                    Next

                Next
            Next

        End Sub

        Private Sub InitializeGradients(
            allGradsAcc As WeightComposite(),
            prevGradsAcc As WeightComposite(),
            prevDeltas As WeightComposite())

            For i = 1 To Me.LayerCount - 1
                Dim size% = Me.Sizes(i)
                Dim prevSize% = Me.Sizes(i - 1)

                ' accumulated over all training data
                allGradsAcc(i).Biases = New Double(size - 1) {}
                allGradsAcc(i).Weights = MakeMatrix(size, prevSize, 0.0)

                ' accumulated, previous iteration
                prevGradsAcc(i).Biases = New Double(size - 1) {}
                prevGradsAcc(i).Weights = MakeMatrix(size, prevSize, 0.0)

                ' must save previous weight deltas
                prevDeltas(i).Biases = MakeVector(size, SmallValue)
                prevDeltas(i).Weights = MakeMatrix(size, prevSize, SmallValue)
            Next

        End Sub

        Public Function TrainRPROPAndTest(trainData#()(), maxEpochs%, testData#()()) As Double()

            ' Acc: Accumulated
            Dim allGradsAcc As WeightComposite() = New WeightComposite(Me.LayerCount - 1) {}
            Dim prevGradsAcc As WeightComposite() = New WeightComposite(Me.LayerCount - 1) {}
            Dim prevDeltas As WeightComposite() = New WeightComposite(Me.LayerCount - 1) {}
            InitializeGradients(allGradsAcc, prevGradsAcc, prevDeltas)

            If Me.multiThread AndAlso Me.ThreadCount = 0 Then
                Me.ThreadCount = Environment.ProcessorCount - 1
                ' Round ThreadCount to a pair value to reproduce exactly the tests ?
                If Me.ThreadCount Mod 2 > 0 Then Me.ThreadCount -= 1
            End If

            Dim timer1 As Stopwatch = New Stopwatch()
            Dim timer2 As Stopwatch = New Stopwatch()
            Dim timer3 As Stopwatch = New Stopwatch()
            If Me.consoleDemo Then
                Dim currWts1#() = GetWeights()
                Dim err1#() = RootMeanSquaredError(trainData, currWts1)
                Dim err_t1#() = RootMeanSquaredError(testData, currWts1)
                Console.WriteLine(vbLf &
                    "epoch = pre; err = {0:F4} [{1:F4}]" & vbTab &
                    "test err = {2:F4} [{3:F4}]", err1(0), err1(1), err_t1(0), err_t1(1))
                timer3.Start()
            End If

            Dim epoch% = 0
            While epoch < maxEpochs

                epoch += 1

                If Me.consoleDemo Then
                    timer3.Stop()
                    timer1.Start()
                End If

                ' 1. compute and accumulate all gradients
                For layer = 1 To Me.LayerCount - 1
                    ' zero-out values from prev iteration
                    ZeroOut(allGradsAcc(layer).Weights)
                    ZeroOut(allGradsAcc(layer).Biases)
                Next

                ' 0: sumSquaredErrorItem, 1:sumSquaredError
                Dim err#() = ComputeGraduate(trainData, allGradsAcc)
                Dim finalErr = err(0)
                ' update all weights and biases (in any order)
                UpdateWeigtsAndBiases(allGradsAcc, prevGradsAcc, prevDeltas)

                If Me.consoleDemo Then
                    timer1.Stop()
                    timer3.Start()
                    If showProgress Then Console.Write(".")
                End If

                If epoch Mod displayMod = 0 OrElse finalErr <= VerySmallErr Then

                    If Me.consoleDemo Then
                        timer3.Stop()
                        timer2.Start()
                        Dim currWts#() = GetWeights()
                        Dim err_t#() = RootMeanSquaredError(testData, currWts)
                        Console.WriteLine(vbLf &
                            "epoch = {0} err = {1:F4} [{2:F4}]" & vbTab &
                            "test err = {3:F4} [{4:F4}]",
                            epoch, finalErr, err(1), err_t(0), err_t(1))
                        timer2.Stop()
                        timer3.Start()
                    End If

                    If Me.saveWeights Then Save("epoch-" & epoch & ".dat")

                    If Me.fastMode Then Exit While
                    If finalErr <= SmallErr Then Exit While

                Else

                    If Me.consoleDemo AndAlso showProgress AndAlso epoch Mod displayMod2 = 0 Then _
                        Console.Write(" ")

                End If

            End While

            If Me.consoleDemo Then
                timer3.Stop()
                Console.WriteLine("Elapsed time. Neuro = {0}, RMSE calculation = {1}, Other work = {2}",
                    timer1.ElapsedMilliseconds / 1000, timer2.ElapsedMilliseconds / 1000,
                    timer3.ElapsedMilliseconds / 1000)
            End If

            Dim wts#() = GetWeights()
            Return wts

        End Function

        Public Sub TrainRPROP(trainData#()(), maxEpochs%, ByRef finalErr#)

            finalErr = 0

            ' Acc: Accumulated
            Dim allGradsAcc As WeightComposite() = New WeightComposite(Me.LayerCount - 1) {}
            Dim prevGradsAcc As WeightComposite() = New WeightComposite(Me.LayerCount - 1) {}
            Dim prevDeltas As WeightComposite() = New WeightComposite(Me.LayerCount - 1) {}
            InitializeGradients(allGradsAcc, prevGradsAcc, prevDeltas)

            If Me.multiThread AndAlso Me.ThreadCount = 0 Then
                Me.ThreadCount = Environment.ProcessorCount - 1
                ' Round ThreadCount to a pair value to reproduce exactly the tests ?
                If Me.ThreadCount Mod 2 > 0 Then Me.ThreadCount -= 1
                'Me.ThreadCount = 1 ' multithread disabled
            End If

            If Me.consoleDemo Then
                Dim currWts1#() = GetWeights()
                Dim err1#() = RootMeanSquaredError(trainData, currWts1)
                Console.WriteLine(vbLf & "epoch = pre; err = {0:F4} [{1:F4}]", err1(0), err1(1))
            End If

            Dim timer1 As Stopwatch = New Stopwatch()

            Dim epoch% = 0
            While epoch < maxEpochs

                epoch += 1

                If Me.consoleDemo Then timer1.Start()

                ' 1. compute and accumulate all gradients
                For layer = 1 To Me.LayerCount - 1
                    ' zero-out values from prev iteration
                    ZeroOut(allGradsAcc(layer).Weights)
                    ZeroOut(allGradsAcc(layer).Biases)
                Next

                ' 0: sumSquaredErrorItem, 1:sumSquaredError
                Dim err#() = ComputeGraduate(trainData, allGradsAcc)
                finalErr = err(0)
                If Me.debugConsoleDemo Then
                    Me.ShowMessage("epoch " & epoch & " : err=" & finalErr.ToString("0.00000") &
                        ", multithread = " & Me.multiThread)
                End If
                ' update all weights and biases (in any order)
                UpdateWeigtsAndBiases(allGradsAcc, prevGradsAcc, prevDeltas)

                If Me.consoleDemo Then
                    timer1.Stop()
                    If showProgress Then Console.Write(".")
                End If

                If epoch Mod displayMod = 0 OrElse finalErr <= VerySmallErr Then

                    If Me.saveWeights Then Save("epoch-" & epoch & ".dat")

                    If Me.fastMode Then Exit While
                    If finalErr <= SmallErr Then Exit While

                Else

                    If Me.consoleDemo AndAlso showProgress AndAlso epoch Mod displayMod2 = 0 Then _
                        Console.Write(" ")

                End If

            End While

            If Me.consoleDemo Then Console.WriteLine("Elapsed time. Neuro = {0}",
                timer1.ElapsedMilliseconds / 1000)

        End Sub

        Public Sub TestRPROP(testData#()())

            Dim timer2 As Stopwatch = New Stopwatch()
            If Me.consoleDemo Then timer2.Start()

            Dim currWts#() = GetWeights()
            Dim err_t#() = RootMeanSquaredError(testData, currWts)

            If Me.consoleDemo Then
                Console.WriteLine(vbLf & "test err = {0:F4} [{1:F4}]", err_t(0), err_t(1))
                timer2.Stop()
            End If

        End Sub

        Protected Shared Sub ZeroOut(matrix#()())

            For Each t As Double() In matrix
                For i = 0 To t.Length - 1
                    t(i) = 0.0
                Next
            Next

        End Sub

        Protected Shared Sub ZeroOut(array#()) ' helper for Train

            For i = 0 To array.Length - 1
                array(i) = 0.0
            Next

        End Sub

        ''' <summary>
        ''' WeightsCount
        ''' </summary>
        Public Function GetWeightsCount%()

            Dim numWeights% = 0
            For layerNum = 1 To Me.LayerCount - 1
                numWeights += (Me.Sizes(layerNum - 1) + 1) * Me.Sizes(layerNum)
            Next
            Return numWeights

        End Function

        Public Sub SetWeights(weights#())

            ' copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
            Dim numWeights% = GetWeightsCount()
            If weights.Length <> numWeights Then Throw New Exception("Bad weights array in SetWeights")

            Dim k% = 0 ' points into weights param
            For layerNum = 1 To Me.LayerCount - 1

                For i = 0 To Me.Sizes(layerNum) - 1
                    For j = 0 To Me.Sizes(layerNum - 1) - 1
                        Me.Neurons(layerNum).Weights(i)(j) = weights(k)
                        k += 1
                    Next
                Next

                For i = 0 To Me.Sizes(layerNum) - 1
                    Me.Neurons(layerNum).Biases(i) = weights(k)
                    k += 1
                Next

            Next

        End Sub

        Public Function GetWeights() As Double()

            Dim numWeights% = GetWeightsCount()
            Dim result#() = New Double(numWeights - 1) {}

            Dim k% = 0
            For layerNum = 1 To Me.LayerCount - 1

                For i = 0 To Me.Sizes(layerNum) - 1
                    For j = 0 To Me.Sizes(layerNum - 1) - 1
                        result(k) = Me.Neurons(layerNum).Weights(i)(j)
                        k += 1
                    Next
                Next

                For i = 0 To Me.Sizes(layerNum) - 1
                    result(k) = Me.Neurons(layerNum).Biases(i)
                    k += 1
                Next

            Next

            Return result

        End Function

        Public Function ComputeOutputs(xValues#(),
                Optional outputLayers#()() = Nothing) As Double()

            Dim field = If(outputLayers, Me.Layers)
            field(0) = xValues

            For layer = 1 To Me.LayerCount - 1
                field(layer) = New Double(Me.Sizes(layer) - 1) {}
                Array.Copy(Me.Neurons(layer).Biases, field(layer), Me.Sizes(layer))
                For j = 0 To Me.Sizes(layer) - 1
                    For i = 0 To Me.Sizes(layer - 1) - 1
                        field(layer)(j) += field(layer - 1)(i) * Me.Neurons(layer).Weights(j)(i)
                    Next
                Next
                field(layer) = ActivateFunction(field(layer), layer)
            Next

            Return field(Me.LayerCount - 1)

        End Function

        Public Function Accuracy(testData#()(), weights#()) As Double

            SetWeights(weights)
            ' percentage correct using winner-takes all
            Dim numCorrect% = 0
            Dim numWrong% = 0
            Dim lastLayerId% = Me.LayerCount - 1
            Dim xValues#() = New Double(Me.Sizes(0) - 1) {} ' inputs
            Dim tValues#() = New Double(Me.Sizes(lastLayerId) - 1) {} ' targets
            For Each t As Double() In testData
                Array.Copy(t, xValues, Me.Sizes(0)) ' parse data into x-values and t-values
                Array.Copy(t, Me.Sizes(0), tValues, 0, Me.Sizes(lastLayerId))
                Dim yValues = ComputeOutputs(xValues) ' computed Y
                Dim maxIndex% = AbstractNeuralNetwork.MaxIndex(yValues) ' which cell in yValues has largest value?

                If tValues(maxIndex) = 1.0 Then ' ugly. consider AreEqual(double x, double y, double epsilon)
                    numCorrect += 1
                Else
                    numWrong += 1
                End If

            Next

            Return numCorrect / (numCorrect + numWrong) ' ugly 2 - check for divide by zero

        End Function

        Public Function RootMeanSquaredError(trainData#()(), weights#()) As Double()

            Return If(Me.multiThread,
                RootMeanSquaredErrorMultiThread(trainData, weights),
                RootMeanSquaredErrorSingleThread(trainData, weights))

        End Function

        Public Function RootMeanSquaredErrorSingleThread(trainData#()(), weights#()) As Double()

            SetWeights(weights) ' copy the weights to evaluate in
            Dim lastLayerId% = Me.LayerCount - 1
            Dim outputSize% = Me.Sizes(lastLayerId)
            Dim trainDataSize% = trainData.Length
            Dim xValues#() = New Double(Me.Sizes(0) - 1) {} ' inputs
            Dim tValues#() = New Double(outputSize - 1) {} ' targets
            Dim sumSquaredError# = 0.0
            Dim sumSquaredErrorItem# = 0.0

            For Each t As Double() In trainData
                ' following assumes data has all x-values first, followed by y-values!
                Array.Copy(t, xValues, Me.Sizes(0)) ' extract inputs
                Array.Copy(t, Me.Sizes(0), tValues, 0, outputSize) ' extract targets
                Dim yValues#() = ComputeOutputs(xValues)

                For j = 0 To outputSize - 1
                    'Dim err# = Math.Pow(yValues(j) - tValues(j), 2)
                    Dim delta = yValues(j) - tValues(j)
                    Dim err# = delta * delta
                    sumSquaredError += err / trainDataSize
                    sumSquaredErrorItem += err / trainDataSize / outputSize
                Next
            Next

            Dim d#() = {Math.Sqrt(sumSquaredErrorItem), Math.Sqrt(sumSquaredError)}
            Return d

        End Function

        Private Shared Function MaxIndex%(vector As IReadOnlyList(Of Double)) ' helper for Accuracy()

            ' index of largest value
            Dim bigIndex% = 0
            Dim biggestVal# = vector(0)
            For i = 0 To vector.Count - 1
                If vector(i) <= biggestVal Then Continue For
                biggestVal = vector(i)
                bigIndex = i
            Next

            Return bigIndex

        End Function

        Public Sub Save(filename As String)

            Using fo As FileStream = File.Open(filename, FileMode.Create)
                Using writer As New BinaryWriter(fo)

                    writer.Write(Me.LayerCount)

                    For i = 0 To Me.LayerCount - 1
                        writer.Write(Me.Sizes(i))
                    Next

                    Dim weights#() = GetWeights()

                    For Each t# In weights
                        writer.Write(t)
                    Next

                    writer.Write(0)
                    writer.Write(4)

                End Using 'writer.Close()
            End Using 'fo.Close()

        End Sub

        ''' <summary>
        ''' Update all weights and biases
        ''' </summary>
        Protected Function ComputeGraduate(
                trainData#()(), allGradsAcc As WeightComposite()) As Double()

            Return If(Me.multiThread,
                ComputeGraduateMultiThread(trainData, allGradsAcc),
                ComputeGraduateSingleThread(trainData, allGradsAcc))

        End Function

        ''' <summary>
        ''' Update all weights and biases
        ''' </summary>
        Protected Function ComputeGraduateSingleThread(trainData#()(),
                allGradsAcc As WeightComposite()) As Double()

            Dim lastLayerId% = Me.LayerCount - 1
            Dim outputSize% = Me.Sizes(lastLayerId)
            Dim xValues#() = New Double(Me.Sizes(0) - 1) {} ' inputs
            Dim tValues#() = New Double(outputSize - 1) {} ' target values
            Dim sumSquaredErrors#() = {0, 0}

            For Each t As Double() In trainData
                ' no need to visit in random order because all rows processed before any updates ('batch')
                Array.Copy(t, xValues, Me.Sizes(0)) ' get the inputs
                Array.Copy(t, Me.Sizes(0), tValues, 0, outputSize) ' get the target values
                ' copy xValues in, compute outputs using curr weights (and store outputs internally)
                Dim yValues#() = ComputeOutputs(xValues)

                Dim gradTerms = CalculateGradTerms(Me.Layers, tValues)

                For layer = lastLayerId To 0 + 1 Step -1
                    ' add input to h-o component to make h-o weight gradients, and accumulate
                    For j = 0 To Me.Sizes(layer) - 1
                        Dim grad# = gradTerms(layer)(j)
                        allGradsAcc(layer).Biases(j) += grad
                        For i = 0 To Me.Sizes(layer - 1) - 1
                            grad = gradTerms(layer)(j) * Me.Layers(layer - 1)(i)
                            allGradsAcc(layer).Weights(j)(i) += grad
                        Next
                    Next
                Next

                For j = 0 To outputSize - 1
                    Dim delta# = yValues(j) - tValues(j)
                    Dim err# = delta * delta 'Math.Pow(delta, 2)
                    sumSquaredErrors(sumIndice) += err / trainData.Length
                    sumSquaredErrors(averageIndice) += err / trainData.Length / Me.Sizes(Me.LayerCount - 1)
                Next

            Next

            Return sumSquaredErrors

        End Function

        ''' <summary>
        ''' Calculating the gradient in multiple streams
        ''' </summary>
        Protected Function ComputeGraduateMultiThread(trainData#()(),
                allGradsAcc As WeightComposite()) As Double()

            Dim taskFactory As TaskFactory = New TaskFactory()
            Dim tasks As Task() = New Task(Me.ThreadCount - 1) {}
            Dim threadInputData As ThreadInputDatum() =
                New ThreadInputDatum(Me.ThreadCount - 1) {}
            InitializeThreads(threadInputData, trainData)

            Dim innerTrainData As List(Of Double()) = New List(Of Double())(trainData)
            Dim innerTrainDataChunk As List(Of Double()) = New List(Of Double())()

            ' multithread version does not compute the same value using this:
            'Dim chunk_size% = CInt(innerTrainData.Count * 0.8 / Me.ThreadCount)
            'Dim rest% = CInt(innerTrainData.Count * 0.8) Mod Me.ThreadCount
            ' Warning: this solution will not always work:
            'Debug.WriteLine("ThreadCount=" & Me.ThreadCount)
            'Debug.WriteLine("innerTrainData.Count=" & innerTrainData.Count)
            Dim chunk_size% = CInt(innerTrainData.Count / Me.ThreadCount)
            'Debug.WriteLine("chunk_size=" & chunk_size)

            While innerTrainData.Count > 0
                Dim currentThread% = -1

                For i = 0 To Me.ThreadCount - 1
                    If tasks(i) Is Nothing OrElse tasks(i).IsCompleted Then
                        currentThread = i
                        Exit For
                    End If
                Next

                If currentThread = -1 Then
                    Thread.Sleep(20)
                    Continue While
                End If

                innerTrainDataChunk.Clear()
                While innerTrainDataChunk.Count < chunk_size AndAlso innerTrainData.Count > 0
                    innerTrainDataChunk.Add(innerTrainData(0))
                    innerTrainData.RemoveAt(0)
                End While

                threadInputData(currentThread).trainDatum = innerTrainDataChunk.ToArray()

                For layer = 1 To Me.LayerCount - 1
                    ' zero-out values from prev. iteration
                    ZeroOut(threadInputData(currentThread).allGradsAcc(layer).Weights)
                    ZeroOut(threadInputData(currentThread).allGradsAcc(layer).Biases)
                Next

                tasks(currentThread) = taskFactory.StartNew(
                    New Action(Of Object)(AddressOf ComputeGraduateInThread),
                        threadInputData(currentThread))

            End While

            For i = 0 To Me.ThreadCount - 1
                If tasks(i) IsNot Nothing Then
                    tasks(i).Wait()
                End If
            Next

            ' All in allGradsAcc
            For i = 0 To Me.ThreadCount - 1
                For layer = 1 To Me.LayerCount - 1
                    For size = 0 To Me.Sizes(layer) - 1
                        allGradsAcc(layer).Biases(size) +=
                            threadInputData(i).allGradsAcc(layer).Biases(size)
                        For prev_size = 0 To Me.Sizes(layer - 1) - 1
                            allGradsAcc(layer).Weights(size)(prev_size) +=
                                threadInputData(i).allGradsAcc(layer).Weights(size)(prev_size)
                        Next
                    Next
                Next
            Next

            Dim sumSquaredErrorItem# = 0
            Dim sumSquaredError# = 0
            For i = 0 To Me.ThreadCount - 1
                sumSquaredError += threadInputData(i).sumSquaredErrors(sumIndice)
                sumSquaredErrorItem += threadInputData(i).sumSquaredErrors(averageIndice)
            Next

            Dim d#() = {Math.Sqrt(sumSquaredErrorItem), Math.Sqrt(sumSquaredError)}
            Return d

        End Function

        Private Sub InitializeThreads(threadInputData As ThreadInputDatum(), trainData#()())

            For i = 0 To Me.ThreadCount - 1
                threadInputData(i).field = New Double(Me.LayerCount - 1)() {}
                threadInputData(i).allGradsAcc = New WeightComposite(Me.LayerCount - 1) {}
                threadInputData(i).xValues = New Double(Me.Sizes(0) - 1) {} ' inputs
                threadInputData(i).tValues = New Double(Me.Sizes(Me.LayerCount - 1) - 1) {} ' targets
                threadInputData(i).delim1 = 1.0 / trainData.Length
                threadInputData(i).delim2 = 1.0 / trainData.Length / Me.Sizes(Me.LayerCount - 1)
                threadInputData(i).sumSquaredErrors = New Double() {0, 0}

                For j = 0 To Me.LayerCount - 1
                    threadInputData(i).field(j) = New Double(Me.Sizes(j) - 1) {}
                    If j <= 0 Then Continue For
                    threadInputData(i).allGradsAcc(j).Biases = New Double(Me.Sizes(j) - 1) {}
                    threadInputData(i).allGradsAcc(j).Weights =
                        MakeMatrix(Me.Sizes(j), Me.Sizes(j - 1), 0.0)
                Next
            Next

        End Sub

        Public Sub ComputeGraduateInThread(input As Object)

            Dim inputDatum As ThreadInputDatum = CType(input, ThreadInputDatum)
            Dim lastLayerId% = Me.LayerCount - 1

            For Each t As Double() In inputDatum.trainDatum

                ' no need to visit in random order because all rows processed before
                '  any updates ('batch')
                Array.Copy(t, inputDatum.xValues, Me.Sizes(0)) ' get the inputs
                ' get the target values
                Array.Copy(t, Me.Sizes(0), inputDatum.tValues, 0, Me.Sizes(lastLayerId))
                ' copy xValues in, compute outputs using curr weights
                '  (and store outputs internally)
                Dim yValues = ComputeOutputs(inputDatum.xValues, inputDatum.field)
                Dim gradTerms = CalculateGradTerms(inputDatum.field, inputDatum.tValues)

                For layer = lastLayerId To 0 + 1 Step -1
                    ' add input to h-o component to make h-o weight gradients, and accumulate
                    For j = 0 To Me.Sizes(layer) - 1
                        Dim grad# = gradTerms(layer)(j)
                        inputDatum.allGradsAcc(layer).Biases(j) += grad
                        For i = 0 To Me.Sizes(layer - 1) - 1
                            grad = gradTerms(layer)(j) * inputDatum.field(layer - 1)(i)
                            inputDatum.allGradsAcc(layer).Weights(j)(i) += grad
                        Next
                    Next
                Next

                For j = 0 To Me.Sizes(lastLayerId) - 1
                    'Dim err# = Math.Pow(yValues(j) - inputDatum.tValues(j), 2)
                    Dim delta = yValues(j) - inputDatum.tValues(j)
                    Dim err# = delta * delta
                    inputDatum.sumSquaredErrors(sumIndice) += err * inputDatum.delim1
                    inputDatum.sumSquaredErrors(averageIndice) += err * inputDatum.delim2
                Next

            Next

        End Sub

        Public Function RootMeanSquaredErrorMultiThread(trainData#()(), weights#()) As Double()

            SetWeights(weights) ' copy the weights to evaluate in
            Dim taskFactory As New TaskFactory()
            Dim tasks As Task() = New Task(Me.ThreadCount - 1) {}
            Dim lastLayerId% = Me.LayerCount - 1
            Dim outputSize% = Me.Sizes(lastLayerId)
            Dim threadInputData = New RMSEThreadInputDatum(Me.ThreadCount - 1) {}
            InitializeThreads2(threadInputData, trainData, outputSize)

            Dim innerTrainData As List(Of Double()) = New List(Of Double())(trainData)
            Dim innerTrainDataChunk As List(Of Double()) = New List(Of Double())()
            Dim chunk_size% = CInt(innerTrainData.Count * 0.8 / Me.ThreadCount)

            While innerTrainData.Count > 0
                Dim currentThread% = -1

                For i = 0 To Me.ThreadCount - 1
                    If tasks(i) Is Nothing OrElse tasks(i).IsCompleted Then
                        currentThread = i
                        Exit For
                    End If
                Next

                If currentThread = -1 Then
                    Thread.Sleep(20)
                    Continue While
                End If

                innerTrainDataChunk.Clear()

                While innerTrainDataChunk.Count < chunk_size AndAlso innerTrainData.Count > 0
                    innerTrainDataChunk.Add(innerTrainData(0))
                    innerTrainData.RemoveAt(0)
                End While

                threadInputData(currentThread).trainDatum = innerTrainDataChunk.ToArray()
                tasks(currentThread) = taskFactory.StartNew(
                    New Action(Of Object)(AddressOf ComputeRMSEInThread),
                        threadInputData(currentThread))

            End While

            For i = 0 To Me.ThreadCount - 1
                If tasks(i) IsNot Nothing Then
                    tasks(i).Wait()
                End If
            Next

            Dim sumSquaredErrorItem# = 0
            Dim sumSquaredError# = 0
            For i = 0 To Me.ThreadCount - 1
                sumSquaredError += threadInputData(i).sumSquaredErrors(sumIndice)
                sumSquaredErrorItem += threadInputData(i).sumSquaredErrors(averageIndice)
            Next

            Dim d#() = {Math.Sqrt(sumSquaredErrorItem), Math.Sqrt(sumSquaredError)}
            Return d

        End Function

        Private Sub InitializeThreads2(threadInputData As RMSEThreadInputDatum(),
            trainData#()(), outputSize%)

            For i = 0 To Me.ThreadCount - 1
                threadInputData(i).field = New Double(Me.LayerCount - 1)() {}
                threadInputData(i).xValues = New Double(Me.Sizes(0) - 1) {} ' inputs
                threadInputData(i).tValues = New Double(outputSize - 1) {} ' targets
                threadInputData(i).delim1 = 1.0 / trainData.Length
                threadInputData(i).delim2 = 1.0 / trainData.Length / outputSize
                threadInputData(i).sumSquaredErrors = New Double(1) {}

                For j = 0 To Me.LayerCount - 1
                    threadInputData(i).field(j) = New Double(Me.Sizes(j) - 1) {}
                Next
            Next

        End Sub

        Public Sub ComputeRMSEInThread(input As Object)

            Dim threadInputDatum = CType(input, RMSEThreadInputDatum)
            Dim outputSize% = Me.Sizes(Me.LayerCount - 1)

            For Each t As Double() In threadInputDatum.trainDatum
                ' following assumes data has all x-values first, followed by y-values!
                Array.Copy(t, threadInputDatum.xValues, Me.Sizes(0)) ' extract inputs
                Array.Copy(t, Me.Sizes(0), threadInputDatum.tValues, 0, outputSize) ' extract targets
                Dim yValues#() = ComputeOutputs(threadInputDatum.xValues, threadInputDatum.field)

                For j = 0 To outputSize - 1
                    'Dim err# = Math.Pow(yValues(j) - threadInputDatum.tValues(j), 2)
                    Dim delta# = yValues(j) - threadInputDatum.tValues(j)
                    Dim err# = delta * delta
                    threadInputDatum.sumSquaredErrors(sumIndice) += err * threadInputDatum.delim1
                    threadInputDatum.sumSquaredErrors(averageIndice) += err * threadInputDatum.delim2
                Next
            Next

        End Sub

        Protected Sub UpdateWeigtsAndBiases(allGradsAcc As WeightComposite(),
                prevGradsAcc As WeightComposite(), prevDeltas As WeightComposite())

            ' update input-hidden weights
            For layer = 1 To Me.LayerCount - 1

                Dim size% = Me.Sizes(layer)
                Dim previousSize% = Me.Sizes(layer - 1)
                For i = 0 To previousSize - 1
                    For j = 0 To size - 1

                        Dim delta# = prevDeltas(layer).Weights(j)(i)
                        Dim t# = prevGradsAcc(layer).Weights(j)(i) * allGradsAcc(layer).Weights(j)(i)

                        If t > 0 Then ' no sign change, increase delta

                            delta *= EtaPlus ' compute delta
                            ' keep it in range
                            If delta > DeltaMax Then delta = DeltaMax

                            ' determine direction and magnitude
                            Dim tmp# = -Math.Sign(allGradsAcc(layer).Weights(j)(i)) * delta
                            Me.Neurons(layer).Weights(j)(i) += tmp ' update weights

                        ElseIf t < 0 Then ' grad changed sign, decrease delta

                            delta *= EtaMinus ' the delta (not used, but saved for later)
                            ' keep it in range
                            If delta < DeltaMin Then delta = DeltaMin

                            ' revert to previous weight
                            Me.Neurons(layer).Weights(j)(i) -= prevDeltas(layer).Weights(j)(i)
                            ' forces next if-then branch, next iteration
                            allGradsAcc(layer).Weights(j)(i) = 0
                            ' this happens next iteration after 2nd branch above
                            '  (just had a change in gradient)
                        Else
                            ' no change to delta
                            ' no way should delta be 0 ...
                            ' determine direction
                            Dim tmp# = -Math.Sign(allGradsAcc(layer).Weights(j)(i)) * delta
                            Me.Neurons(layer).Weights(j)(i) += tmp ' update
                        End If

                        prevDeltas(layer).Weights(j)(i) = delta ' save delta
                        ' save the (accumulated) gradient
                        prevGradsAcc(layer).Weights(j)(i) = allGradsAcc(layer).Weights(j)(i)
                    Next
                Next

                ' update (input-to-) hidden biases
                For i = 0 To size - 1
                    Dim delta# = prevDeltas(layer).Biases(i)
                    Dim t# = prevGradsAcc(layer).Biases(i) * allGradsAcc(layer).Biases(i)

                    If t > 0 Then ' no sign change, increase delta

                        delta *= EtaPlus ' compute delta

                        If delta > DeltaMax Then delta = DeltaMax

                        ' determine direction
                        Dim tmp# = -Math.Sign(allGradsAcc(layer).Biases(i)) * delta
                        Me.Neurons(layer).Biases(i) += tmp ' update

                    ElseIf t < 0 Then ' grad changed sign, decrease delta

                        delta *= EtaMinus ' the delta (not used, but saved later)

                        If delta < DeltaMin Then delta = DeltaMin

                        ' revert to previous weight
                        Me.Neurons(layer).Biases(i) -= prevDeltas(layer).Biases(i)
                        allGradsAcc(layer).Biases(i) = 0 ' forces next branch, next iteration
                        ' this happens next iteration after 2nd branch above
                        '  (just had a change in gradient)

                    Else

                        If delta > DeltaMax Then
                            delta = DeltaMax
                        ElseIf delta < DeltaMin Then
                            delta = DeltaMin
                        End If

                        ' no way should delta be 0 . . .
                        ' determine direction
                        Dim tmp# = -Math.Sign(allGradsAcc(layer).Biases(i)) * delta
                        Me.Neurons(layer).Biases(i) += tmp ' update

                    End If

                    prevDeltas(layer).Biases(i) = delta
                    prevGradsAcc(layer).Biases(i) = allGradsAcc(layer).Biases(i)
                Next
            Next

        End Sub

        Public Sub ShowMessage(msg$)
            If isConsoleApp() Then Console.WriteLine(msg)
            Debug.WriteLine(msg)
        End Sub

    End Class ' NeuralNetwork

End Class

' Nguyen-Widrow weights distribution examples (the last column is the bias):
'
' neuron count={2, 2, 1}
' 
' W(1)={
'  {0.700, 0.700, 0.001},
'  {0.700, 0.700, 0.001}}
' 
' W(2)={
'  {0.495, 0.495, 0.001}}
' 
' 
' neuron count={2, 2, 2, 1}
' 
' W(1)={
'  {0.700, 0.700, 0.001},
'  {0.700, 0.700, 0.001}}
' 
' W(2)={
'  {0.700, 0.700, 0.001},
'  {0.700, 0.700, 0.001}}
' 
' W(3)={
'  {0.495, 0.495, 0.001}}
' 
' 
' neuron count={4, 4, 2}
' 
' W(1)={
'  {0.495, 0.495, 0.495, 0.495, 0.001},
'  {0.495, 0.495, 0.495, 0.495, 0.001},
'  {0.495, 0.495, 0.495, 0.495, 0.001},
'  {0.495, 0.495, 0.495, 0.495, 0.001}}
' 
' W(2)={
'  {0.416, 0.416, 0.416, 0.416, 0.001},
'  {0.416, 0.416, 0.416, 0.416, 0.001}}
' 
' 
' neuron count={4, 4, 4, 2}
' 
' W(1)={
'  {0.495, 0.495, 0.495, 0.495, 0.001},
'  {0.495, 0.495, 0.495, 0.495, 0.001},
'  {0.495, 0.495, 0.495, 0.495, 0.001},
'  {0.495, 0.495, 0.495, 0.495, 0.001}}
' 
' W(2)={
'  {0.495, 0.495, 0.495, 0.495, 0.001},
'  {0.495, 0.495, 0.495, 0.495, 0.001},
'  {0.495, 0.495, 0.495, 0.495, 0.001},
'  {0.495, 0.495, 0.495, 0.495, 0.001}}
' 
' W(3)={
'  {0.416, 0.416, 0.416, 0.416, 0.001},
'  {0.416, 0.416, 0.416, 0.416, 0.001}}
' 
' 
' neuron count={6, 6, 3}
' 
' W(1)={
'  {0.385, 0.385, 0.385, 0.385, 0.385, 0.385, 0.001},
'  {0.385, 0.385, 0.385, 0.385, 0.385, 0.385, 0.001},
'  {0.385, 0.385, 0.385, 0.385, 0.385, 0.385, 0.001},
'  {0.385, 0.385, 0.385, 0.385, 0.385, 0.385, 0.001},
'  {0.385, 0.385, 0.385, 0.385, 0.385, 0.385, 0.001},
'  {0.385, 0.385, 0.385, 0.385, 0.385, 0.385, 0.001}}
' 
' W(2)={
'  {0.343, 0.343, 0.343, 0.343, 0.343, 0.343, 0.001},
'  {0.343, 0.343, 0.343, 0.343, 0.343, 0.343, 0.001},
'  {0.343, 0.343, 0.343, 0.343, 0.343, 0.343, 0.001}}
' 
' 
' neuron count={6, 6, 6, 3}
' 
' W(1)={
'  {0.385, 0.385, 0.385, 0.385, 0.385, 0.385, 0.001},
'  {0.385, 0.385, 0.385, 0.385, 0.385, 0.385, 0.001},
'  {0.385, 0.385, 0.385, 0.385, 0.385, 0.385, 0.001},
'  {0.385, 0.385, 0.385, 0.385, 0.385, 0.385, 0.001},
'  {0.385, 0.385, 0.385, 0.385, 0.385, 0.385, 0.001},
'  {0.385, 0.385, 0.385, 0.385, 0.385, 0.385, 0.001}}
' 
' W(2)={
'  {0.385, 0.385, 0.385, 0.385, 0.385, 0.385, 0.001},
'  {0.385, 0.385, 0.385, 0.385, 0.385, 0.385, 0.001},
'  {0.385, 0.385, 0.385, 0.385, 0.385, 0.385, 0.001},
'  {0.385, 0.385, 0.385, 0.385, 0.385, 0.385, 0.001},
'  {0.385, 0.385, 0.385, 0.385, 0.385, 0.385, 0.001},
'  {0.385, 0.385, 0.385, 0.385, 0.385, 0.385, 0.001}}
' 
' W(3)={
'  {0.343, 0.343, 0.343, 0.343, 0.343, 0.343, 0.001},
'  {0.343, 0.343, 0.343, 0.343, 0.343, 0.343, 0.001},
'  {0.343, 0.343, 0.343, 0.343, 0.343, 0.343, 0.001}}
' 
' 
' neuron count={4, 20, 3}
' 
' W(1)={
'  {0.740, 0.740, 0.740, 0.740, 0.001},
'  {0.740, 0.740, 0.740, 0.740, 0.001},
'  {0.740, 0.740, 0.740, 0.740, 0.001},
'  {0.740, 0.740, 0.740, 0.740, 0.001},
'  {0.740, 0.740, 0.740, 0.740, 0.001},
'  {0.740, 0.740, 0.740, 0.740, 0.001},
'  {0.740, 0.740, 0.740, 0.740, 0.001},
'  {0.740, 0.740, 0.740, 0.740, 0.001},
'  {0.740, 0.740, 0.740, 0.740, 0.001},
'  {0.740, 0.740, 0.740, 0.740, 0.001},
'  {0.740, 0.740, 0.740, 0.740, 0.001},
'  {0.740, 0.740, 0.740, 0.740, 0.001},
'  {0.740, 0.740, 0.740, 0.740, 0.001},
'  {0.740, 0.740, 0.740, 0.740, 0.001},
'  {0.740, 0.740, 0.740, 0.740, 0.001},
'  {0.740, 0.740, 0.740, 0.740, 0.001},
'  {0.740, 0.740, 0.740, 0.740, 0.001},
'  {0.740, 0.740, 0.740, 0.740, 0.001},
'  {0.740, 0.740, 0.740, 0.740, 0.001},
'  {0.740, 0.740, 0.740, 0.740, 0.001}}
' 
' W(2)={
'  {0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.001},
'  {0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.001},
'  {0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.001}}
' 
' 
' neuron count={4, 16, 8, 3}
' 
' W(1)={
'  {0.700, 0.700, 0.700, 0.700, 0.001},
'  {0.700, 0.700, 0.700, 0.700, 0.001},
'  {0.700, 0.700, 0.700, 0.700, 0.001},
'  {0.700, 0.700, 0.700, 0.700, 0.001},
'  {0.700, 0.700, 0.700, 0.700, 0.001},
'  {0.700, 0.700, 0.700, 0.700, 0.001},
'  {0.700, 0.700, 0.700, 0.700, 0.001},
'  {0.700, 0.700, 0.700, 0.700, 0.001},
'  {0.700, 0.700, 0.700, 0.700, 0.001},
'  {0.700, 0.700, 0.700, 0.700, 0.001},
'  {0.700, 0.700, 0.700, 0.700, 0.001},
'  {0.700, 0.700, 0.700, 0.700, 0.001},
'  {0.700, 0.700, 0.700, 0.700, 0.001},
'  {0.700, 0.700, 0.700, 0.700, 0.001},
'  {0.700, 0.700, 0.700, 0.700, 0.001},
'  {0.700, 0.700, 0.700, 0.700, 0.001}}
' 
' W(2)={
'  {0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.001},
'  {0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.001},
'  {0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.001},
'  {0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.001},
'  {0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.001},
'  {0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.001},
'  {0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.001},
'  {0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.199, 0.001}}
' 
' W(3)={
'  {0.284, 0.284, 0.284, 0.284, 0.284, 0.284, 0.284, 0.284, 0.001},
'  {0.284, 0.284, 0.284, 0.284, 0.284, 0.284, 0.284, 0.284, 0.001},
'  {0.284, 0.284, 0.284, 0.284, 0.284, 0.284, 0.284, 0.284, 0.001}}