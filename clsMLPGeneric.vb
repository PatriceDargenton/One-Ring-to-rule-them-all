
Imports Perceptron.MLP.ActivationFunction
Imports Perceptron.Utility ' Matrix
Imports System.Text ' StringBuilder

''' <summary>
''' MultiLayer Perceptron (MLP) generic class
''' </summary>
Public MustInherit Class clsMLPGeneric

#Region "Declaration"

    Public Overridable Sub InitializeStruct(neuronCount%(), addBiasColumn As Boolean)
        Me.useBias = addBiasColumn
        Me.layerCount = neuronCount.Length
        Me.neuronCount = neuronCount
        Me.nbInputNeurons = Me.neuronCount(0)
        Me.nbHiddenNeurons = Me.neuronCount(1)
        Me.nbOutputNeurons = Me.neuronCount(Me.layerCount - 1)
    End Sub

    Public MustOverride Sub InitializeWeights(layer%, weights#(,))

    Protected nbInputNeurons%
    Protected nbHiddenNeurons%
    Protected nbOutputNeurons%

    ''' <summary>
    ''' Round random weights to reproduce functionnal tests exactly
    ''' </summary>
    Public Const nbRoundingDigits% = 2

    ''' <summary>
    ''' Set a random value other than 0 to avoid to nullify the gradient
    ''' (only for RProp MLP for the moment)
    ''' </summary>
    Public Const minRandomValue! = 0.0001!

    ''' <summary>
    ''' Use Nguyen-Widrow weights initialization (only for RProp MLP for the moment)
    ''' </summary>
    Public useNguyenWidrowWeightsInitialization As Boolean = False

    Public Const expMax! = 50
    Public Const expMax20! = 20

    Public Enum enumLearningMode
        Defaut = Systematic
        ''' <summary>
        ''' Learn all samples in order
        ''' </summary>
        Systematic = 0
        ''' <summary>
        ''' Learn all samples randomly
        ''' </summary>
        SemiStochastic = 1
        ''' <summary>
        ''' Learn samples randomly
        ''' </summary>
        Stochastic = 2
        ''' <summary>
        ''' Learn all samples in order as a vector
        ''' </summary>
        Vectorial = 3
        ''' <summary>
        ''' Learn all samples in order as a vector for a batch of iterations
        ''' </summary>
        VectorialBatch = 4
    End Enum

    ''' <summary>
    ''' Learning mode of the MLP
    ''' </summary>
    Public learningMode As enumLearningMode

    Public printOutput_ As Boolean = False
    Public printOutputMatrix As Boolean = False
    Public useBias As Boolean = False

    Public inputArray!(,)
    Public targetArray!(,)

    ''' <summary>
    ''' Output matrix
    ''' </summary>
    Public output As Matrix

    ''' <summary>
    ''' Output array 1D
    ''' </summary>
    Public lastOutputArray1DSingle!()

    ''' <summary>
    ''' Output array 1D
    ''' </summary>
    Public lastOutputArray1D#() ' 29/11/2020

    Public lastErrorArray#(,)

    ''' <summary>
    ''' Last error of the output matrix
    ''' </summary>
    Protected lastError As Matrix

    ''' <summary>
    ''' Result success matrix (1: success, 0: fail)
    ''' </summary>
    Protected success As Matrix

    ''' <summary>
    ''' Number of success according to the treshold between target and output
    ''' </summary>
    Public nbSuccess%

    ''' <summary>
    ''' Percentage of success according to the number of ouputs 
    ''' </summary>
    Public successPC!

    ''' <summary>
    ''' Output must be 10% close to target to be considered successful
    ''' </summary>
    Public minimalSuccessTreshold! = 0.1 ' 10%

    Public nbIterations%

    Protected layerCount%
    Protected neuronCount%()

    ''' <summary>
    ''' Learning rate of the MLP (Eta coeff.)
    ''' </summary>
    Public learningRate!

    ''' <summary>
    ''' Weight adjustment of the MLP (Alpha coeff. or momentum)
    ''' (can be 0, but works best if 0.1 for example)
    ''' </summary>
    Public weightAdjustment!

    ''' <summary>
    ''' Set true if the objective is to classify an output (1 among N)
    '''  within a homogeneous group (for example, the Iris Flower logical test,
    '''  but not the 2XOR test, nor the 3XOR test, because the outputs are independent)
    ''' Then a softmax activation is used for last layer, it can speed up training
    ''' (only for RProp MLP for the moment)
    ''' </summary>
    Public classificationObjective As Boolean = False

    Public Sub Initialize(learningRate!, Optional weightAdjustment! = 0)

        Me.learningRate = learningRate
        Me.weightAdjustment = weightAdjustment

    End Sub

    ''' <summary>
    ''' Average error of the output matrix
    ''' </summary>
    Public averageError#

    ''' <summary>
    ''' Random generator (Shared)
    ''' </summary>
    Public Shared rndShared As New Random

    ''' <summary>
    ''' Random generator
    ''' </summary>
    Public rnd As Random

#End Region

#Region "Activation function"

    ''' <summary>
    ''' Lambda function for the activation function
    ''' </summary>
    Protected lambdaFnc As Func(Of Double, Double)

    ''' <summary>
    ''' Lambda function for the derivative of the activation function
    ''' </summary>
    Protected lambdaFncD As Func(Of Double, Double)

    ''' <summary>
    ''' Lambda function for the derivative of the activation function,
    ''' from the original function
    ''' </summary>
    Protected lambdaFncDFOF As Func(Of Double, Double)

    ''' <summary>
    ''' Activate function of each neuron of the MLP
    ''' </summary>
    Protected activFnc As MLP.ActivationFunction.IActivationFunction

    Protected m_gain!
    Protected m_center!
    Protected m_actFunc As enumActivationFunction = enumActivationFunction.Undefined

    Public Overridable Function GetActivationFunctionType() As enumActivationFunctionType
        Return enumActivationFunctionType.Normal
    End Function


    ''' <summary>
    ''' Set registered activation function
    ''' </summary>
    Public Overridable Sub SetActivationFunction(actFnc As enumActivationFunction,
        Optional gain! = 1,
        Optional center! = 0)

        Select Case actFnc
            Case enumActivationFunction.Undefined : Me.activFnc = Nothing
            Case enumActivationFunction.Identity : Me.activFnc = New IdentityFunction
            Case enumActivationFunction.Sigmoid : Me.activFnc = New SigmoidFunction
            Case enumActivationFunction.HyperbolicTangent : Me.activFnc = New HyperbolicTangentFunction
            Case enumActivationFunction.Gaussian : Me.activFnc = New GaussianFunction
            Case enumActivationFunction.ArcTangent : Me.activFnc = New ArcTangentFunction
            Case enumActivationFunction.Sinus : Me.activFnc = New SinusFunction
            Case enumActivationFunction.ELU : Me.activFnc = New ELUFunction
            Case enumActivationFunction.ReLu : Me.activFnc = New ReLuFunction
            Case enumActivationFunction.ReLuSigmoid : Me.activFnc = New ReLuSigmoidFunction
            Case enumActivationFunction.DoubleThreshold : Me.activFnc = New DoubleThresholdFunction
            Case Else
                Me.activFnc = Nothing
                Throw New ArgumentException("Activation function undefined!")
        End Select

        If Not IsNothing(Me.activFnc) Then
            Me.lambdaFnc = Function(x#) Me.activFnc.Activation(x, gain, center)
            Me.lambdaFncD = Function(x#) Me.activFnc.Derivative(x, gain, center)
            Me.lambdaFncDFOF = Function(x#) Me.activFnc.DerivativeFromOriginalFunction(x, gain)
        End If
        m_gain = gain
        m_center = center
        m_actFunc = actFnc

    End Sub

    ''' <summary>
    ''' Activation function using optimised derivative: 
    ''' </summary>
    Public Overridable Sub SetActivationFunctionOptimized(
        fctAct As enumActivationFunctionOptimized, Optional gain! = 1, Optional center! = 0)

        Select Case fctAct
            Case enumActivationFunctionOptimized.Sigmoid
                Me.m_actFunc = enumActivationFunction.Sigmoid
                Me.activFnc = New SigmoidFunction
            Case enumActivationFunctionOptimized.HyperbolicTangent
                Me.m_actFunc = enumActivationFunction.HyperbolicTangent
                Me.activFnc = New HyperbolicTangentFunction
            Case enumActivationFunctionOptimized.ELU
                Me.m_actFunc = enumActivationFunction.ELU
                Me.activFnc = New ELUFunction
            Case Else
                Me.activFnc = Nothing
                Me.m_actFunc = enumActivationFunction.Undefined
                Throw New ArgumentException("Activation function undefined!")
        End Select

        Me.lambdaFnc = Function(x#) Me.activFnc.Activation(x, gain, center)
        Me.lambdaFncD = Function(x#) Me.activFnc.Derivative(x, gain, center)
        'Me.lambdaFncD = Function(x#) Me.activFnc.DerivativeFromOriginalFunction(x, gain)
        Me.lambdaFncDFOF = Function(x#) Me.activFnc.DerivativeFromOriginalFunction(x, gain)

        m_gain = gain
        m_center = center

        ' Optimized activation function must be expressed from its direct function: f'(x)=g(f(x))
        If Not IsNothing(Me.activFnc) AndAlso
           Not Me.activFnc.DoesDerivativeDependOnOriginalFunction() Then _
            MsgBox("Activation function must be like this form: f'(x)=g(f(x))",
                MsgBoxStyle.Exclamation)

    End Sub

#End Region

#Region "Randomize"

    ''' <summary>
    ''' Randomize weights
    ''' </summary>
    Public MustOverride Sub Randomize(Optional minValue! = -0.5!, Optional maxValue! = 0.5!)

    Public Overridable Sub RoundWeights()
    End Sub

#End Region

#Region "Error"

    ''' <summary>
    ''' Compute error of the output matrix for all samples
    ''' </summary>
    Public Overridable Sub ComputeError()
        ' Calculate the error: ERROR = TARGETS - OUTPUTS
        Dim m As Matrix = Me.targetArray
        Me.lastError = m - Me.output
        ComputeSuccess()
    End Sub

    ''' <summary>
    ''' Compute error of the output matrix for one sample
    ''' </summary>
    Public Overridable Sub ComputeErrorOneSample(targetArray!(,))
        ' Calculate the error: ERROR = TARGETS - OUTPUTS
        Dim m As Matrix = targetArray
        Me.lastError = m - Me.output
        ComputeSuccess()
    End Sub

    Public Overridable Sub ComputeSuccess()
        Me.success = Me.lastError.Threshold(minimalSuccessTreshold)
        Dim sum# = Me.success.Sumatory()(0, 0)
        Me.nbSuccess = CInt(Math.Round(sum))
        Me.successPC = CSng(Me.nbSuccess / (Me.success.r * Me.success.c))
    End Sub

    ''' <summary>
    ''' Compute average error of the output matrix for all samples from last error layer
    ''' </summary>
    Public Overridable Sub ComputeAverageErrorFromLastError()
        ' Compute first abs then average:
        Me.averageError = Me.lastError.Abs.Average
    End Sub

    ''' <summary>
    ''' Compute average error of the output matrix for all samples
    ''' </summary>
    Public Overridable Function ComputeAverageError#()
        Me.ComputeError()
        Me.ComputeAverageErrorFromLastError()
        Return Me.averageError
    End Function

    ''' <summary>
    ''' Compute average error of the output matrix for one sample
    ''' </summary>
    Public Overridable Function ComputeAverageErrorOneSample#(targetArray!(,))
        Me.ComputeErrorOneSample(targetArray)
        Me.ComputeAverageErrorFromLastError()
        Return Me.averageError
    End Function

#End Region

#Region "Train"

    ''' <summary>
    ''' Train one sample (run one iteration)
    ''' </summary>
    Public MustOverride Sub TrainOneSample(input!(), target!())

    ''' <summary>
    ''' Train all samples (run epoch for one iteration)
    ''' </summary>
    Public Sub Train(Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Dim sw As New Stopwatch
        sw.Start()
        Debug.WriteLine(Now() & " Train...")
        Train(Me.inputArray, Me.targetArray, Me.nbIterations, learningMode)
        sw.Stop()
        Debug.WriteLine(Now() & " Train: Done. " &
            sw.Elapsed.TotalSeconds.ToString("0.0") & " sec.")
        ' If it is not already printed, print now
        If Not Me.printOutput_ Then PrintSuccess(Me.nbIterations - 1)

    End Sub

    Public Sub Train(nbIterations%,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)
        Train(Me.inputArray, Me.targetArray, nbIterations, learningMode)
    End Sub

    Public Sub Train(inputs!(,), nbIterations%,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)
        Train(inputs, Me.targetArray, nbIterations, learningMode)
    End Sub

    Public Sub Train(inputs!(,), targets!(,), nbIterations%,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Me.nbIterations = nbIterations
        Me.learningMode = learningMode
        Select Case learningMode
            Case enumLearningMode.Vectorial, enumLearningMode.VectorialBatch
                TrainSystematic(inputs, targets, learningMode)
            Case enumLearningMode.Systematic
                TrainSystematic(inputs, targets)
            Case enumLearningMode.SemiStochastic
                TrainSemiStochastic(inputs, targets)
            Case enumLearningMode.Stochastic
                TrainStochastic(inputs, targets)
        End Select

    End Sub

    ''' <summary>
    ''' Train all samples (one iteration)
    ''' </summary>
    Public Sub TrainAllSamples(inputs!(,), targets!(,))

        Dim nbLines = inputs.GetLength(0)
        For j = 0 To nbLines - 1 ' Systematic learning
            Dim inp = clsMLPHelper.GetVector(inputs, j)
            Dim targ = clsMLPHelper.GetVector(targets, j)
            TrainOneSample(inp, targ)
        Next

    End Sub

    ''' <summary>
    ''' Train samples in random order
    ''' </summary>
    Public Overridable Sub TrainStochastic(inputs!(,), targets!(,))

        Me.learningMode = enumLearningMode.Stochastic
        Dim nbLines = inputs.GetLength(0)
        Dim nbTargets = targets.GetLength(1)
        For iteration = 0 To Me.nbIterations - 1
            Dim r% = rndShared.Next(maxValue:=nbLines) ' Stochastic learning
            Dim inp = clsMLPHelper.GetVector(inputs, r)
            Dim targ = clsMLPHelper.GetVector(targets, r)
            TrainOneSample(inp, targ)
            If Me.printOutput_ Then PrintOutput(iteration)
        Next
        TestAllSamples(inputs, nbTargets)

    End Sub

    ''' <summary>
    ''' Train all samples in random order
    ''' </summary>
    Public Overridable Sub TrainSemiStochastic(inputs!(,), targets!(,))

        Me.learningMode = enumLearningMode.SemiStochastic
        Dim nbLines = inputs.GetLength(0)
        Dim nbInputs = inputs.GetLength(1)
        Dim nbTargets = targets.GetLength(1)

        For iteration = 0 To Me.nbIterations - 1

            ' Semi-stochastic learning
            Dim lstEch As New List(Of Integer)
            For i = 0 To nbLines - 1
                lstEch.Add(i)
            Next
            For j = 0 To nbLines - 1

                Dim nbItemsRemaining = lstEch.Count
                ' 28/05/2020 In two stages!
                Dim k = rndShared.Next(maxValue:=nbItemsRemaining)
                Dim r = lstEch(k)
                lstEch.RemoveAt(k)

                Dim inp = clsMLPHelper.GetVector(inputs, r)
                Dim targ = clsMLPHelper.GetVector(targets, r)
                TrainOneSample(inp, targ)

            Next j

            If Me.printOutput_ Then PrintOutput(iteration)

        Next
        TestAllSamples(inputs, nbTargets)

    End Sub

    ''' <summary>
    ''' Train all samples in order
    ''' </summary>
    Public Overridable Sub TrainSystematic(inputs!(,), targets!(,),
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Me.learningMode = learningMode
        Dim nbTargets = targets.GetLength(1)
        For iteration = 0 To Me.nbIterations - 1
            TrainAllSamples(inputs, targets)
            If Me.printOutput_ Then PrintOutput(iteration)
        Next
        TestAllSamples(inputs, nbTargets)

    End Sub

    ''' <summary>
    ''' Close the training session
    ''' </summary>
    Public Overridable Sub CloseTrainingSession()
    End Sub

#End Region

#Region "Test"

    ''' <summary>
    ''' Test one sample: Propagate the input signal into the MLP (feed forward)
    ''' </summary>
    Public MustOverride Sub TestOneSample(input!())

    ''' <summary>
    ''' Test one sample: Propagate the input signal into the MLP and return the ouput
    ''' </summary>
    Public Overridable Sub TestOneSample(input!(), ByRef ouput!())
        TestOneSample(input)
        ouput = Me.lastOutputArray1DSingle
    End Sub

    Public Overridable Sub TestOneSampleAndComputeError(input!(), target!()) ' 29/11/2020

        TestOneSample(input)

        Dim targetArray2D!(0, target.GetUpperBound(0))
        clsMLPHelper.Fill2DArrayOfSingle(targetArray2D, target, 0)
        ComputeAverageErrorOneSample(targetArray2D)

    End Sub

    ''' <summary>
    ''' Test all samples
    ''' </summary>
    Public Sub TestAllSamples(inputs!(,), nbOutputs%)
        Dim length = inputs.GetLength(0)
        Dim nbInputs = inputs.GetLength(1)
        Dim outputs!(0 To length - 1, 0 To nbOutputs - 1)
        For i = 0 To length - 1
            Dim inp = clsMLPHelper.GetVector(inputs, i)
            TestOneSample(inp)
            Dim output!() = Me.lastOutputArray1DSingle
            For j = 0 To output.GetLength(0) - 1
                outputs(i, j) = output(j)
            Next
        Next
        Me.output = outputs
        ComputeAverageError()
    End Sub

    ''' <summary>
    ''' Test all samples
    ''' </summary>
    Public Sub TestAllSamples(inputs!(,))
        Dim nbOutputs = Me.targetArray.GetLength(1)
        TestAllSamples(inputs, nbOutputs)
    End Sub

    ''' <summary>
    ''' Test all samples
    ''' </summary>
    Public Sub TestAllSamples(inputs!(,), targets!(,))
        Me.targetArray = targets
        Dim nbOutputs = targets.GetLength(1)
        TestAllSamples(inputs, nbOutputs)
    End Sub

    ''' <summary>
    ''' Test all samples
    ''' </summary>
    Public Sub TestAllSamples(inputs!(,), targets!(,), nbOutputs%)
        Me.targetArray = targets
        TestAllSamples(inputs, nbOutputs)
    End Sub

#End Region

#Region "Print"

    ''' <summary>
    ''' Print weights for functionnal test
    ''' </summary>
    Public Overridable Sub PrintWeights()
        ShowMessage(ShowWeights())
    End Sub

    Public Overridable Function ShowWeights$()
        Return ""
    End Function

    Public Overridable Sub PrintOutput(iteration%, Optional force As Boolean = False)

        If force OrElse ShowThisIteration(iteration) Then
            Dim nbTargets = Me.targetArray.GetLength(1)
            TestAllSamples(Me.inputArray, nbTargets)
            ComputeAverageError()
            PrintSuccess(iteration)
        End If

    End Sub

    Public Function ShowParameters$()

        Dim sb As New StringBuilder()
        sb.AppendLine("")
        If Me.learningMode <> enumLearningMode.Defaut Then sb.AppendLine(
            "learning mode=" & clsMLPHelper.ReadEnumDescription(Me.learningMode))
        sb.AppendLine("layer count=" & Me.layerCount)
        sb.AppendLine("neuron count=" & clsMLPHelper.ArrayToString(Me.neuronCount))
        sb.AppendLine("use bias=" & Me.useBias)
        If Me.learningRate <> 0 Then sb.AppendLine("learning rate=" & Me.learningRate)
        If Me.weightAdjustment <> 0 Then sb.AppendLine(
            "weight adjustment=" & Me.weightAdjustment)
        Dim afType = Me.GetActivationFunctionType()
        If afType <> enumActivationFunctionType.Normal Then _
            sb.AppendLine("activation function type=" & clsMLPHelper.ReadEnumDescription(afType))
        sb.AppendLine("activation function=" & clsMLPHelper.ReadEnumDescription(Me.m_actFunc))
        sb.AppendLine("gain=" & Me.m_gain)
        If Me.m_center <> 0 Then sb.AppendLine("center=" & Me.m_center)
        If Me.classificationObjective Then _
            sb.AppendLine("use softmax for last layer=True (ojective is classification)")
        If Me.useNguyenWidrowWeightsInitialization Then _
            sb.AppendLine("use Nguyen-Widrow weights initialization=True")
        If Me.minimalSuccessTreshold <> 0 Then sb.AppendLine(
            "minimal success treshold=" & Me.minimalSuccessTreshold)
        sb.AppendLine("")

        Return sb.ToString()

    End Function

    Public Sub PrintParameters()

        Dim sb As New StringBuilder()
        sb.AppendLine("")
        sb.AppendLine(Now() & " :")
        sb.Append(ShowParameters())
        ShowMessage(sb.ToString())

    End Sub

    Public Function ShowThisIteration(iteration%) As Boolean
        If (iteration >= Me.nbIterations - 1 OrElse
            iteration < 10 OrElse
            ((iteration + 1) Mod 10 = 0 AndAlso iteration < 100) OrElse
            ((iteration + 1) Mod 100 = 0 AndAlso iteration < 1000) OrElse
            ((iteration + 1) Mod 1000 = 0 AndAlso iteration < 10000) OrElse
            (iteration + 1) Mod 10000 = 0) Then Return True
        Return False
    End Function

    Public Sub ShowMessage(msg$)
        If isConsoleApp() Then Console.WriteLine(msg)
        Debug.WriteLine(msg)
    End Sub

    Protected Sub PrintSuccess(iteration%)
        Dim msg$ = vbLf & "Iteration n°" & iteration + 1 & "/" & nbIterations & vbLf
        If Me.printOutputMatrix Then msg &= "Output: " & Me.output.ToString() & vbLf
        If Not IsNothing(Me.success) Then msg &=
            "Average error: " & Me.averageError.ToString(format6Dec) & vbLf &
            "Success (" & (minimalSuccessTreshold).ToString("0%") & "): " &
            Me.nbSuccess & "/" & Me.success.r * Me.success.c & ": " &
            Me.successPC.ToString("0.0%")
        ShowMessage(msg)
    End Sub

    Public Sub PrintSuccessPrediction()
        Dim msg$ = vbLf &
            "Prediction: " & vbLf &
            "Success (" & (minimalSuccessTreshold).ToString("0%") & "): " &
            Me.nbSuccess & "/" & Me.success.r * Me.success.c & ": " &
            Me.successPC.ToString("0.0%")
        ShowMessage(msg)
    End Sub

#End Region

End Class