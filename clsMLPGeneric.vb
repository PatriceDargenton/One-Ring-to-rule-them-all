
Imports Perceptron.MLP.ActivationFunction

Public MustInherit Class clsMLPGeneric

#Region "Declaration"

    Public MustOverride Sub InitStruct(aiNeuronCount%(), addBiasColumn As Boolean)

    ''' <summary>
    ''' Round random weights to reproduce functionnal tests exactly
    ''' </summary>
    Public Const roundWeights% = 2

    Public Const expMax As Single = 50

    Public Enum enumLearningMode
        Defaut = Systematique
        ''' <summary>
        ''' Learn all samples in order
        ''' </summary>
        Systematique = 0
        ''' <summary>
        ''' Learn all samples randomly
        ''' </summary>
        SemiStochastique = 1
        ''' <summary>
        ''' Learn samples randomly
        ''' </summary>
        Stochastique = 2
        ''' <summary>
        ''' Leanr all samples in order as a vector
        ''' </summary>
        Vectoriel = 3
    End Enum

    Public printOutput_ As Boolean = False
    Public useBias As Boolean = False

    Public inputArray As Single(,)
    Public targetArray As Single(,)

    ''' <summary>
    ''' Output array (returned to compute average error, and discrete error)
    ''' </summary>
    Public outputArray As Double(,)

    ''' <summary>
    ''' Output array (as Single(,))
    ''' </summary>
    Public outputArraySingle As Single(,)

    ''' <summary>
    ''' Output array of the last layer
    ''' </summary>
    Public lastOutputArray As Double()
    Public lastOutputArraySingle As Single()

    Public lastErrorArray As Double(,)

    Public nbIterations%

    Protected layerCount%

    ''' <summary>
    ''' Learning rate of the MLP (Eta coeff.)
    ''' </summary>
    Protected learningRate!

    ''' <summary>
    ''' Weight adjustment of the MLP (Alpha coeff.)
    ''' </summary>
    Protected weightAdjustment!

    Public Sub Init(learningRate!, weightAdjustment!)

        Me.learningRate = learningRate
        Me.weightAdjustment = weightAdjustment

    End Sub

    ''' <summary>
    ''' Average error of the output matrix
    ''' </summary>
    Public averageError!

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

    ''' <summary>
    ''' Set registered activation function
    ''' </summary>
    Public Sub SetActivationFunction(ActFnc As TActivationFunction, gain!, center!)

        Select Case ActFnc
            Case TActivationFunction.Identity : Me.activFnc = New IdentityFunction
            Case TActivationFunction.Sigmoid : Me.activFnc = New SigmoidFunction
            Case TActivationFunction.HyperbolicTangent : Me.activFnc = New HyperbolicTangentFunction
            Case TActivationFunction.Gaussian : Me.activFnc = New GaussianFunction
            Case TActivationFunction.ArcTangent : Me.activFnc = New ArcTangentFunction
            Case TActivationFunction.Sinus : Me.activFnc = New SinusFunction
            Case TActivationFunction.ReLu : Me.activFnc = New ReLuFunction
            Case TActivationFunction.ELU : Me.activFnc = New ELUFunction
            Case TActivationFunction.ReLuSigmoid : Me.activFnc = New ReLuSigmoidFunction
            Case Else
                Stop
        End Select

        Me.lambdaFnc = Function(x#) Me.activFnc.Activation(x, gain, center)
        Me.lambdaFncD = Function(x#) Me.activFnc.Derivative(x, gain, center)
        Me.lambdaFncDFOF = Function(x#) Me.activFnc.DerivativeFromOriginalFunction(x, gain)
    End Sub

#End Region

#Region "Randomize"

    ''' <summary>
    ''' Randomize weights
    ''' </summary>
    Public MustOverride Sub Randomize(Optional minValue! = 0, Optional maxValue! = 1)

#End Region

#Region "Error"

    ''' <summary>
    ''' Compute error of the output matrix for all samples
    ''' </summary>
    Public MustOverride Sub ComputeError()

    ''' <summary>
    ''' Compute average error of the output matrix for all samples from last error layer
    ''' </summary>
    Public MustOverride Sub ComputeAverageErrorFromLastError()

    ''' <summary>
    ''' Compute average error of the output matrix for all samples
    ''' </summary>
    Public MustOverride Function ComputeAverageError!()

#End Region

#Region "Train"

    ''' <summary>
    ''' Train one sample
    ''' </summary>
    Public MustOverride Sub TrainOneSample(inputs_array!(), targets_array!())

    Public Sub Train(Optional learningMode As enumLearningMode = enumLearningMode.Defaut)
        Train(Me.inputArray, Me.targetArray, Me.nbIterations, learningMode)
    End Sub

    Public Sub Train(nbIterations%,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)
        Train(Me.inputArray, Me.targetArray, nbIterations, learningMode)
    End Sub

    Public Sub Train(inputArray!(,), nbIterations%,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)
        Train(inputArray, Me.targetArray, nbIterations, learningMode)
    End Sub

    Public Sub Train(inputArray!(,), targetArray!(,), nbIterations%,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Me.nbIterations = nbIterations
        Select Case learningMode
            Case enumLearningMode.Vectoriel
                TrainSystematic(inputArray, targetArray, learningMode)
            Case enumLearningMode.Systematique
                TrainSystematic(inputArray, targetArray)
            Case enumLearningMode.SemiStochastique
                TrainSemiStochastic(inputArray, targetArray)
            Case enumLearningMode.Stochastique
                TrainStochastic(inputArray, targetArray)
        End Select

    End Sub

    ''' <summary>
    ''' Train all samples (one iteration)
    ''' </summary>
    Public Sub TrainAllSamples(inputs!(,), targets!(,))

        Dim nbLines% = inputs.GetLength(0)
        For j As Integer = 0 To nbLines - 1 ' Systematic learning
            Dim inp = clsMLPHelper.GetVector(inputs, j)
            Dim targ = clsMLPHelper.GetVector(targets, j)
            TrainOneSample(inp, targ)
        Next

    End Sub

    ''' <summary>
    ''' Train samples in random order
    ''' </summary>
    Public Sub TrainStochastic(inputs!(,), targets!(,))

        Dim nbLines% = inputs.GetLength(0)
        For iteration As Integer = 0 To Me.nbIterations - 1
            Dim r% = rndShared.Next(maxValue:=nbLines) ' Stochastic learning
            Dim inp = clsMLPHelper.GetVector(inputs, r)
            Dim targ = clsMLPHelper.GetVector(targets, r)
            TrainOneSample(inp, targ)
            If Me.printOutput_ Then PrintOutput(iteration)
        Next

    End Sub

    ''' <summary>
    ''' Train all samples in random order
    ''' </summary>
    Public Sub TrainSemiStochastic(inputs!(,), targets!(,))

        Dim nbLines% = inputs.GetLength(0)
        Dim nbInputs% = inputs.GetLength(1)
        Dim nbTargets% = targets.GetLength(1)

        For iteration As Integer = 0 To Me.nbIterations - 1

            ' Semi-stochastic learning
            Dim lstEch As New List(Of Integer)
            For i As Integer = 0 To nbLines - 1
                lstEch.Add(i)
            Next
            For j As Integer = 0 To nbLines - 1

                Dim iNbElemRestants% = lstEch.Count
                Dim r% = rndShared.Next(maxValue:=iNbElemRestants)
                lstEch.RemoveAt(r)

                Dim inp = clsMLPHelper.GetVector(inputs, r)
                Dim targ = clsMLPHelper.GetVector(targets, r)
                TrainOneSample(inp, targ)

            Next j

            If Me.printOutput_ Then PrintOutput(iteration)

        Next

    End Sub

    ''' <summary>
    ''' Train all samples in order
    ''' </summary>
    Public Overridable Sub TrainSystematic(inputs!(,), targets!(,),
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Dim nbTargets% = targets.GetLength(1)
        For iteration As Integer = 0 To Me.nbIterations - 1
            TrainAllSamples(inputs, targets)
            If Me.printOutput_ Then PrintOutput(iteration)
        Next
        TestAllSamples(inputs, nbTargets)

    End Sub

#End Region

#Region "Test"

    ''' <summary>
    ''' Test one sample
    ''' </summary>
    Public MustOverride Sub TestOneSample(inputArray!())

    ''' <summary>
    ''' Test all samples
    ''' </summary>
    Public Sub TestAllSamples(inputs!(,), nbOutputs%)
        Dim length% = inputs.GetLength(0)
        Dim nbInputs% = inputs.GetLength(1)
        Dim outputs!(0 To length - 1, 0 To nbOutputs - 1)
        For i As Integer = 0 To length - 1
            Dim inp = clsMLPHelper.GetVector(inputs, i)
            TestOneSample(inp)
            Dim output!() = Me.lastOutputArraySingle
            For j As Integer = 0 To output.GetLength(0) - 1
                outputs(i, j) = output(j)
            Next
        Next
        Me.outputArraySingle = outputs
        Me.outputArray = clsMLPHelper.ConvertSingleToDouble(outputs)
    End Sub

#End Region

#Region "Print"

    ''' <summary>
    ''' Print weights for functionnal test
    ''' </summary>
    Public MustOverride Sub PrintWeights()

    Public MustOverride Sub PrintOutput(iteration%)

    Public Function ShowThisIteration(iteration%) As Boolean
        If (iteration < 10 OrElse
            ((iteration + 1) Mod 100 = 0 AndAlso iteration < 1000) OrElse
            ((iteration + 1) Mod 1000 = 0 AndAlso iteration < 10000) OrElse
            (iteration + 1) Mod 10000 = 0) Then Return True
        Return False
    End Function

    Public Sub ShowMessage(msg$)
        Console.WriteLine(msg)
        Debug.WriteLine(msg)
    End Sub

#End Region

End Class