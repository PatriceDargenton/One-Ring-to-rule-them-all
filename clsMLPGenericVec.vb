
Public MustInherit Class clsVectorizedMLPGeneric : Inherits clsMLPGeneric

    Public vectorizedLearningMode As Boolean = True
    Public exampleCount%

    'Protected neuronCount%()

    ''' <summary>
    ''' Train all samples at once (run epoch for all iterations: all samples ordered in one vector)
    ''' </summary>
    Public MustOverride Sub TrainVector()

    ''' <summary>
    ''' Train all samples at once (run epoch for one iteration: all samples ordered in one vector)
    ''' </summary>
    Public MustOverride Sub TrainVectorOneIteration()

    Public nbIterationsBatch% = 10
    ''' <summary>
    ''' Train all samples at once for a batch of iterations
    ''' </summary>
    Public Overridable Sub TrainVectorBatch(nbIterationsBatch%)

        ' Default implementation: call TrainVectorOneIteration()
        For iteration = 0 To nbIterationsBatch - 1
            TrainVectorOneIteration()
        Next
        SetOuput1D()

    End Sub

    ''' <summary>
    ''' Train all samples in batch learning mode
    ''' </summary>
    Public Overridable Sub TrainVectorBatch()

        Me.vectorizedLearningMode = True

        If Not Me.printOutput_ Then

            Dim iteration = 0
            Do While iteration < Me.nbIterations
                If iteration + Me.nbIterationsBatch > Me.nbIterations Then
                    Me.nbIterationsBatch = Me.nbIterations - iteration
                End If
                If Me.nbIterationsBatch <= 0 Then Exit Do
                TrainVectorBatch(Me.nbIterationsBatch)
                iteration += Me.nbIterationsBatch
            Loop

        Else

            Dim iteration = 0
            Do While iteration < Me.nbIterations

                Dim nbIterationsBatch0 = 1
                If iteration < 10 - 1 Then
                ElseIf iteration < 100 - 1 Then
                    nbIterationsBatch0 = 10
                ElseIf iteration < 1000 - 1 Then
                    nbIterationsBatch0 = 100
                Else
                    nbIterationsBatch0 = 1000
                End If
                If iteration + nbIterationsBatch0 > Me.nbIterations Then
                    nbIterationsBatch0 = Me.nbIterations - iteration
                End If
                If nbIterationsBatch0 > Me.nbIterationsBatch Then
                    nbIterationsBatch0 = Me.nbIterationsBatch
                End If
                If nbIterationsBatch0 <= 0 Then Exit Do

                TrainVectorBatch(nbIterationsBatch0)

                PrintOutput(iteration)
                If iteration + nbIterationsBatch0 >= Me.nbIterations Then
                    iteration = Me.nbIterations - 1
                    Exit Do
                End If
                iteration += nbIterationsBatch0

            Loop

            If Not ShowThisIteration(iteration) Then PrintOutput(iteration, force:=True)

        End If

        SetOuput1D()
        ComputeAverageError()

    End Sub

    Public Overrides Sub TrainSystematic(inputs!(,), targets!(,),
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        If learningMode = enumLearningMode.VectorialBatch Then
            TrainVectorBatch()
            Exit Sub
        End If

        If learningMode = enumLearningMode.Vectorial Then
            TrainVector()
            Exit Sub
        End If

        Me.vectorizedLearningMode = False
        Me.exampleCount = 1
        MyBase.TrainSystematic(inputs, targets, learningMode)

    End Sub

    Public Overrides Sub TrainStochastic(inputs!(,), targets!(,))
        Me.vectorizedLearningMode = False
        Me.exampleCount = 1
        MyBase.TrainStochastic(inputs, targets)
    End Sub

    Public Overrides Sub TrainSemiStochastic(inputs!(,), targets!(,))
        Me.vectorizedLearningMode = False
        Me.exampleCount = 1
        MyBase.TrainSemiStochastic(inputs, targets)
    End Sub

    Public Overridable Sub SetOuput1D()
    End Sub

    Public Overrides Sub PrintOutput(iteration%, Optional force As Boolean = False)

        If force OrElse ShowThisIteration(iteration) Then
            If Not Me.vectorizedLearningMode Then
                'Dim nbTargets = Me.targetArray.GetLength(1)
                TestAllSamples(Me.inputArray) ', nbOutputs:=nbTargets)
            Else
                SetOuput1D()
                ComputeAverageError()
            End If
            PrintSuccess(iteration)
        End If

    End Sub

End Class