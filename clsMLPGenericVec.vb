
Public MustInherit Class clsVectorizedMLPGeneric : Inherits clsMLPGeneric

    Public vectorizedLearningMode As Boolean = True

    Public exampleCount%
    Public minBatchSize% = 1

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
        Me.learningMode = enumLearningMode.VectorialBatch
        Me.vectorizedLearningMode = True
        For iteration = 0 To nbIterationsBatch - 1
            TrainVectorOneIteration()
        Next
        SetOuput1D()

    End Sub

    ''' <summary>
    ''' Train all samples in batch learning mode
    ''' </summary>
    Public Overridable Sub TrainVectorBatch()

        Me.learningMode = enumLearningMode.VectorialBatch
        Me.vectorizedLearningMode = True

        If Not Me.printOutput_ Then

            Dim nbIterationsBatch0 = Me.nbIterationsBatch
            Dim nbIterations0 = CInt(Me.nbIterations / nbIterationsBatch0)
            If Me.nbIterations < nbIterationsBatch0 Then nbIterations0 = 1
            Dim iteration = 0
            Dim iterationTot = 0
            Do While iteration < nbIterations0
                If iterationTot + nbIterationsBatch0 > Me.nbIterations Then
                    nbIterationsBatch0 = Me.nbIterations - iterationTot
                End If
                If nbIterationsBatch0 <= 0 Then Exit Do

                Me.numIteration += nbIterationsBatch0

                TrainVectorBatch(nbIterationsBatch0)

                ' Debug one batch of iterations:
                'Debug.WriteLine(iterationTot & "/" & Me.nbIterations)
                'SetOuput1D()
                'ComputeAverageError()
                'PrintOutput(iterationTot, force:=True)

                iteration += 1
                iterationTot += nbIterationsBatch0

            Loop
            'Debug.WriteLine(iterationTot & "/" & Me.nbIterations)
        Else

            Dim minBatchSizeFound = False
            Dim iteration = 0
            Dim iterationTot = 0
            Do While iteration < Me.nbIterations

                Dim nbIterationsBatch0%
                If iteration < 10 - 1 Then
                    nbIterationsBatch0 = 1
                ElseIf iteration < 100 - 1 Then
                    nbIterationsBatch0 = 10
                ElseIf iteration < 1000 - 1 Then
                    nbIterationsBatch0 = 100
                Else
                    nbIterationsBatch0 = 1000
                End If

                If nbIterationsBatch0 < Me.minBatchSize Then
                    nbIterationsBatch0 = Me.minBatchSize
                    minBatchSizeFound = True
                End If

                If iteration + nbIterationsBatch0 > Me.nbIterations Then
                    nbIterationsBatch0 = Me.nbIterations - iteration
                End If
                If nbIterationsBatch0 > Me.nbIterationsBatch Then
                    nbIterationsBatch0 = Me.nbIterationsBatch
                End If
                If nbIterationsBatch0 <= 0 Then Exit Do

                Me.numIteration += nbIterationsBatch0
                TrainVectorBatch(nbIterationsBatch0)

                If minBatchSizeFound AndAlso iteration > 0 Then
                    PrintOutput(iteration - 1)
                Else
                    PrintOutput(iteration)
                End If
                If iteration + nbIterationsBatch0 >= Me.nbIterations Then
                    iteration = Me.nbIterations - 1
                    Exit Do
                End If
                iteration += nbIterationsBatch0
                iterationTot += nbIterationsBatch0

            Loop

            If minBatchSizeFound OrElse Not ShowThisIteration(iteration) Then _
                PrintOutput(iteration, force:=True)

        End If

        SetOuput1D()
        ComputeAverageError()

    End Sub

    Public Overrides Sub TrainSystematic(inputs!(,), targets!(,),
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Me.learningMode = learningMode
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
        Me.learningMode = enumLearningMode.Stochastic
        Me.vectorizedLearningMode = False
        Me.exampleCount = 1
        MyBase.TrainStochastic(inputs, targets)
    End Sub

    Public Overrides Sub TrainSemiStochastic(inputs!(,), targets!(,))
        Me.learningMode = enumLearningMode.SemiStochastic
        Me.vectorizedLearningMode = False
        Me.exampleCount = 1
        MyBase.TrainSemiStochastic(inputs, targets)
    End Sub

    Public Overridable Sub SetOuput1D()
    End Sub

    Public Overrides Sub PrintOutput(iteration%, Optional force As Boolean = False)

        If force OrElse ShowThisIteration(iteration) Then
            If Not Me.vectorizedLearningMode Then
                TestAllSamples(Me.inputArray)
            Else
                SetOuput1D()
                ComputeAverageError()
            End If
            PrintSuccess(iteration)
        End If

    End Sub

End Class