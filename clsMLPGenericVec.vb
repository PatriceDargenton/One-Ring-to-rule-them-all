
Imports Perceptron.Utility ' Matrix

Public MustInherit Class clsVectorizedMLPGeneric : Inherits clsMLPGeneric

    Public vectorizedLearningMode As Boolean = True
    Public exampleCount%

    Protected neuronCount%()

    ''' <summary>
    ''' Train all samples at once (run epoch for one iteration: all samples ordered in one vector)
    ''' </summary>
    Public MustOverride Sub TrainVector()

    Public Overrides Sub TrainSystematic(inputs!(,), targets!(,),
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

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

    Public Overrides Sub PrintOutput(iteration%)

        If ShowThisIteration(iteration) Then
            If Not Me.vectorizedLearningMode Then
                Dim nbTargets = Me.targetArray.GetLength(1)
                TestAllSamples(Me.inputArray, nbOutputs:=nbTargets)
            End If
            SetOuput1D()
            ComputeAverageError()
            PrintSuccess(iteration)
        End If

    End Sub

End Class