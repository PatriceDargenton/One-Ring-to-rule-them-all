
Imports Perceptron.Utility ' Matrix

Public MustInherit Class clsVectorizedMLPGeneric : Inherits clsMLPGeneric

    Public vectorizedLearningMode As Boolean = True
    Public exampleCount%

    Protected neuronCount%()

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

    Public Overrides Sub TestOneSample(input!(), ByRef ouput!())
        TestOneSample(input)
        ouput = Me.lastOutputArray1DSingle
    End Sub

End Class