
' From: https://github.com/nlabiris/perceptrons : C# -> VB .NET conversion

Class ML_TrainingData

    Public data!(,)

    Private inputsLength%

    Private targetsLength%

    Public Sub New(inputsLength%, targetsLength%)
        Me.inputsLength = inputsLength
        Me.targetsLength = targetsLength
    End Sub

    Public Sub Create()

        ' XOR sample
        Me.data = New Single(,) {
            {1.0!, 0.0!, 1.0!},
            {0.0!, 0.0!, 0.0!},
            {0.0!, 1.0!, 1.0!},
            {1.0!, 1.0!, 0.0!}}

    End Sub

    Public Function GetInputs() As Single(,)

        Dim rows = Me.data.GetLength(0)
        Dim cols = Me.inputsLength
        Dim inp!(rows - 1, cols - 1)

        For i = 0 To rows - 1
            For j = 0 To cols - 1
                inp(i, j) = Me.data(i, j)
            Next
        Next
        Return inp

    End Function

    Public Function GetOutputs() As Single(,)

        Dim rows = Me.data.GetLength(0) ' 4
        Dim cols = Me.data.GetLength(1) ' 3
        Dim tgts = Me.targetsLength ' 1
        Dim tgt!(rows - 1, tgts - 1)

        For i = 0 To rows - 1
            Dim k = 0
            For j = cols - tgts To cols - 1
                tgt(i, k) = Me.data(i, j)
            Next
        Next

        Return tgt

    End Function

End Class