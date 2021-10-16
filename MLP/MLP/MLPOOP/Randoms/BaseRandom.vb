
Imports Perceptron.Utilities

Namespace Randoms
    Public MustInherit Class BaseRandom

        Public Property Range As Range

        Public Sub New(range As Range)
            Me.Range = range
        End Sub

        Public MustOverride Function Generate#()

    End Class
End Namespace