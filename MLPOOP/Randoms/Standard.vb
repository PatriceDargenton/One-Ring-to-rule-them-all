
Imports Perceptron.Utilities

Namespace Randoms
    Public Class Standard : Inherits BaseRandom

        Private Random As Random

        Public Sub New(range As Range, seed%)
            MyBase.New(range)
            Me.Random = New Random(seed)
        End Sub

        Public Overrides Function Generate#()
            Return Me.Random.NextDouble() * Me.Range.Delta + Me.Range.Minimum
        End Function

    End Class
End Namespace