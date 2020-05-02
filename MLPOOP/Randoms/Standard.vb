Imports Perceptron.Utilities

Namespace Randoms
    Public Class Standard
        Inherits BaseRandom

        Private Random As Random

        Public Sub New(range As Range, seed As Integer)
            MyBase.New(range)
            Me.Random = New Random(seed)
        End Sub

        Public Overrides Function Generate() As Double
            Return Me.Random.NextDouble() * Me.Range.Delta + Me.Range.Minimum
        End Function

    End Class
End Namespace