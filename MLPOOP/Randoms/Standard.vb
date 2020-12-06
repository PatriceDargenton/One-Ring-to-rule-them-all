
Imports Perceptron.Utilities

Namespace Randoms
    Public Class Standard : Inherits BaseRandom

        Private m_random As Random

        Public Sub New(range As Range, seed%)
            MyBase.New(range)
            Me.m_random = New Random(seed)
        End Sub

        Public Overrides Function Generate#()
            Dim r = Me.m_random.NextDouble() * Me.Range.Delta + Me.Range.Minimum
            Dim rounded# = Math.Round(r, clsMLPGeneric.nbRoundingDigits) ' 28/11/2020
            Return rounded
        End Function

    End Class
End Namespace