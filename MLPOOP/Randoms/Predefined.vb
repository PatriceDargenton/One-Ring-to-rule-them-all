Imports Perceptron.Utilities

Namespace Randoms
    Public Class Predefined
        Inherits BaseRandom

        Private Values As List(Of Double)

        Public Sub New(range As Range, values As Double())
            MyBase.New(range)
            Me.Values = New List(Of Double)
            Me.Values.AddRange(values)
        End Sub

        Public Overrides Function Generate() As Double
            Dim result = Me.Values(0)
            Me.Values.RemoveAt(0)
            Return result
        End Function

    End Class
End Namespace