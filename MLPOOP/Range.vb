Namespace Utilities
    Public Class Range

        Public Property Minimum As Double
        Public Property Maximum As Double

        Public ReadOnly Property Delta As Double
            Get
                Return Me.Maximum - Me.Minimum
            End Get
        End Property

        Public Sub New(min As Double, max As Double)
            Me.Minimum = min
            Me.Maximum = max
        End Sub

    End Class
End Namespace