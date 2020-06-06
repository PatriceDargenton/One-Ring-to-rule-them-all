
Namespace Utilities
    Public Class Range

        Public Property Minimum#
        Public Property Maximum#

        Public ReadOnly Property Delta#
            Get
                Return Me.Maximum - Me.Minimum
            End Get
        End Property

        Public Sub New(min#, max#)
            Me.Minimum = min
            Me.Maximum = max
        End Sub

    End Class
End Namespace