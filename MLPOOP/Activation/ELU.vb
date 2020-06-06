
Imports Perceptron.Utilities

Namespace Activation
    Public Class ELU : Inherits BaseActivation

        Public Property Alpha#

        Public Sub New()
            MyBase.New()
            Me.Alpha = 1
            Me.in_range = New Range(Double.NegativeInfinity, Double.PositiveInfinity)
            Me.out_range = New Range(0, 1)
        End Sub

        Public Sub New(alpha#, center#)
            Me.Alpha = alpha
            Me.Center = center
            Me.in_range = New Range(Double.NegativeInfinity, Double.PositiveInfinity)
            Me.out_range = New Range(0, 1)
        End Sub

        Public Overrides Function AbstractedDerivative#(value#)

            ' If gain < 0 the derivate is undefined
            If Me.Alpha < 0 Then Return 0

            Dim y#
            If value >= 0 Then
                y = 1
            Else
                y = value + Me.Alpha
            End If
            Return y

        End Function

        Public Overrides Function Derivative#(value#)

            ' If gain < 0 the derivate is undefined
            If Me.Alpha < 0 Then Return 0

            Dim xc# = value - Me.Center
            Dim y#
            If xc >= 0 Then
                y = 1
            Else
                Dim fx# = Evaluate(value)
                y = fx + Me.Alpha
            End If
            Return y

        End Function

        Public Overrides Function Evaluate#(value#)
            Dim xc# = value - Me.Center
            Dim y#
            If xc >= 0 Then
                y = xc
            Else
                y = Me.Alpha * (Math.Exp(xc) - 1)
            End If
            Return y
        End Function

    End Class
End Namespace