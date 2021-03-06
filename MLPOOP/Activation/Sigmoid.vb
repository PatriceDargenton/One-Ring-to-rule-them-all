﻿
Imports Perceptron.Utilities

Namespace Activation
    Public Class Sigmoid : Inherits BaseActivation

        Public Property Alpha#

        Public Sub New()
            MyBase.New()
            Me.Alpha = 1
            Me.in_range = New Range(Double.NegativeInfinity, Double.PositiveInfinity)
            Me.out_range = New Range(0, 1)
        End Sub

        Public Sub New(alpha#, Optional center# = 0)
            Me.Alpha = alpha
            Me.Center = center
            Me.in_range = New Range(Double.NegativeInfinity, Double.PositiveInfinity)
            Me.out_range = New Range(0, 1)
        End Sub

        Public Overrides Function AbstractedDerivative#(value#)
            Dim xc# = value - Me.Center
            Return Me.Alpha * xc * (1 - xc)
        End Function

        Public Overrides Function Derivative#(value#)
            Dim xc# = value - Me.Center
            Dim exp = Math.Exp(Me.Alpha * xc)
            'Dim exp2 = Math.Exp(-Me.Alpha * xc) ' Quasi-same value
            Return (Me.Alpha * exp) / ((exp + 1) * (exp + 1))
        End Function

        Public Overrides Function Evaluate#(value#)
            Dim xc# = value - Me.Center
            Return 1 / (1 + Math.Exp(-Me.Alpha * xc))
        End Function

    End Class
End Namespace