
Imports Perceptron.Utilities

Namespace Activation
    Public Class AdjustableSigmoid : Inherits BaseActivation

        Public Property Alpha#

        Public Sub New()
            Me.Alpha = 1
            Me.in_range = New Range(Double.NegativeInfinity, Double.PositiveInfinity)
            Me.out_range = New Range(-1, 1)
        End Sub

        Public Sub New(alpha#, outputRange As Range)
            Me.Alpha = alpha
            Me.in_range = New Range(Double.NegativeInfinity, Double.PositiveInfinity)
            Me.out_range = Me.OutputRange
        End Sub

        Public Overrides Function AbstractedDerivative#(value#)
            Throw New NotImplementedException
        End Function

        Public Overrides Function Derivative#(value#)
            Dim exp = Math.Exp(Me.Alpha * value)
            Return Me.OutputRange.Delta * (Me.Alpha * exp) / ((exp + 1) * (exp + 1))
        End Function

        Public Overrides Function Evaluate#(value#)
            Return Me.OutputRange.Delta / (1 + Math.Exp(-Me.Alpha * value)) + Me.OutputRange.Minimum
        End Function

    End Class
End Namespace