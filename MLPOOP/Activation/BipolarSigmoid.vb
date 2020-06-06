
Imports Perceptron.Utilities

Namespace Activation
    ' BipolarSigmoid (alpha x) = HyperbolicTangent(-alpha x / 2)
    Public Class BipolarSigmoid : Inherits BaseActivation

        Public Property Alpha#

        Public Sub New()
            Me.Alpha = 1
            Me.in_range = New Range(Double.NegativeInfinity, Double.PositiveInfinity)
            Me.out_range = New Range(-1, 1)
        End Sub

        Public Sub New(alpha#)
            Me.Alpha = alpha
            Me.in_range = New Range(Double.NegativeInfinity, Double.PositiveInfinity)
            Me.out_range = New Range(-1, 1)
        End Sub

        Public Overrides Function AbstractedDerivative#(value#)
            Throw New NotImplementedException
        End Function

        Public Overrides Function Derivative#(value#)
            Dim exp = Math.Exp(Me.Alpha * value)
            Return 2 * (Me.Alpha * exp) / ((exp + 1) * (exp + 1))
        End Function

        Public Overrides Function Evaluate#(value#)
            Return 2 / (1 + Math.Exp(-Me.Alpha * value)) - 1
        End Function

    End Class
End Namespace