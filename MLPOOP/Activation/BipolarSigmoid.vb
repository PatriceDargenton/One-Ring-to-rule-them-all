Imports Perceptron.Utilities

Namespace Activation
    Public Class BipolarSigmoid ' BipolarSigmoid (alpha x) = HyperbolicTangent(-alpha x / 2)
        Inherits BaseActivation

        Public Property Alpha As Double

        Public Sub New()
            Me.Alpha = 1
            Me.in_range = New Range(Double.NegativeInfinity, Double.PositiveInfinity)
            Me.out_range = New Range(-1, 1)
        End Sub

        Public Sub New(Alpha As Double)
            Me.Alpha = Alpha
            Me.in_range = New Range(Double.NegativeInfinity, Double.PositiveInfinity)
            Me.out_range = New Range(-1, 1)
        End Sub

        Public Overrides Function AbstractedDerivative(value As Double) As Double
            Throw New NotImplementedException
        End Function

        Public Overrides Function Derivative(value As Double) As Double
            Dim exp = Math.Exp(Me.Alpha * value)
            Return 2 * (Me.Alpha * exp) / ((exp + 1) * (exp + 1))
        End Function

        Public Overrides Function Evaluate(value As Double) As Double
            Return 2 / (1 + Math.Exp(-Me.Alpha * value)) - 1
        End Function

    End Class
End Namespace
