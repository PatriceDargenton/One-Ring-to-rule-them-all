Imports Perceptron.Utilities

Namespace Activation

    Public Class HyperbolicTangent
        Inherits BaseActivation

        Public Property Alpha As Double

        Public Sub New()
            MyBase.New()
            Me.Alpha = 1
            Me.in_range = New Range(Double.NegativeInfinity, Double.PositiveInfinity)
            Me.out_range = New Range(-1, 1)
        End Sub

        Public Sub New(Alpha As Double, center As Double)
            Me.Alpha = Alpha
            Me.Center = center
            Me.in_range = New Range(Double.NegativeInfinity, Double.PositiveInfinity)
            Me.out_range = New Range(-1, 1)
        End Sub

        Public Overrides Function AbstractedDerivative(value As Double) As Double
            Dim xc# = value - Me.Center
            Return Me.Alpha * (1 - xc * xc)
        End Function

        Public Overrides Function Derivative(value As Double) As Double
            Dim xc# = value - Me.Center
            Return Me.Alpha * (1 - Math.Tanh(Me.Alpha * xc) ^ 2)
        End Function

        Public Overrides Function Evaluate(value As Double) As Double
            Dim xc# = value - Me.Center
            Return Math.Tanh(Me.Alpha * xc)
        End Function

    End Class
End Namespace
