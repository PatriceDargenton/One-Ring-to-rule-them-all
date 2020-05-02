Imports Perceptron.Utilities

Namespace Activation

    Public Class Linear
        Inherits BaseActivation

        Public Property Slope As Double

        Public Sub New()
            Me.Slope = 1
            Me.in_range = New Range(Double.NegativeInfinity, Double.PositiveInfinity)
            Me.out_range = New Range(Double.NegativeInfinity, Double.PositiveInfinity)
        End Sub

        Public Sub New(slope As Double)
            Me.Slope = slope
            Me.in_range = New Range(Double.NegativeInfinity, Double.PositiveInfinity)
            Me.out_range = New Range(Double.NegativeInfinity, Double.PositiveInfinity)
        End Sub

        Public Sub New(slope As Integer)
            Me.Slope = slope
        End Sub

        Public Overrides Function AbstractedDerivative(value As Double) As Double
            Return Me.Slope
        End Function

        Public Overrides Function Derivative(value As Double) As Double
            Return Me.Slope
        End Function

        Public Overrides Function Evaluate(value As Double) As Double
            Return Me.Slope * value
        End Function

    End Class
End Namespace
