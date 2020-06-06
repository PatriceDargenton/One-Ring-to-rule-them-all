
Imports Perceptron.Utilities

Namespace Activation
    Public Class Linear : Inherits BaseActivation

        Public Property Slope#

        Public Sub New()
            Me.Slope = 1
            Me.in_range = New Range(Double.NegativeInfinity, Double.PositiveInfinity)
            Me.out_range = New Range(Double.NegativeInfinity, Double.PositiveInfinity)
        End Sub

        Public Sub New(slope#)
            Me.Slope = slope
            Me.in_range = New Range(Double.NegativeInfinity, Double.PositiveInfinity)
            Me.out_range = New Range(Double.NegativeInfinity, Double.PositiveInfinity)
        End Sub

        Public Sub New(slope%)
            Me.Slope = slope
        End Sub

        Public Overrides Function AbstractedDerivative#(value#)
            Return Me.Slope
        End Function

        Public Overrides Function Derivative#(value#)
            Return Me.Slope
        End Function

        Public Overrides Function Evaluate#(value#)
            Return Me.Slope * value
        End Function

    End Class
End Namespace