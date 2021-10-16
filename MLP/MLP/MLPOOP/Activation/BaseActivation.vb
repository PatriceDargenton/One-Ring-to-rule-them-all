
Imports Perceptron.Utilities

Namespace Activation
    Public MustInherit Class BaseActivation

        Protected in_range As Range
        Protected out_range As Range

        Public Property Center#

        Public ReadOnly Property InputRange As Range
            Get
                Return Me.in_range
            End Get
        End Property

        Public ReadOnly Property OutputRange As Range
            Get
                Return Me.out_range
            End Get
        End Property

        Public MustOverride Function Evaluate#(value#)
        Public MustOverride Function Derivative#(value#)
        Public MustOverride Function AbstractedDerivative#(value#)

        Public Sub New()
            Me.Center = 0
        End Sub
        Public Sub New(center!)
            Me.Center = center
        End Sub

    End Class
End Namespace