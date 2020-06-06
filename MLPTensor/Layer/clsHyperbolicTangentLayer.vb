
Imports Perceptron.DLFramework.Layers.Activation

Namespace DLFramework.Layers

    Public Class HyperbolicTangentLayer : Inherits Layer

        Private m_center!

        Public ReadOnly Property Center!
            Get
                Return m_center
            End Get
        End Property

        Public Sub New(center!)
            Me.m_center = center
        End Sub

        Public Overrides Function Forward(input As Tensor) As Tensor
            Return HyperbolicTangent.Forward(input, Me.Center)
        End Function

    End Class

End Namespace
