
Namespace DLFramework.Layers

    Public Class Layer

        Protected m_parameters As List(Of Tensor)

        Public Overridable Property Parameters As List(Of Tensor)
            Get
                Return m_parameters
            End Get
            Set(value As List(Of Tensor))
                m_parameters = value
            End Set
        End Property

        Public Sub New()
            m_parameters = New List(Of Tensor)()
        End Sub

        Public Overridable Function Forward(input As Tensor) As Tensor
            Return Nothing
        End Function

        Public Overridable Function Forward(input As Tensor, target As Tensor) As Tensor
            Return Nothing
        End Function

        'Public Overridable Function Forward(input As Tensor, target As Tensor,
        '    useBias As Boolean) As Tensor
        '    Return Nothing
        'End Function

    End Class

End Namespace
