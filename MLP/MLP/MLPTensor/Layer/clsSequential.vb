
Imports System.Collections.Generic
Imports System.Text

Namespace DLFramework.Layers

    Public Class Sequential : Inherits Layer

        Private m_layers As List(Of Layer)

        Public ReadOnly Property Layers As List(Of Layer)
            Get
                Return m_layers
            End Get
        End Property

        Public Overrides Property Parameters As List(Of Tensor)
            Get
                Return GetParameters()
            End Get
            Set(value As List(Of Tensor))
                m_parameters = value
            End Set
        End Property

        Public Sub New(layers As List(Of Layer))
            Me.m_layers = layers
        End Sub

        Public Sub New()
            Me.m_layers = New List(Of Layer)()
        End Sub

        Public Overrides Function Forward(input As Tensor) As Tensor
            'Dim numLayer% = 0
            For Each layer In Me.m_layers
                input = layer.Forward(input)
                'numLayer += 1
                'Debug.WriteLine("Layer n°" & numLayer & ": " & input.ToString())
            Next
            Return input
        End Function

        Public Function GetParameters() As List(Of Tensor)
            Dim temp As List(Of Tensor) = New List(Of Tensor)()
            For Each layer In Me.m_layers
                temp.AddRange(layer.Parameters)
            Next
            Return temp
        End Function

        Public Function ParametersToString$()
            Dim sb As New StringBuilder()
            For Each parameter In Parameters
                sb.Append(parameter.ToString())
            Next
            Return sb.ToString
        End Function

    End Class

End Namespace