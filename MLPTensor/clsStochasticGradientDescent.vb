
Imports System.Text
Imports Perceptron.Utility

Namespace DLFramework.Optimizers

    Public Class StochasticGradientDescent

        Private m_weightAdjustment! ' (alpha coefficient)
        Private m_learningRate!
        Private m_parameters As List(Of Tensor)

        Public Property Parameters As List(Of Tensor)
            Get
                Return m_parameters
            End Get
            Set(value As List(Of Tensor))
                m_parameters = value
            End Set
        End Property

        Public ReadOnly Property LearningRate!
            Get
                Return m_learningRate
            End Get
        End Property

        Public ReadOnly Property WeightAdjustment!
            Get
                Return m_weightAdjustment
            End Get
        End Property

        Public Sub New(parameters As List(Of Tensor),
            Optional learningRate! = 0.1!, Optional weightAdjustment! = 0)
            Me.m_parameters = parameters
            Me.m_learningRate = learningRate
            Me.m_weightAdjustment = weightAdjustment
        End Sub

        Public Sub Step_(zero As Boolean)
            For Each parameter In Parameters
                Dim m As Matrix = parameter.Gradient.Data * Me.LearningRate
                parameter.Data -= m
                If zero OrElse Me.WeightAdjustment <> 0 Then _
                    parameter.Gradient.Data *= Me.WeightAdjustment
            Next
        End Sub

        Public Function ParametersToString$()
            Dim sb As New StringBuilder()
            Dim numPrm% = 0
            For Each parameter In Parameters
                numPrm += 1
                sb.AppendLine("prm n°" & numPrm & "=" & parameter.ToString())
            Next
            Return sb.ToString
        End Function

    End Class

End Namespace