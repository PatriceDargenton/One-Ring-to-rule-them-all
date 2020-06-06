
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

        Public Sub Step_()
            For Each parameter In Parameters
                parameter.Data -= parameter.Gradient.Data * Me.LearningRate
                If Me.WeightAdjustment <> 0 Then _
                    parameter.Gradient.Data *= Me.WeightAdjustment
            Next
        End Sub

    End Class

End Namespace