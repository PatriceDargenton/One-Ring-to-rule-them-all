Namespace Neurons
    Public Class Weight

        Public Property Value As Double
        Public Property Previous As Double

        Public Property Child As Neuron
        Public Property Parent As Neuron

        Public Sub New(value As Double, parent_node As Neuron, child_node As Neuron)
            Me.Previous = 0
            Me.Value = value
            Me.Child = child_node
            Me.Parent = parent_node
        End Sub

    End Class
End Namespace