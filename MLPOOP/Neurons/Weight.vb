
Namespace Neurons
    Public Class Weight

        Public Property Value#
        Public Property Previous#

        Public Property Child As Neuron
        Public Property Parent As Neuron

        Public Sub New(value#, parent_node As Neuron, child_node As Neuron)
            Me.Previous = 0
            Me.Value = value
            Me.Child = child_node
            Me.Parent = parent_node
        End Sub

    End Class
End Namespace