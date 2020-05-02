Imports Perceptron.Activation
Imports Perceptron.Neurons

Namespace Layers
    Public Class InputLayer
        Inherits BaseLayer

        Public Sub New(Size As Integer, Activation As BaseActivation)
            MyBase.New(Size, Activation)
            For x = 1 To Size
                Me.Neurons.Add(New Neuron(NeuronType.Input))
            Next
        End Sub

        Public Sub SetInput(input As List(Of Double))
            For x = 0 To Me.Size - 1
                Me.Neurons(x).Input = input(x)
                Me.Neurons(x).Output = input(x)
            Next
        End Sub

    End Class
End Namespace
