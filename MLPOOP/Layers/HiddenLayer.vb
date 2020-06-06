
Imports Perceptron.Activation
Imports Perceptron.Neurons

Namespace Layers
    Public Class HiddenLayer : Inherits BaseLayer

        Public Sub New(Size%, Activation As BaseActivation)
            MyBase.New(Size, Activation)
            For x = 1 To Size
                Me.Neurons.Add(New Neuron(NeuronType.Hidden))
            Next
        End Sub

    End Class
End Namespace

