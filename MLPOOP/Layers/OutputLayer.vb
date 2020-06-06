
Imports Perceptron.Activation
Imports Perceptron.Neurons

Namespace Layers
    Public Class OutputLayer : Inherits BaseLayer

        Public Sub New(Size%, Activation As BaseActivation)
            MyBase.New(Size, Activation)
            For x = 1 To Size
                Me.Neurons.Add(New Neuron(NeuronType.Output))
            Next
        End Sub

        Public Sub AssignErrors(expected As List(Of Double))
            For x = 0 To Me.Size - 1
                Me.Neurons(x).ErrorDelta = expected(x) - Me.Neurons(x).Output
            Next
        End Sub

        Public Function ExtractOutputs() As List(Of Double)
            Dim results = New List(Of Double)
            For Each n In Me.Neurons
                results.Add(n.Output)
            Next
            Return results
        End Function

        Public Function CalculateSquaredError#()
            Dim sum# = 0.0
            For Each n In Me.Neurons
                sum += n.ErrorDelta * n.ErrorDelta
            Next
            Return sum / 2
        End Function

    End Class
End Namespace