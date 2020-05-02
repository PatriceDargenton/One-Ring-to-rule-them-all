Namespace Neurons
    Public Class Neuron

        Public Property NumericalFormat As String = "0.000"
        Public Property Input As Double
        Public Property Output As Double
        Public Property ErrorDelta As Double
        Public Property Primed As Double

        Public Property Type As NeuronType
        Public Property WeightsToChild As List(Of Weight)
        Public Property WeightsToParent As List(Of Weight)
        Public Property WeightToBias As Weight

        Public Sub New(type As NeuronType)
            Me.Input = 0
            Me.Output = 0
            Me.Primed = 0
            Me.ErrorDelta = 0
            Me.Type = type
            Select Case type
                Case NeuronType.Input
                    Me.WeightsToChild = New List(Of Weight)
                Case NeuronType.Hidden
                    Me.WeightsToChild = New List(Of Weight)
                    Me.WeightsToParent = New List(Of Weight)
                Case NeuronType.Output
                    Me.WeightsToParent = New List(Of Weight)
            End Select
        End Sub

        Public Overrides Function ToString() As String
            Dim result = "Input = " & Me.Input.ToString(NumericalFormat) & vbCr
            result &= "Output = " & Me.Output.ToString(NumericalFormat) & vbCr
            result &= "Error = " & Me.ErrorDelta.ToString(NumericalFormat)
            Return result
        End Function

    End Class
End Namespace