
Imports Perceptron.Utility ' AxisZero

Namespace DLFramework.Layers.Loss

    Public Class MeanSquaredError : Inherits Layer

        Public Overrides Function Forward(prediction As Tensor, target As Tensor) As Tensor
            Dim diff = Tensor.Substract(prediction, target)
            Dim tnsor = Tensor.Sum(Tensor.Mul(diff, diff), AxisZero.vertical)
            Return tnsor
        End Function

        'Public Overrides Function Forward(prediction As Tensor, target As Tensor,
        '    useBias As Boolean) As Tensor

        '    If Not useBias Then
        '        ' Cut first column?
        '        'delta(i) = delta(i).Slice(0, 1, delta(i).x, delta(i).y)
        '    End If

        '    Dim diff = Tensor.Substract(prediction, target)
        '    Return Tensor.Sum(Tensor.Mul(diff, diff), AxisZero.vertical)
        'End Function

    End Class

End Namespace