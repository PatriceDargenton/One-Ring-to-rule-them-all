
Imports Perceptron.Utility ' Matrix

Namespace DLFramework.Layers.Activation

    Public Class Sigmoid

        Public Shared Function Forward(input As Tensor, center!) As Tensor

            Dim output#(,) = input.Data
            Matrix.MatrixLoop(
                Sub(i, j)
                    Dim x# = output(i, j)
                    Dim xc# = x - center
                    output(i, j) = 1 / (1 + Math.Exp(-xc))
                End Sub, input.Data.r, input.Data.c)

            If input.AutoGrad Then
                Dim Creators = New List(Of Tensor)() From {input}
                Return New Tensor(data:=output, autoGrad:=True, creators:=Creators,
                    creationOperation:=TensorOperations.Other, arguments:=Nothing,
                    backwardCallback:=AddressOf Sigmoid.Backward)
            End If

            Return New Tensor(output)

        End Function

        Public Shared Sub Backward(self As Tensor, gradient As Tensor,
            creators As List(Of Tensor))

            Dim ones = New Tensor(Matrix.Ones(gradient.Data.r, gradient.Data.c))
            Dim derivative#(,) = self.Data
            Matrix.MatrixLoop(
                Sub(i, j)
                    derivative(i, j) = derivative(i, j) * (1.0 - derivative(i, j))
                End Sub, self.Data.r, self.Data.c)
            Dim derivatives = New Tensor(derivative)
            creators(0).Backward(Tensor.Mul(gradient, derivatives))

        End Sub

    End Class

    Public Class HyperbolicTangent

        Public Shared Function Forward(input As Tensor, center!) As Tensor

            Dim output#(,) = input.Data
            Matrix.MatrixLoop(
                Sub(i, j)
                    Dim x# = output(i, j)
                    Dim xc# = x - center
                    output(i, j) = Math.Tanh(xc)
                End Sub, input.Data.r, input.Data.c)

            If input.AutoGrad Then
                Dim Creators = New List(Of Tensor)() From {input}
                Return New Tensor(data:=output, autoGrad:=True, creators:=Creators,
                    creationOperation:=TensorOperations.Other, arguments:=Nothing,
                    backwardCallback:=AddressOf HyperbolicTangent.Backward)
            End If

            Return New Tensor(output)

        End Function

        Public Shared Sub Backward(self As Tensor, gradient As Tensor,
            creators As List(Of Tensor))

            Dim ones = New Tensor(Matrix.Ones(gradient.Data.r, gradient.Data.c))
            Dim derivative#(,) = self.Data
            Matrix.MatrixLoop(
                Sub(i, j)
                    derivative(i, j) = 1.0 - derivative(i, j) * derivative(i, j)
                End Sub, self.Data.r, self.Data.c)
            Dim derivatives = New Tensor(derivative)
            creators(0).Backward(Tensor.Mul(gradient, derivatives))

        End Sub

    End Class

    Public Class ELU ' Exponential Linear Units

        Const gain# = 1

        Public Shared Function Forward(input As Tensor, center!) As Tensor

            Dim output#(,) = input.Data
            Matrix.MatrixLoop(
                Sub(i, j)
                    Dim x# = output(i, j)
                    Dim xc# = x - center
                    Dim y#
                    If xc >= 0 Then
                        y = xc
                    Else
                        y = gain * (Math.Exp(xc) - 1)
                    End If
                    output(i, j) = y
                End Sub, input.Data.r, input.Data.c)

            If input.AutoGrad Then
                Dim Creators = New List(Of Tensor)() From {input}
                Return New Tensor(data:=output, autoGrad:=True, creators:=Creators,
                    creationOperation:=TensorOperations.Other, arguments:=Nothing,
                    backwardCallback:=AddressOf ELU.Backward)
            End If

            Return New Tensor(output)

        End Function

        Public Shared Sub Backward(self As Tensor, gradient As Tensor,
            creators As List(Of Tensor))

            Dim ones = New Tensor(Matrix.Ones(gradient.Data.r, gradient.Data.c))
            Dim derivative#(,) = self.Data
            Matrix.MatrixLoop(
                Sub(i, j)
                    Dim y# = 0
                    If gain > 0 Then
                        Dim fx# = derivative(i, j)
                        If fx >= 0 Then
                            y = 1
                        Else
                            y = fx + gain
                        End If
                    End If
                    derivative(i, j) = y
                End Sub, self.Data.r, self.Data.c)
            Dim derivatives = New Tensor(derivative)
            creators(0).Backward(Tensor.Mul(gradient, derivatives))

        End Sub

    End Class

End Namespace