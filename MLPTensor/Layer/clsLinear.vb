﻿
Imports Perceptron.Utility ' Matrix

Namespace DLFramework.Layers

    Public Class Linear : Inherits Layer

        Private m_addBias As Boolean = False

        Public Sub New(input%, output%, w As Matrix, addBias As Boolean)
            Dim weights = New Tensor(w, autoGrad:=True)
            Parameters.Add(weights)
            Me.m_addBias = addBias
        End Sub

        Public Sub New(input%, output%, w As Matrix, bias0 As Matrix)
            Dim weights = New Tensor(w, autoGrad:=True)
            Dim bias = New Tensor(bias0, autoGrad:=True)
            Parameters.Add(weights)
            Parameters.Add(bias)
            Me.m_addBias = True
        End Sub

        Public Sub New(input%, output%, addBias As Boolean)
            Dim w = Matrix.Zeros(input, output) - 1
            Dim weights = New Tensor(w, autoGrad:=True)
            Parameters.Add(weights)
            Me.m_addBias = addBias
            If Not Me.m_addBias Then Exit Sub
            Dim bias = New Tensor(Matrix.Zeros(1, output), autoGrad:=True)
            Parameters.Add(bias)
        End Sub

        Public Sub New(input%, output%, r As Random, addBias As Boolean)
            Dim w = (Matrix.Randomize(input, output, r) * 2) - 1
            Dim weights = New Tensor(w, autoGrad:=True)
            Parameters.Add(weights)
            Me.m_addBias = addBias
            If Not Me.m_addBias Then Exit Sub
            Dim bias = New Tensor(Matrix.Zeros(1, output), autoGrad:=True)
            Parameters.Add(bias)
        End Sub

        Public Overrides Function Forward(input As Tensor) As Tensor
            If Me.m_addBias Then
                Dim bias = Tensor.Expand(Parameters(1), AxisZero.vertical, input.Data.r)
                Dim tnsor = Tensor.Add(Tensor.MatMult(input, Parameters(0)), bias)
                Return tnsor
            Else
                Return input
            End If
        End Function

    End Class

End Namespace
