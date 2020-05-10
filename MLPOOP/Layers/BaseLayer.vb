
Imports Perceptron.Activation
Imports Perceptron.Neurons
Imports Perceptron.Randoms

Namespace Layers

    Public Class BaseLayer

        Public Property Size As Integer
        Public Property Neurons As List(Of Neuron)
        Public Property ActivationFunction As BaseActivation

        Public Sub New(Size As Integer, Activation As BaseActivation)
            Me.Size = Size
            Me.Neurons = New List(Of Neuron)
            Me.ActivationFunction = Activation
        End Sub

        Public Sub Init()
            For Each n As Neuron In Me.Neurons
                n.WeightsToChild = New List(Of Weight)
                n.WeightsToParent = New List(Of Weight)
                n.WeightToBias = Nothing
            Next
        End Sub

        Public Sub ConnectParent(layer As BaseLayer, Random As BaseRandom)
            For Each n2 As Neuron In Me.Neurons
                For Each n As Neuron In layer.Neurons
                    Dim weight = New Weight(Random.Generate(), n, n2)
                    n.WeightsToChild.Add(weight)
                    n2.WeightsToParent.Add(weight)
                Next
            Next
        End Sub

        Public Sub ConnectChild(layer As BaseLayer, Random As BaseRandom)
            For Each n2 As Neuron In Me.Neurons
                For Each n As Neuron In layer.Neurons
                    Dim weight = New Weight(Random.Generate(), n2, n)
                    n.WeightsToParent.Add(weight)
                    n2.WeightsToChild.Add(weight)
                Next
            Next
        End Sub

        Public Sub InitChild(layer As BaseLayer, Random As BaseRandom)
            Dim i% = 0
            For Each n2 As Neuron In Me.Neurons
                Dim j% = 0
                For Each n As Neuron In layer.Neurons
                    Dim weight = New Weight(Random.Generate(), n2, n)
                    n.WeightsToParent(i) = weight
                    n2.WeightsToChild(j) = weight
                    j += 1
                Next
                i += 1
            Next
        End Sub

        Public Sub ConnectChildInit(layer As BaseLayer)
            For Each n2 As Neuron In Me.Neurons
                For Each n As Neuron In layer.Neurons
                    Dim weight = New Weight(0, n2, n)
                    n.WeightsToParent.Add(weight)
                    n2.WeightsToChild.Add(weight)
                Next
            Next
        End Sub

        Public Sub ConnectBias(bias As Neuron, Random As BaseRandom)
            For Each n As Neuron In Me.Neurons
                Dim weight = New Weight(Random.Generate(), bias, n)
                n.WeightToBias = weight
                bias.WeightsToChild.Add(weight)
            Next
        End Sub

        Public Sub InitBias(bias As Neuron, Random As BaseRandom)
            Dim i% = 0
            For Each n As Neuron In Me.Neurons
                Dim weight = New Weight(Random.Generate(), bias, n)
                n.WeightToBias = weight
                bias.WeightsToChild(i) = weight
                i += 1
            Next
        End Sub

        Public Sub ConnectBiasInit(bias As Neuron)
            For Each n As Neuron In Me.Neurons
                Dim weight = New Weight(0, bias, n)
                n.WeightToBias = weight
                bias.WeightsToChild.Add(weight)
            Next
        End Sub

        Public Function PrintWeights$()

            Dim sb As New System.Text.StringBuilder("{" & vbCrLf)

            Dim nbNeurons% = Me.Neurons.Count
            Dim numNeuron% = 0
            For Each n2 As Neuron In Me.Neurons
                numNeuron += 1
                If n2.Type = NeuronType.Input Then Continue For
                sb.Append(" {")
                Dim iNbW% = n2.WeightsToParent.Count
                Dim iNumW% = 0
                For Each rVal In n2.WeightsToParent
                    iNumW += 1
                    sb.Append(rVal.Value.ToString(format2Dec).ReplaceCommaByDot())
                    If iNumW < iNbW Then sb.Append(", ")
                Next

                If Not IsNothing(n2.WeightToBias) Then
                    sb.Append(", " & n2.WeightToBias.Value.ToString(format2Dec).ReplaceCommaByDot())
                End If

                Dim sVirg$ = ""
                If numNeuron < nbNeurons Then
                    sVirg = ","
                Else
                    sVirg = "}"
                End If

                sb.Append("}" & sVirg & vbCrLf)

            Next

            Return sb.ToString

        End Function

        Public Sub RestoreWeightsWithBias(weightsWithBias#(,),
            useBias As Boolean, bias As Neuron, layerParent As BaseLayer)

            Dim layerParentSize% = weightsWithBias.GetUpperBound(1)
            Dim weights#(Me.Neurons.Count - 1, layerParentSize)
            Dim biasWeights#(Me.Neurons.Count - 1)

            Dim i% = 0
            Dim nbNeurons% = Me.Neurons.Count
            Dim nbNeuronsParent% = layerParent.Neurons.Count

            For Each n2 As Neuron In Me.Neurons
                If n2.Type = NeuronType.Input Then Continue For
                Dim j% = 0
                For j = 0 To nbNeuronsParent - 1
                    weights(i, j) = weightsWithBias(i, j)
                Next
                If useBias Then biasWeights(i) = weightsWithBias(i, j)
                i += 1
            Next

            Me.Init()
            Me.ConnectParent2(layerParent, weights)
            If useBias Then Me.ConnectBias2(bias, biasWeights)

        End Sub

        Public Sub ConnectParent2(layer As BaseLayer, weights#(,))
            Dim i% = 0
            For Each n2 As Neuron In Me.Neurons
                Dim j% = 0
                Dim nbLayerNeurons% = layer.Neurons.Count
                For Each n As Neuron In layer.Neurons
                    Dim weight = New Weight(weights(i, j), n, n2)
                    If n.Type <> NeuronType.Output Then n.WeightsToChild.Add(weight)
                    n2.WeightsToParent.Add(weight)
                    j += 1
                Next
                i += 1
            Next
        End Sub

        Public Sub ConnectBias2(bias As Neuron, weights#())
            Dim i% = 0
            For Each n As Neuron In Me.Neurons
                Dim weight = New Weight(weights(i), bias, n)
                n.WeightToBias = weight
                bias.WeightsToChild.Add(weight)
                i += 1
            Next
        End Sub

    End Class

End Namespace