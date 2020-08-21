
' From  : https://github.com/SciSharp/SciSharp-Stack-Examples/blob/master/src/TensorFlowNET.Examples/NeuralNetworks/NeuralNetXor.cs : C# -> VB .NET conversion
' https://www.nuget.org/packages/TensorFlow.NET
' https://www.nuget.org/packages/Microsoft.ML.TensorFlow.Redist
' Install-Package TensorFlow.NET -Version 0.15.1
' Install-Package Microsoft.ML.TensorFlow.Redist
' https://github.com/SciSharp/SciSharp-Stack-Examples
' https://github.com/SciSharp/TensorFlow.NET
' https://tensorflownet.readthedocs.io/en/latest

Imports NumSharp
Imports Tensorflow
Imports Tensorflow.Binding

Imports System.Text

' Tuples are not available for Visual Studio 2013: Set 0: Off
#Const Implementation = 1 ' 0: Off, 1: On

Public Class clsMLPTensorFlow : Inherits clsVectorizedMLPGeneric

    Const computeScoreOneByOne = False

    Private sess As Session
    Private data As NDArray
    Private target As NDArray
    Private targetArray1D!()

    Private features, labels As Tensor

    Dim outputs1D#()

    Dim hiddenWeights#(), outputWeights#()

#If Implementation Then
    Private graphTuple As (
        trainOperation As Operation, loss As Tensor,
        globalStep As Tensor, prediction As Tensor, hw As Tensor, ow As Tensor)
#End If

    Public Overrides Sub InitializeStruct(neuronCount%(), addBiasColumn As Boolean)

        Me.useBias = addBiasColumn
        Me.layerCount = neuronCount.Length
        Me.neuronCount = neuronCount
        Me.nbInputNeurons = Me.neuronCount(0)
        Me.nbHiddenNeurons = Me.neuronCount(1)
        Me.nbOutputNeurons = Me.neuronCount(Me.layerCount - 1)

        ReDim Me.hiddenWeights(Me.nbInputNeurons * Me.nbHiddenNeurons - 1)
        ReDim Me.outputWeights(Me.nbHiddenNeurons * Me.nbOutputNeurons - 1)

        If Me.useBias Then
            Throw New NotImplementedException(
                "useBias is not implemented for clsMLPTensorFlow!")
        End If
        If Me.layerCount <> 3 Then
            MsgBox("This TensorFlow implementation can only compute one hidden layer!",
                MsgBoxStyle.Exclamation)
            Me.layerCount = 3
        End If

        InitializeTensorFlow()

    End Sub

#If Implementation Then

    Private Sub InitializeTensorFlow()

        ' Check TensorFlow.Redist dll tensorflow.dll
        'Dim exePath = Application.StartupPath() ' WinForm
        ' Console app.:
        Dim asmPath = System.Reflection.Assembly.GetExecutingAssembly().Location
        Dim exePath = System.IO.Path.GetDirectoryName(asmPath)
        Dim dllPath = exePath & "\tensorflow.dll"
        If Not System.IO.File.Exists(dllPath) Then
            Const TFDllPath = "\packages\Microsoft.ML.TensorFlow.Redist.0.14.0\runtimes\win-x64\native\tensorflow.dll"
            Dim srcDllPath = System.IO.Path.GetDirectoryName(exePath) & TFDllPath
            ' If \bin\Debug directory:
            Dim srcDllPath2 = System.IO.Path.GetDirectoryName(
                System.IO.Path.GetDirectoryName(exePath)) & TFDllPath
            If Not System.IO.File.Exists(srcDllPath) AndAlso
               Not System.IO.File.Exists(srcDllPath2) Then _
                Throw New System.IO.FileNotFoundException(
                    "Please build the solution in Debug mode!")
            If System.IO.File.Exists(srcDllPath) Then
                System.IO.File.Copy(srcDllPath, dllPath)
            ElseIf System.IO.File.Exists(srcDllPath2) Then
                System.IO.File.Copy(srcDllPath2, dllPath)
            End If
        End If

        Dim graph = tf.Graph.as_default
        Me.exampleCount = Me.inputArray.GetLength(0)
        Me.features = tf.placeholder(tf.float32,
            New TensorShape(Me.exampleCount, Me.nbInputNeurons))
        Me.labels = tf.placeholder(tf.int32,
            New TensorShape(Me.exampleCount, Me.nbOutputNeurons))
        If Me.nbOutputNeurons = 1 Then _
            Me.labels = tf.placeholder(tf.int32, New TensorShape(Me.exampleCount))

        Dim num_hidden = Me.nbHiddenNeurons
        Me.graphTuple = makeGraph(Me.features, Me.labels,
            num_hidden:=num_hidden, num_input:=Me.nbInputNeurons,
            num_output:=Me.nbOutputNeurons)

        Dim init = tf.global_variables_initializer
        Me.sess = tf.Session(graph)
        Me.sess.run(init)

        Me.data = Me.inputArray
        If Me.nbOutputNeurons = 1 Then
            Me.targetArray1D = clsMLPHelper.GetColumn(Me.targetArray, 0)
            Me.target = np.array(Me.targetArray1D, dtype:=np.float32)
        Else
            Me.target = np.array(Me.targetArray, dtype:=np.float32)
        End If

    End Sub

    Private Function makeGraph(
        features As Tensor, labels As Tensor, num_hidden%, num_input%, num_output%) _
        As (op As Operation, t1 As Tensor, t2 As Tensor, t3 As Tensor,
            t4 As Tensor, t5 As Tensor)

        Dim stddev = 1 / Math.Sqrt(2)
        Dim hidden_weights = tf.Variable(tf.truncated_normal(
            New Integer() {num_input, num_hidden},
            seed:=1, stddev:=CType(stddev, Single)), name:="hidden_weights")
        Dim hidden_activations = tf.nn.relu(tf.matmul(features, hidden_weights))

        Dim output_weights = tf.Variable(tf.truncated_normal(
            {num_hidden, num_output}, seed:=17,
            stddev:=CSng(1 / Math.Sqrt(num_hidden))), name:="output_weights")

        ' Shape [4, 1] for one XOR
        Dim logits = tf.matmul(hidden_activations, output_weights)

        ' Shape [4] for one XOR
        Dim prediction = tf.tanh(tf.squeeze(logits), name:="prediction")

        Dim tfCast = tf.cast(labels, tf.float32)
        Dim diff = prediction - tfCast
        Dim loss = tf.reduce_mean(tf.square(diff), name:="loss")

        Dim globalStep = tf.Variable(0, trainable:=False, name:="global_step")
        Dim trainOperation = tf.train.GradientDescentOptimizer(
            Me.learningRate).minimize(loss, global_step:=globalStep)

        Dim returnTuple As (
            trainOperation As Operation, loss As Tensor,
            globalStep As Tensor, prediction As Tensor, hw As Tensor, ow As Tensor) =
            (trainOperation, loss, globalStep, prediction, hidden_weights, output_weights)

        Return returnTuple

    End Function

    Private Sub ReadWeights()

        Dim resultTuple As (
            _x As NDArray, step_ As NDArray,
            loss_value As NDArray, predictionValue As NDArray, hw As NDArray) =
            Me.sess.run(
                (Me.graphTuple.trainOperation,
                 Me.graphTuple.globalStep,
                 Me.graphTuple.loss,
                 Me.graphTuple.prediction, Me.graphTuple.hw),
                (Me.features, Me.data),
                (Me.labels, Me.target))

        Me.hiddenWeights = ConvertNDArrayToArrayOfDouble(resultTuple.hw,
            Me.nbInputNeurons, Me.nbHiddenNeurons)

        Dim resultTuple2 As (
            _x As NDArray, step_ As NDArray,
            loss_value As NDArray, predictionValue As NDArray, ow As NDArray) =
            Me.sess.run(
                (Me.graphTuple.trainOperation,
                    Me.graphTuple.globalStep,
                    Me.graphTuple.loss,
                    Me.graphTuple.prediction, Me.graphTuple.ow),
                (Me.features, Me.data),
                (Me.labels, Me.target))

        Me.outputWeights = ConvertNDArrayToArrayOfDouble(resultTuple2.ow,
            Me.nbHiddenNeurons, Me.nbOutputNeurons)

    End Sub

#Else
    Private Sub InitializeTensorFlow()
    End Sub
    Private Sub ReadWeights()
    End Sub
#End If

    Public Overrides Sub SetActivationFunction(
        actFnc As enumActivationFunction, gain!, center!)

        ' gain can only be 2 for TensorFlow MLP
        gain = 2
        center = 0
        Me.weightAdjustment = 0 ' Not used

        MyBase.SetActivationFunction(actFnc, gain, center)

        If actFnc <> enumActivationFunction.HyperbolicTangent Then
            Throw New NotImplementedException(
                "This activation function is not available!")
        End If

    End Sub

    Public Overrides Sub InitializeWeights(numLayer%, weights#(,))

        ' Just display a message, let's play all the tests
        ShowMessage("NotImplementedException: InitializeWeights is not implemented for clsMLPTensorFlow")

    End Sub

    Public Overrides Sub Randomize(Optional minValue! = 0, Optional maxValue! = 1)

        ReadWeights()

    End Sub

    Public Overrides Sub TrainVector()

        Me.vectorizedLearningMode = True
        For iteration = 0 To Me.nbIterations - 1
            TrainVectorOneIteration()
            If Me.printOutput_ Then PrintOutput(iteration)
        Next

        If computeScoreOneByOne Then
            Dim nbTargets = Me.targetArray.GetLength(1)
            TestAllSamples(Me.inputArray, nbOutputs:=nbTargets)
            ComputeAverageError()
        Else
            SetOuput1D()
        End If

        CloseSession()

    End Sub

    Public Overrides Sub CloseSession()
        If Not IsNothing(Me.sess) Then Me.sess.close()
    End Sub

    Public Sub TrainVectorOneIteration()

#If Implementation Then

        Dim resultTuple As (
            _x As NDArray, step_ As NDArray,
            loss_value As NDArray, predictionValue As NDArray) =
            Me.sess.run(
                (Me.graphTuple.trainOperation,
                 Me.graphTuple.globalStep,
                 Me.graphTuple.loss,
                 Me.graphTuple.prediction),
                (Me.features, Me.data),
                (Me.labels, Me.target))

        Me.averageError = resultTuple.loss_value

        Me.outputs1D = ConvertNDArrayToArrayOfDouble(resultTuple.predictionValue,
            Me.exampleCount, Me.nbOutputNeurons)

#End If

    End Sub

    Private Function ConvertNDArrayToArrayOfDouble(nda As NDArray, r%, c%) As Double()

        ' input.ToArray<float> works in C#, but
        ' input.ToArray(Of Single) does not work in VB.Net : BC30649 "Unsupported Type"
        ' https://github.com/SciSharp/NumSharp.Lite
        ' https://github.com/shimat/opencvsharp_samples/issues/23 Probably similar issue

        Dim length = r * c
        Dim output#(length - 1)

        Dim k = 0
        For i = 0 To r - 1 ' row
            For j = 0 To c - 1 ' column
                Dim vt As ValueType = nda.GetAtIndex(i * c + j)
                ' Then no other solution to parse the string containing the value,
                '  because ValueType is a generic type
                Dim strVal = vt.ToString()
                Dim sngVal = Single.Parse(strVal)
                output(k) = sngVal
                k += 1
            Next
        Next

        Return output

    End Function

    Public Overrides Sub SetOuput1D()

        If IsNothing(Me.outputs1D) Then Exit Sub
        Dim nbInputs = Me.inputArray.GetLength(0)
        Dim nbTargets = Me.targetArray.GetLength(0)
        Dim outputs2D#(nbTargets - 1, Me.nbOutputNeurons - 1)
        For i = 0 To nbInputs - 1
            For j = 0 To Me.nbOutputNeurons - 1
                outputs2D(i, j) = Me.outputs1D(i)
            Next
        Next
        Me.output = outputs2D
        Me.lastOutputArray1DSingle = clsMLPHelper.Convert1DArrayOfDoubleToSingle(Me.outputs1D)

    End Sub

    Public Overrides Sub TrainSystematic(inputs!(,), targets!(,),
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        'If learningMode = enumLearningMode.Vectorial Then
        ' This is the unique learning mode for this MLP
        TrainVector()
        '    Exit Sub
        'End If

        'Me.vectorizedLearningMode = False
        'TrainAllSamples(inputs, targets)

    End Sub

    Public Overrides Sub TrainStochastic(inputs!(,), targets!(,))
        Throw New NotImplementedException("There is no TrainOneSample() function!")
    End Sub

    Public Overrides Sub TrainSemiStochastic(inputs!(,), targets!(,))
        Throw New NotImplementedException("There is no TrainOneSample() function!")
    End Sub

    Public Overrides Sub TrainOneSample(input!(), target!())
        Throw New NotImplementedException("There is no TrainOneSample() function!")
    End Sub

#If Implementation Then

    Public Overrides Sub TestOneSample(input!())

        ' We use only the first placeholder

        ' 1XOR
        'Me.data({0, 0}) = input(0)
        'Me.data({0, 1}) = input(1)

        ' 2XOR
        'Me.data({0, 0}) = input(0)
        'Me.data({0, 1}) = input(1)
        'Me.data({0, 2}) = input(2)
        'Me.data({0, 3}) = input(3)

        For m = 0 To Me.nbInputNeurons - 1
            Dim lst As New List(Of Integer)
            Dim k = 0
            For l = 0 To 1
                lst.Add(k)
                k = m
            Next
            Dim prmArr = lst.ToArray
            Dim value = input(m)
            Me.data(prmArr) = value
        Next

        Dim resultTuple As (
            _x As NDArray, step_ As NDArray,
            loss_value As NDArray, predictionValue As NDArray) =
            Me.sess.run(
                (Me.graphTuple.trainOperation,
                 Me.graphTuple.globalStep,
                 Me.graphTuple.loss,
                 Me.graphTuple.prediction),
                (Me.features, Me.data),
                (Me.labels, Me.target))

        Me.outputs1D = ConvertNDArrayToArrayOfDouble(resultTuple.predictionValue,
            r:=1, c:=Me.nbOutputNeurons)

        Me.lastOutputArray1DSingle = clsMLPHelper.Convert1DArrayOfDoubleToSingle(Me.outputs1D)

        'Dim s = clsMLPHelper.ArrayToString(input)
        'Debug.WriteLine(s & " -> " & clsMLPHelper.ArrayToString(Me.lastOutputArray1DSingle))

    End Sub

#Else

    Public Overrides Sub TestOneSample(input!())
        ReDim Me.lastOutputArray1DSingle(Me.nbOutputNeurons - 1)
    End Sub

#End If

    Public Overrides Sub PrintWeights()

        Me.PrintParameters()

        For i = 0 To Me.layerCount - 1
            ShowMessage("Neuron count(" & i & ")=" & Me.neuronCount(i))
        Next

        ShowMessage("")

        Dim sb As New StringBuilder


        For i = 1 To Me.layerCount - 1

            sb.AppendLine("W(" & i & ")={")

            Dim nbNeuronsLayer = Me.nbHiddenNeurons
            Dim nbNeuronsPreviousLayer = Me.nbHiddenNeurons
            Dim lMax%
            If i = 1 Then
                nbNeuronsPreviousLayer = Me.nbInputNeurons
                lMax = Me.hiddenWeights.GetUpperBound(0)
            ElseIf i = Me.layerCount - 1 Then
                nbNeuronsLayer = Me.nbOutputNeurons
                lMax = Me.outputWeights.GetUpperBound(0)
            End If

            Dim l = 0
            For j = 0 To nbNeuronsLayer - 1
                sb.Append(" {")

                Dim nbWeights = nbNeuronsPreviousLayer
                For k = 0 To nbWeights - 1
                    Dim weight# = 0
                    If i = 1 Then
                        If l <= lMax Then weight = Me.hiddenWeights(l)
                    ElseIf i = Me.layerCount - 1 Then
                        If l <= lMax Then weight = Me.outputWeights(l)
                    End If
                    l += 1
                    Dim sVal$ = weight.ToString(format2Dec).ReplaceCommaByDot()
                    sb.Append(sVal)
                    If Me.useBias OrElse k < nbWeights - 1 Then sb.Append(", ")
                Next k

                If Me.useBias Then
                    Dim weightT = 0
                    Dim sValT$ = weightT.ToString(format2Dec).ReplaceCommaByDot()
                    sb.Append(sValT)
                End If

                sb.Append("}")
                If j < nbNeuronsLayer - 1 Then sb.Append("," & vbLf)
            Next j
            sb.Append("}" & vbLf)

            If i < Me.layerCount - 1 Then sb.AppendLine()

        Next i

        ShowMessage(sb.ToString())

    End Sub

    Public Overrides Sub PrintOutput(iteration%)

        If ShowThisIteration(iteration) Then

            If computeScoreOneByOne Then
                Dim nbTargets = Me.targetArray.GetLength(1)
                TestAllSamples(Me.inputArray, nbOutputs:=nbTargets)
                If IsNothing(Me.outputs1D) Then
                    ShowMessage(vbLf & "Iteration n°" & iteration + 1 & "/" & nbIterations & vbLf &
                        "Output: nothing!")
                    Exit Sub
                End If
            Else
                If Not Me.vectorizedLearningMode Then
                    Dim nbTargets = Me.targetArray.GetLength(1)
                    TestAllSamples(Me.inputArray, nbOutputs:=nbTargets)
                End If
                If IsNothing(Me.outputs1D) Then
                    ShowMessage(vbLf & "Iteration n°" & iteration + 1 & "/" & nbIterations & vbLf &
                        "Output: nothing!")
                    Exit Sub
                End If
                SetOuput1D()
            End If
            ComputeAverageError()
            PrintSuccess(iteration)
        End If

    End Sub

End Class