
Module modTests

    Sub Main()

Retry:
        Console.WriteLine("")
        Console.WriteLine("")
        Console.WriteLine("MLP Test, choose an option from the following list:")
        Console.WriteLine("0: Exit")
        Console.WriteLine("1: 1 XOR")
        Console.WriteLine("2: 2 XOR")
        Console.WriteLine("3: 3 XOR")
        Console.WriteLine("4: IRIS (Logical)")
        Console.WriteLine("5: IRIS (Analog)")
        Console.WriteLine("6: Sunspot")

        Dim k = Console.ReadKey
        Select Case k.KeyChar
            Case "0"c : Exit Sub
            Case "1"c : XORTest(nbXor:=1)
            Case "2"c : XORTest(nbXor:=2)
            Case "3"c : XORTest(nbXor:=3)
            Case "4"c : IrisFlowerTestLogical()
            Case "5"c : IrisFlowerTestAnalog()
            Case "6"c : SunspotTest()
        End Select

        GoTo Retry

    End Sub

    Private Sub XORTest(nbXor%)

        Console.WriteLine("")

        NeuralNetMLPXorTest(nbXor)
        NextTest()
        Exit Sub

        RPropMLPXorTest(nbXor)
        NextTest()

        ClassicMLPXorTest(nbXor)
        NextTest()

        OOPMLPXorTest(nbXor)
        NextTest()

        MatrixMLPXorTest(nbXor)
        NextTest()

        VectorizedMatrixMLPXorTest(nbXor)
        NextTest()

        TensorMLPXorTest(nbXor)
        NextTest()

        AccordMLPXorTest(nbXor)
        NextTest()

        EncogMLPXorTest(nbXor)
        NextTest()

        'TensorFlowMLPXORTest(nbXor) ' Works only with 1XOR?
        'NextTest()

        'KerasMLPXorTest(nbXor)
        'NextTest()

    End Sub

    Private Sub IrisFlowerTestLogical()

        Console.WriteLine("")

        MLPGenericIrisFlowerTest(New clsMLPNeuralNet, "NeuralNet MLP Iris flower test",
            nbIterations:=3000, threeLayers:=True, sigmoid:=False)
        NextTest()
        Exit Sub


        Dim mlpRProp As New clsMLPRProp
        mlpRProp.classificationObjective = True ' Sometimes 100% prediction
        MLPGenericIrisFlowerTest(mlpRProp, "RProp MLP Iris flower test",
            nbIterations:=200, threeLayers:=True, minValue:=-10.0!, maxValue:=10.0!)
        NextTest()

        'MLPGenericIrisFlowerTest(mlpRProp, "RProp MLP Iris flower test",
        ' nbIterations:=200, sigmoid:=True)
        'NextTest()

        'MLPGenericIrisFlowerTest(New clsMLPRProp, "RProp MLP Iris flower test",
        '    nbIterations:=200, sigmoid:=True)
        'NextTest()


        ' Works only using sigmoid activation
        MLPGenericIrisFlowerTest(New clsMLPClassic, "Classic MLP Iris flower test",
            sigmoid:=True)
        NextTest()

        MLPGenericIrisFlowerTest(New NetworkOOP.MultilayerPerceptron,
            "Object-oriented programming MLP Iris flower test")
        NextTest()

        ' Three layers only, poor results!
        MLPGenericIrisFlowerTest(New MatrixMLP.MultiLayerPerceptron,
            "Matrix MLP Iris flower test", nbIterations:=4000, threeLayers:=True)
        NextTest()

        ' Works only using sigmoid activation
        MLPGenericIrisFlowerTest(New VectorizedMatrixMLP.clsVectorizedMatrixMLP,
            "Vectorized Matrix MLP Iris flower test", nbIterations:=1000, sigmoid:=True)
        NextTest()

        ' 97.8% prediction, 98.9% learning with 300 iterations
        ' Nb hidden neurons = nb input neurons, works only using sigmoid activation
        MLPGenericIrisFlowerTest(New clsMLPTensor, "Tensor MLP Iris flower test",
            nbIterations:=300, nbHiddenLayersFromInput:=True, sigmoid:=True)
        NextTest()

        MLPGenericIrisFlowerTest(New clsMLPAccord, "Accord MLP Iris flower test")
        NextTest()

        MLPGenericIrisFlowerTest(New clsMLPEncog, "Encog MLP Iris flower test")
        NextTest()

        '' No bias, only 3 layers, poor results! (only 50% learning, 90% prediction)
        'MLPGenericIrisFlowerTest(New clsMLPTensorFlow, "TensorFlow MLP Iris flower test",
        '    nbIterations:=4000, threeLayers:=True, addBiasColumn:=False)
        'NextTest()

        '' Works only using sigmoid activation
        'MLPGenericIrisFlowerTest(New clsMLPKeras, "Keras.NET MLP Iris flower test",
        '    nbIterations:=100, sigmoid:=True)
        'NextTest()

    End Sub

    Private Sub IrisFlowerTestAnalog()

        Console.WriteLine("")

        MLPGenericIrisFlowerTestAnalog(New clsMLPNeuralNet,
            "NeuralNet MLP Iris flower test", nbIterations:=3000, sigmoid:=False)
        NextTest()
        Exit Sub

        MLPGenericIrisFlowerTestAnalog(New clsMLPRProp,
            "RProp MLP Iris flower test", sigmoid:=True, nbIterations:=300) ' minValue:=-10, maxValue:=10, gain:=0.5!)
        NextTest()

        MLPGenericIrisFlowerTestAnalog(New clsMLPClassic, "Classic MLP Iris flower test")
        NextTest()

        MLPGenericIrisFlowerTestAnalog(New NetworkOOP.MultilayerPerceptron,
            "Object-oriented programming MLP Iris flower test")
        NextTest()

        ' Three layers only, same results!
        MLPGenericIrisFlowerTestAnalog(New MatrixMLP.MultiLayerPerceptron,
            "Matrix MLP Iris flower test", nbIterations:=4000, threeLayers:=True)
        NextTest()

        ' Works only using sigmoid activation, poor results!
        MLPGenericIrisFlowerTestAnalog(New VectorizedMatrixMLP.clsVectorizedMatrixMLP,
            "Vectorized Matrix MLP Iris flower test", sigmoid:=True)
        NextTest()

        ' 93.3% prediction, 95% learning with 300 iterations
        ' Nb hidden neurons = nb input neurons, works only using sigmoid activation
        MLPGenericIrisFlowerTestAnalog(New clsMLPTensor, "Tensor MLP Iris flower test",
            nbIterations:=300, nbHiddenLayersFromInput:=True, sigmoid:=True)
        NextTest()

        MLPGenericIrisFlowerTestAnalog(New clsMLPAccord, "Accord MLP Iris flower test")
        NextTest()

        MLPGenericIrisFlowerTestAnalog(New clsMLPEncog, "Encog MLP Iris flower test")
        NextTest()

        ' No bias, only 3 layers, no result!
        'MLPGenericIrisFlowerTestAnalog(New clsMLPTensorFlow, "TensorFlow MLP Iris flower test",
        '    nbIterations:=4000, threeLayers:=True, addBiasColumn:=False)
        'NextTest()

        '' Works only using sigmoid activation
        'MLPGenericIrisFlowerTestAnalog(New clsMLPKeras, "Keras.NET MLP Iris flower test",
        '    nbIterations:=100, sigmoid:=True)
        'NextTest()

    End Sub

    Public Sub SunspotTest()

        Console.WriteLine("")

        MLPGenericSunspotTest(New clsMLPNeuralNet,
            "NeuralNet MLP Sunspot test", sigmoid:=True, nbIterations:=3000)
        NextTest()
        Exit Sub

        MLPGenericSunspotTest(New clsMLPRProp,
            "RProp MLP Sunspot test", sigmoid:=True, nbIterations:=300)
        NextTest()

        MLPGenericSunspotTest(New clsMLPClassic, "Classic MLP Sunspot test")
        NextTest()

        MLPGenericSunspotTest(New NetworkOOP.MultilayerPerceptron,
            "Object-oriented programming MLP Sunspot test")
        NextTest()

        MLPGenericSunspotTest(New MatrixMLP.MultiLayerPerceptron,
            "Matrix MLP Sunspot test")
        NextTest()

        ' Works only using sigmoid activation
        MLPGenericSunspotTest(New VectorizedMatrixMLP.clsVectorizedMatrixMLP,
            "Vectorized Matrix MLP Sunspot test", sigmoid:=True)
        NextTest()

        ' Nb hidden neurons = nb input neurons, works only using sigmoid activation
        MLPGenericSunspotTest(New clsMLPTensor, "Tensor MLP Sunspot test",
            nbHiddenLayersFromInput:=True, sigmoid:=True)
        NextTest()

        MLPGenericSunspotTest(New clsMLPAccord, "Accord MLP Sunspot test")
        NextTest()

        MLPGenericSunspotTest(New clsMLPEncog, "Encog MLP Sunspot test")
        NextTest()

        '' No bias, poor result!
        'MLPGenericSunspotTest(New clsMLPTensorFlow, "TensorFlow MLP Sunspot test",
        '    nbIterations:=4000, addBiasColumn:=False)
        'NextTest()

        '' Works only using sigmoid activation
        'MLPGenericSunspotTest(New clsMLPKeras, "Keras.NET MLP Sunspot test",
        '    nbIterations:=100, sigmoid:=True)
        'NextTest()

    End Sub

    Public Sub NextTest()
        Console.WriteLine("Press a key to continue.")
        Console.ReadKey()
        Console.WriteLine()
    End Sub

    Public Sub WaitForKeyToContinue(msg$)
        If Not isConsoleApp() Then Exit Sub
        Console.WriteLine(msg)
        Console.ReadKey()
    End Sub

    Public Sub WaitForKeyToStart()
        If Not isConsoleApp() Then Exit Sub
        Console.WriteLine("Press a key to start.")
        Console.ReadKey()
    End Sub

    Public Sub WaitForKeyToQuit()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()
    End Sub

End Module
