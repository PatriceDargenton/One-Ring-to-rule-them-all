
Module modTests

    Sub Main()

        'XORTest()

        IrisFlowerTest()

        WaitForKeyToQuit()

    End Sub

    Private Sub XORTest()

        For nbXor = 1 To 3

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

        Next

    End Sub

    Private Sub IrisFlowerTest()

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
