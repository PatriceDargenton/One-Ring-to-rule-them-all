
Module modTests

    Sub Main()

        'XORTest()

        IrisTest()

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

    Private Sub IrisTest()

        MLPGenericIrisTest(New clsMLPClassic, "Classic MLP Iris test")
        NextTest()

        MLPGenericIrisTest(New NetworkOOP.MultilayerPerceptron,
            "Object-oriented programming MLP Iris test")
        NextTest()

        MLPGenericIrisTest(New MatrixMLP.MultiLayerPerceptron, "Matrix MLP Iris test")
        NextTest()

        MLPGenericIrisTest(New VectorizedMatrixMLP.clsVectorizedMatrixMLP,
            "Vectorized Matrix MLP Iris test")
        NextTest()

        MLPGenericIrisTest(New clsMLPTensor, "Tensor MLP Iris test")
        NextTest()

        MLPGenericIrisTest(New clsMLPAccord, "Accord MLP Iris test")
        NextTest()

        MLPGenericIrisTest(New clsMLPEncog, "Encog MLP Iris test")
        NextTest()

        EncogMLPIrisTest()
        NextTest()

        ' Poor results:
        'TensorFlowMLPIrisAnalogTest()
        'NextTest()
        'TensorFlowMLPIrisLogicalTest()
        'NextTest()

        'MLPGenericIrisTest(New clsMLPKeras, "Keras.NET MLP Iris test", nbIterations:=800)
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
