
Module modTests

    Sub Main()

        'XORTest()

        IrisTest()

        WaitForKeyToQuit()

    End Sub

    Private Sub XORTest()

        For nbXor = 1 To 3

            ClassicMLPTest(nbXor)
            NextTest()

            OOPMLPTest(nbXor)
            NextTest()

            MatrixMLPTest(nbXor)
            NextTest()

            VectorizedMatrixMLPTest(nbXor)
            NextTest()

            TensorMLPTest(nbXor)
            NextTest()

            AccordMLPTest(nbXor)
            NextTest()

            EncogMLPTest(nbXor)
            NextTest()

            'TensorFlowMLPTest(nbXor) ' Works only with 1XOR?
            'NextTest()

            'KerasMLPTest(nbXor)
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
