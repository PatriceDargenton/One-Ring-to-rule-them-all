
Module modTests

    Sub Main()

        ClassicMLPTest()
        NextTest()

        OOPMLPTest()
        NextTest()

        MatrixMLPTest()
        NextTest()

        VectorizedMatrixMLPTest()
        NextTest()

        TensorMLPTest()

        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()

    End Sub

    Private Sub NextTest()
        Console.WriteLine("Press a key to continue.")
        Console.ReadKey()
        Console.WriteLine()
    End Sub

End Module
