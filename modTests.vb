
Module modTests

    Sub Main()

        ClassicMLPTest()
        Console.WriteLine("Press a key to continue.")
        Console.ReadKey()
        Console.WriteLine()

        MatrixMLP.MatrixMLPTest()
        Console.WriteLine("Press a key to continue.")
        Console.ReadKey()
        Console.WriteLine()

        VectorizedMatrixMLP.VectorizedMatrixMLPTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()

    End Sub

End Module
