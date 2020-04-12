
Module modTests

    Sub Main()

        ClassicMLPTest()
        Console.WriteLine("Press a key to continue.")
        Console.ReadKey()

        MatrixMLP.MatrixMLPTest()
        Console.WriteLine("Press a key to continue.")
        Console.ReadKey()

        VectorizedMatrixMLP.VectorizedMatrixMLPTest()
        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()

    End Sub

End Module
