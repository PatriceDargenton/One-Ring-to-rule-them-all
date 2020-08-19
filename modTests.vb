
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
        NextTest()

        AccordMLPTest()
        NextTest()

        EncogMLPTest()
        NextTest()

        TensorFlowMLPTest()

        Console.WriteLine("Press a key to quit.")
        Console.ReadKey()

    End Sub

    Private Sub NextTest()
        Console.WriteLine("Press a key to continue.")
        Console.ReadKey()
        Console.WriteLine()
    End Sub

    Public Sub WaitForKeyToStart()
        If Not isConsoleApp() Then Exit Sub
        Console.WriteLine("Press a key to start.")
        Console.ReadKey()
    End Sub

End Module
