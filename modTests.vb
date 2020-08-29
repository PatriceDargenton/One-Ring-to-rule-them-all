
Module modTests

    Sub Main()

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

            TensorFlowMLPTest(nbXor) ' Works only with 1XOR?
            NextTest()

            KerasMLPTest(nbXor)
            NextTest()

        Next

        WaitForKeyToQuit()

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
