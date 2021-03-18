
Module modTests

    Sub Main()

        MLPMenu()

    End Sub

    Private Sub MLPMenu()

Retry:
        Console.WriteLine("")
        Console.WriteLine("")
        Console.WriteLine("MLP Test, choose an option from the following list:")
        Console.WriteLine("0: Exit")
        Console.WriteLine("a: Accord MLP MLP")
        Console.WriteLine("c: Classical MLP")
        Console.WriteLine("e: Encog MLP")
        Console.WriteLine("f: TensorFlow.NET MLP")
        Console.WriteLine("k: Keras.NET MLP")
        Console.WriteLine("m: Matrix MLP")
        Console.WriteLine("n: NeuralNet.NET MLP")
        Console.WriteLine("o: Object-oriented programming MLP")
        Console.WriteLine("r: Resilient Propagation MLP")
        Console.WriteLine("t: Tensor MLP")
        Console.WriteLine("v: Vectorized Matrix MLP")

        Dim k = Console.ReadKey
        Select Case k.KeyChar
            Case "0"c : Exit Sub
            Case "a"c : ApplicationMenu(k.KeyChar)
            Case "c"c : ApplicationMenu(k.KeyChar)
            Case "e"c : ApplicationMenu(k.KeyChar)
            Case "f"c : ApplicationMenu(k.KeyChar)
            Case "k"c : ApplicationMenu(k.KeyChar)
            Case "m"c : ApplicationMenu(k.KeyChar)
            Case "n"c : ApplicationMenu(k.KeyChar)
            Case "o"c : ApplicationMenu(k.KeyChar)
            Case "r"c : ApplicationMenu(k.KeyChar)
            Case "t"c : ApplicationMenu(k.KeyChar)
            Case "v"c : ApplicationMenu(k.KeyChar)
        End Select

        GoTo Retry

    End Sub

    Private Sub ApplicationMenu(mlpChoice As Char)

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
            Case "1"c : XORTest(mlpChoice, nbXor:=1)
            Case "2"c : XORTest(mlpChoice, nbXor:=2)
            Case "3"c : XORTest(mlpChoice, nbXor:=3)
            Case "4"c : IrisFlowerTestLogical(mlpChoice)
            Case "5"c : IrisFlowerTestAnalog(mlpChoice)
            Case "6"c : SunspotTest(mlpChoice)
        End Select

        GoTo Retry

    End Sub

    Private Sub XORTest(mlpChoice As Char, nbXor%)

        Console.WriteLine("")

        Select Case mlpChoice
            Case "a"c : AccordMLPXorTest(nbXor)
            Case "c"c : ClassicMLPXorTest(nbXor)
            Case "e"c : EncogMLPXorTest(nbXor)
            Case "f"c : TensorFlowMLPXORTest(nbXor) ' Works only with 1XOR?
            Case "k"c : KerasMLPXorTest(nbXor)
            Case "m"c : MatrixMLPXorTest(nbXor)
            Case "n"c : NeuralNetMLPXorTest(nbXor)
            Case "o"c : OOPMLPXorTest(nbXor)
            Case "r"c : RPropMLPXorTest(nbXor)
            Case "t"c : TensorMLPXorTest(nbXor)
            Case "v"c : VectorizedMatrixMLPXorTest(nbXor)
        End Select

        NextTest()

    End Sub

    Private Sub IrisFlowerTestLogical(mlpChoice As Char)

        Console.WriteLine("")

        Select Case mlpChoice

            Case "a"c
                MLPGenericIrisFlowerTest(New clsMLPAccord,
                    "Accord MLP Iris flower logical test")

            Case "c"c
                ' Works only using sigmoid activation
                MLPGenericIrisFlowerTest(New clsMLPClassic,
                    "Classic MLP Iris flower logical test", sigmoid:=True)

            Case "e"c
                MLPGenericIrisFlowerTest(New clsMLPEncog,
                    "Encog MLP Iris flower logical test")

            Case "f"c
                ' No bias, only 3 layers, poor results! (only 50% learning, 90% prediction)
                MLPGenericIrisFlowerTest(New clsMLPTensorFlow,
                    "TensorFlow.NET MLP Iris flower logical test",
                    nbIterations:=4000, threeLayers:=True, addBiasColumn:=False)

            Case "k"c
                ' Works only using sigmoid activation
                MLPGenericIrisFlowerTest(New clsMLPKeras,
                    "Keras.NET MLP Iris flower logical test",
                    nbIterations:=100, sigmoid:=True)

            Case "m"c
                ' Three layers only, poor results!
                MLPGenericIrisFlowerTest(New MatrixMLP.MultiLayerPerceptron,
                    "Matrix MLP Iris flower logical test", nbIterations:=4000, threeLayers:=True)

            Case "n"c
                MLPGenericIrisFlowerTest(New clsMLPNeuralNet,
                    "NeuralNet.NET MLP Iris flower logical test",
                    nbIterations:=3000, threeLayers:=True, sigmoid:=False)

            Case "o"c
                MLPGenericIrisFlowerTest(New NetworkOOP.MultilayerPerceptron,
                    "Object-oriented programming MLP Iris flower logical test")

            Case "r"c

                Dim mlpRProp As New clsMLPRProp
                mlpRProp.classificationObjective = True ' Sometimes 100% prediction
                MLPGenericIrisFlowerTest(mlpRProp,
                    "RProp MLP Iris flower logical test",
                    nbIterations:=200, threeLayers:=True, minValue:=-10.0!, maxValue:=10.0!)

                'MLPGenericIrisFlowerTest(mlpRProp, "RProp MLP Iris flower logical test",
                ' nbIterations:=200, sigmoid:=True)
                'NextTest()

                'MLPGenericIrisFlowerTest(New clsMLPRProp, "RProp MLP Iris flower logical test",
                '    nbIterations:=200, sigmoid:=True)
                'NextTest()

            Case "t"c
                ' 97.8% prediction, 98.9% learning with 300 iterations
                ' Nb hidden neurons = nb input neurons, works only using sigmoid activation
                MLPGenericIrisFlowerTest(New clsMLPTensor,
                    "Tensor MLP Iris flower logical test",
                    nbIterations:=300, nbHiddenLayersFromInput:=True, sigmoid:=True)

            Case "v"c
                ' Works only using sigmoid activation
                MLPGenericIrisFlowerTest(New VectorizedMatrixMLP.clsVectorizedMatrixMLP,
                    "Vectorized Matrix MLP Iris flower logical test",
                    nbIterations:=1000, sigmoid:=True)

        End Select

        NextTest()

    End Sub

    Private Sub IrisFlowerTestAnalog(mlpChoice As Char)

        Console.WriteLine("")

        Select Case mlpChoice

            Case "a"c
                MLPGenericIrisFlowerTestAnalog(New clsMLPAccord,
                    "Accord MLP Iris flower analog test")

            Case "c"c
                MLPGenericIrisFlowerTestAnalog(New clsMLPClassic,
                    "Classic MLP Iris flower analog test")

            Case "e"c
                MLPGenericIrisFlowerTestAnalog(New clsMLPEncog,
                    "Encog MLP Iris flower analog test")

            Case "f"c
                ' No bias, only 3 layers, no result!
                MLPGenericIrisFlowerTestAnalog(New clsMLPTensorFlow,
                    "TensorFlow.NET MLP Iris flower analog test",
                    nbIterations:=4000, threeLayers:=True, addBiasColumn:=False)

            Case "k"c
                ' Works only using sigmoid activation
                MLPGenericIrisFlowerTestAnalog(New clsMLPKeras,
                    "Keras.NET MLP Iris flower analog test",
                    nbIterations:=100, sigmoid:=True)

            Case "m"c
                ' Three layers only, same results!
                MLPGenericIrisFlowerTestAnalog(New MatrixMLP.MultiLayerPerceptron,
                    "Matrix MLP Iris flower analog test", nbIterations:=4000, threeLayers:=True)

            Case "n"c
                MLPGenericIrisFlowerTestAnalog(New clsMLPNeuralNet,
                    "NeuralNet.NET MLP Iris flower analog test", nbIterations:=3000, sigmoid:=False)

            Case "o"c
                MLPGenericIrisFlowerTestAnalog(New NetworkOOP.MultilayerPerceptron,
                    "Object-oriented programming MLP Iris flower analog test")

            Case "r"c
                MLPGenericIrisFlowerTestAnalog(New clsMLPRProp,
                    "RProp MLP Iris flower analog test", sigmoid:=True, nbIterations:=300) ' minValue:=-10, maxValue:=10, gain:=0.5!)

            Case "t"c
                ' 93.3% prediction, 95% learning with 300 iterations
                ' Nb hidden neurons = nb input neurons, works only using sigmoid activation
                MLPGenericIrisFlowerTestAnalog(New clsMLPTensor,
                    "Tensor MLP Iris flower analog test",
                    nbIterations:=300, nbHiddenLayersFromInput:=True, sigmoid:=True)

            Case "v"c
                ' Works only using sigmoid activation, poor results!
                MLPGenericIrisFlowerTestAnalog(New VectorizedMatrixMLP.clsVectorizedMatrixMLP,
                    "Vectorized Matrix MLP Iris flower analog test", sigmoid:=True)

        End Select

        NextTest()

    End Sub

    Public Sub SunspotTest(mlpChoice As Char)

        Console.WriteLine("")

        Select Case mlpChoice

            Case "a"c
                MLPGenericSunspotTest(New clsMLPAccord,
                    "Accord MLP Sunspot test")

            Case "c"c
                MLPGenericSunspotTest(New clsMLPClassic,
                    "Classic MLP Sunspot test")

            Case "e"c
                MLPGenericSunspotTest(New clsMLPEncog,
                    "Encog MLP Sunspot test")

            Case "f"c
                ' No bias, poor result!
                MLPGenericSunspotTest(New clsMLPTensorFlow,
                    "TensorFlow.NET MLP Sunspot test",
                    nbIterations:=4000, addBiasColumn:=False)

            Case "k"c
                ' Works only using sigmoid activation
                MLPGenericSunspotTest(New clsMLPKeras,
                    "Keras.NET MLP Sunspot test",
                    nbIterations:=100, sigmoid:=True)

            Case "m"c
                MLPGenericSunspotTest(New MatrixMLP.MultiLayerPerceptron,
                    "Matrix MLP Sunspot test")

            Case "n"c
                MLPGenericSunspotTest(New clsMLPNeuralNet,
                    "NeuralNet.NET MLP Sunspot test", sigmoid:=True, nbIterations:=3000)

            Case "o"c
                MLPGenericSunspotTest(New NetworkOOP.MultilayerPerceptron,
                    "Object-oriented programming MLP Sunspot test")

            Case "r"c
                MLPGenericSunspotTest(New clsMLPRProp,
                    "RProp MLP Sunspot test", sigmoid:=True, nbIterations:=300)

            Case "t"c
                ' Nb hidden neurons = nb input neurons, works only using sigmoid activation
                MLPGenericSunspotTest(New clsMLPTensor,
                    "Tensor MLP Sunspot test",
                    nbHiddenLayersFromInput:=True, sigmoid:=True)

            Case "v"c
                ' Works only using sigmoid activation
                MLPGenericSunspotTest(New VectorizedMatrixMLP.clsVectorizedMatrixMLP,
                    "Vectorized Matrix MLP Sunspot test", sigmoid:=True)

        End Select

        NextTest()

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
