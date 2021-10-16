
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode

Public Module modMLPTest

#Region "XOR data set"

    Public ReadOnly m_neuronCountXOR%() = {2, 2, 1}
    Public ReadOnly m_neuronCountXOR231%() = {2, 3, 1} ' With bias
    Public ReadOnly m_neuronCountXOR241%() = {2, 4, 1}
    Public ReadOnly m_neuronCountXOR261%() = {2, 6, 1} ' TensorFlow minimal size
    Public ReadOnly m_neuronCountXOR271%() = {2, 7, 1} ' Keras minimal size
    Public ReadOnly m_neuronCountXOR291%() = {2, 9, 1} ' Keras minimal size for tanh
    Public ReadOnly m_neuronCountXOR2_10_1%() = {2, 10, 1}
    Public ReadOnly m_neuronCountXOR2_16_1%() = {2, 16, 1}

    Public ReadOnly m_neuronCount2XOR%() = {4, 4, 2}
    Public ReadOnly m_neuronCount2XOR4Layers%() = {4, 4, 4, 2}
    Public ReadOnly m_neuronCount2XOR5Layers%() = {4, 4, 4, 4, 2}
    Public ReadOnly m_neuronCount2XOR452%() = {4, 5, 2}
    Public ReadOnly m_neuronCount2XOR462%() = {4, 6, 2} ' TensorFlow minimal size
    Public ReadOnly m_neuronCount2XOR472%() = {4, 7, 2}
    Public ReadOnly m_neuronCount2XOR482%() = {4, 8, 2}
    Public ReadOnly m_neuronCount2XOR4_10_2%() = {4, 10, 2}
    Public ReadOnly m_neuronCount2XOR4_32_2%() = {4, 32, 2} ' Keras minimal size: stable!
    Public ReadOnly m_neuronCount3XOR%() = {6, 6, 3}
    Public ReadOnly m_neuronCount3XOR4Layers%() = {6, 6, 6, 3}
    Public ReadOnly m_neuronCount3XOR5Layers%() = {6, 6, 6, 6, 3}
    Public ReadOnly m_neuronCount3XOR673%() = {6, 7, 3}
    Public ReadOnly m_neuronCount3XOR683%() = {6, 8, 3}
    Public ReadOnly m_neuronCount3XOR6_10_3%() = {6, 10, 3} ' Keras minimal size: stable!
    Public ReadOnly m_neuronCount3XOR6_32_3%() = {6, 32, 3} ' Keras stable size for tanh

    Public ReadOnly m_neuronCountXOR4Layers%() = {2, 2, 2, 1}
    Public ReadOnly m_neuronCountXOR4Layers2331%() = {2, 3, 3, 1}
    Public ReadOnly m_neuronCountXOR4Layers2661%() = {2, 6, 6, 1} ' Keras minimal size

    Public ReadOnly m_neuronCountXOR5Layers%() = {2, 2, 2, 2, 1}
    Public ReadOnly m_neuronCountXOR5Layers23331%() = {2, 3, 3, 3, 1}
    Public ReadOnly m_neuronCountXOR5Layers27771%() = {2, 7, 7, 7, 1} ' Keras minimal size

    Public Sub InitXOR(mlp As clsMLPGeneric)
        mlp.Initialize(learningRate:=0.01!)
        mlp.inputArray = m_inputArrayXOR
        mlp.targetArray = m_targetArrayXOR
    End Sub

    Public Sub Init2XOR(mlp As clsMLPGeneric)
        mlp.Initialize(learningRate:=0.01!)
        mlp.inputArray = m_inputArray2XOR
        mlp.targetArray = m_targetArray2XOR
    End Sub

    Public Sub Init3XOR(mlp As clsMLPGeneric)
        mlp.Initialize(learningRate:=0.01!)
        mlp.inputArray = m_inputArray3XOR
        mlp.targetArray = m_targetArray3XOR
    End Sub

#End Region

#Region "Iris flower data set"

    Public ReadOnly m_neuronCountIrisFlowerAnalog%() = {4, 16, 16, 1}
    Public ReadOnly m_neuronCountIrisFlowerAnalog451%() = {4, 5, 1}
    Public ReadOnly m_neuronCountIrisFlowerAnalog4_20_1%() = {4, 20, 1}
    Public ReadOnly m_neuronCountIrisFlowerAnalog4991%() = {4, 9, 9, 1}
    Public ReadOnly m_neuronCountIrisFlowerLogical%() = {4, 16, 16, 3}
    Public ReadOnly m_neuronCountIrisFlowerLogical443%() = {4, 4, 3}
    Public ReadOnly m_neuronCountIrisFlowerLogical453%() = {4, 5, 3}
    Public ReadOnly m_neuronCountIrisFlowerLogical463%() = {4, 6, 3}
    Public ReadOnly m_neuronCountIrisFlowerLogical4663%() = {4, 6, 6, 3}
    Public ReadOnly m_neuronCountIrisFlowerLogical4773%() = {4, 7, 7, 3}
    Public ReadOnly m_neuronCountIrisFlowerLogical4883%() = {4, 8, 8, 3}
    Public ReadOnly m_neuronCountIrisFlowerLogical4_16_83%() = {4, 16, 8, 3}
    Public ReadOnly m_neuronCountIrisFlowerLogical4_20_3%() = {4, 20, 3}

    Public Sub InitIrisFlowerAnalog4Layers(mlp As clsMLPGeneric)
        mlp.inputArray = m_inputArrayIrisFlowerTrain
        mlp.targetArray = m_targetArrayIrisFlowerAnalogTrain
        mlp.InitializeStruct(m_neuronCountIrisFlowerAnalog4991, addBiasColumn:=True)
    End Sub

    Public Sub InitIrisFlowerLogical(mlp As clsMLPGeneric)
        'mlp.inputArray = m_inputArrayIrisFlower
        'mlp.targetArray = m_targetArrayIrisFlowerLogical
        mlp.inputArray = m_inputArrayIrisFlowerTrain
        mlp.targetArray = m_targetArrayIrisFlowerLogicalTrain
        mlp.InitializeStruct(m_neuronCountIrisFlowerLogical443, addBiasColumn:=True)
    End Sub

    Public Sub InitIrisFlowerLogical4Layers(mlp As clsMLPGeneric)
        mlp.inputArray = m_inputArrayIrisFlowerTrain
        mlp.targetArray = m_targetArrayIrisFlowerLogicalTrain
        mlp.InitializeStruct(m_neuronCountIrisFlowerLogical4_16_83, addBiasColumn:=True)
    End Sub

#End Region

#Region "Sunspot data set"

    Public Sub InitSunspot1(mlp As clsMLPGeneric)
        mlp.seriesArray = m_sunspotArray
        mlp.windowsSize = 7
        mlp.nbLinesToLearn = 48
        mlp.nbLinesToPredict = 10
        mlp.InitializeStruct({7, 20, 1}, addBiasColumn:=True)
    End Sub

    Public Sub InitSunspot2(mlp As clsMLPGeneric)
        mlp.seriesArray = m_sunspotArray
        mlp.windowsSize = 3
        mlp.nbLinesToLearn = 95
        mlp.nbLinesToPredict = 100
        mlp.InitializeStruct({3, 20, 1}, addBiasColumn:=True)
    End Sub

#End Region

#Region "Vectorized tests"
#End Region

#Region "Iris flower standard tests"

    Public Sub MLPGenericIrisFlowerTest(mlp As clsMLPGeneric, testName$,
        Optional nbIterations% = 2000,
        Optional threeLayers As Boolean = False,
        Optional addBiasColumn As Boolean = True,
        Optional nbHiddenLayersFromInput As Boolean = False,
        Optional sigmoid As Boolean = False,
        Optional minValue! = -0.5, Optional maxValue! = 0.5, Optional gain! = 2)

        mlp.ShowMessage(testName)

        mlp.nbIterations = nbIterations

        mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

        mlp.minimalSuccessTreshold = 0.3
        mlp.printOutput_ = True
        mlp.printOutputMatrix = False

        If threeLayers Then
            mlp.inputArray = m_inputArrayIrisFlowerTrain
            mlp.targetArray = m_targetArrayIrisFlowerLogicalTrain
            mlp.InitializeStruct(m_neuronCountIrisFlowerLogical4_20_3, addBiasColumn)
        ElseIf nbHiddenLayersFromInput Then
            mlp.inputArray = m_inputArrayIrisFlowerTrain
            mlp.targetArray = m_targetArrayIrisFlowerLogicalTrain
            ' clsMLPTensor: Set activation function before InitializeStruct
            mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.Sigmoid, gain)
            mlp.InitializeStruct({4, 4, 4, 3}, addBiasColumn)
        Else
            InitIrisFlowerLogical4Layers(mlp)
        End If

        If sigmoid Then
            mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)
        Else
            mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain)
        End If

        'mlp.Randomize()
        mlp.Randomize(minValue, maxValue)

        mlp.PrintParameters()

        WaitForKeyToStart()

        mlp.Train()

        mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
            m_targetArrayIrisFlowerLogicalTest, nbOutputs:=3)
        mlp.PrintSuccessPrediction()

        mlp.ShowMessage(testName & ": Done.")

    End Sub

    Public Sub MLPGenericIrisFlowerTestAnalog(mlp As clsMLPGeneric, testName$,
                Optional nbIterations% = 1000,
                Optional threeLayers As Boolean = False,
                Optional addBiasColumn As Boolean = True,
                Optional nbHiddenLayersFromInput As Boolean = False,
                Optional sigmoid As Boolean = False,
                Optional minValue! = -0.5, Optional maxValue! = 0.5, Optional gain! = 2)

        mlp.ShowMessage(testName)

        mlp.nbIterations = nbIterations

        mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

        mlp.minimalSuccessTreshold = 0.2
        mlp.printOutput_ = True
        mlp.printOutputMatrix = False

        If threeLayers Then
            mlp.inputArray = m_inputArrayIrisFlowerTrain
            mlp.targetArray = m_targetArrayIrisFlowerAnalogTrain
            mlp.InitializeStruct(m_neuronCountIrisFlowerAnalog4_20_1, addBiasColumn)
        ElseIf nbHiddenLayersFromInput Then
            mlp.inputArray = m_inputArrayIrisFlowerTrain
            mlp.targetArray = m_targetArrayIrisFlowerAnalogTrain
            ' clsMLPTensor: Set activation function before InitializeStruct
            mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.Sigmoid, gain)
            mlp.InitializeStruct({4, 4, 4, 1}, addBiasColumn)
        Else
            InitIrisFlowerAnalog4Layers(mlp)
        End If

        If sigmoid Then
            mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)
        Else
            mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain)
        End If

        'mlp.Randomize()
        mlp.Randomize(minValue, maxValue)

        mlp.PrintParameters()

        WaitForKeyToStart()

        mlp.Train()

        mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
            m_targetArrayIrisFlowerAnalogTest, nbOutputs:=1)
        mlp.PrintSuccessPrediction()

        mlp.ShowMessage(testName & ": Done.")

    End Sub

#End Region

#Region "Sunspot tests"

    Public Sub MLPGenericSunspotTest(mlp As clsMLPGeneric, testName$,
                Optional nbIterations% = 500,
                Optional addBiasColumn As Boolean = True,
                Optional nbHiddenLayersFromInput As Boolean = False,
                Optional sigmoid As Boolean = False,
                Optional minValue! = -0.5, Optional maxValue! = 0.5, Optional gain! = 1)

        mlp.ShowMessage(testName)

        mlp.nbIterations = nbIterations

        mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

        mlp.minimalSuccessTreshold = 0.1
        mlp.printOutput_ = True
        mlp.printOutputMatrix = False

        mlp.seriesArray = m_sunspotArray
        mlp.windowsSize = 10
        mlp.nbLinesToLearn = 48
        mlp.nbLinesToPredict = 10

        If nbHiddenLayersFromInput Then
            ' clsMLPTensor: Set activation function before InitializeStruct
            mlp.SetActivationFunctionOptimized(enumActivationFunctionOptimized.Sigmoid, gain)
            mlp.InitializeStruct({10, 10, 1}, addBiasColumn)
        Else
            mlp.InitializeStruct({10, 20, 1}, addBiasColumn)
        End If

        If sigmoid Then
            mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)
        Else
            mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain)
        End If

        mlp.Randomize(minValue, maxValue)

        mlp.PrintParameters()

        WaitForKeyToStart()

        mlp.Train()

        mlp.TestAllSamples(mlp.inputArrayTest, mlp.targetArrayTest, nbOutputs:=1)
        mlp.PrintSuccessPrediction()

        mlp.ShowMessage(testName & ": Done.")

    End Sub

#End Region

End Module