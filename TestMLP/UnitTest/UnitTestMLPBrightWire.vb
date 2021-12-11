
Imports Perceptron
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode
Imports Microsoft.VisualStudio.TestTools.UnitTesting

Namespace BrightWireMLP

    <TestClass()>
    Public Class clsMLPBrightWireTest

        Private m_mlp As New clsMLPBrightWire

        <TestMethod()>
        Public Sub BWMLP1XORSigmoid231()

            InitXOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.InitializeStruct({2, 3, 1}, addBiasColumn:=True)
            m_mlp.nbIterationsBatch = 10

            m_mlp.nbIterations = 2000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {0.02, 0.03, 0.00},
                {0.03, -0.03, 0.00},
                {0.04, 0.05, 0.00}})
            m_mlp.InitializeWeights(2, {
               {-0.03, -0.04, 0.1, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.03
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub BWMLP1XORSigmoid241()

            InitXOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.InitializeStruct({2, 4, 1}, addBiasColumn:=True)
            m_mlp.nbIterationsBatch = 4

            m_mlp.nbIterations = 600
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {0.00, 0.1, 0.00},
                {0.01, 0.00, 0.00},
                {-0.05, 0.07, 0.00},
                {0.00, -0.12, 0.00}})
            m_mlp.InitializeWeights(2, {
               {0.05, 0.1, 0.04, 0.01, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.03
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub BWMLP1XORSigmoid2331()

            InitXOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.InitializeStruct({2, 3, 3, 1}, addBiasColumn:=True)
            m_mlp.nbIterationsBatch = 10

            m_mlp.nbIterations = 6000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {0.08, -0.02, 0.00},
                {-0.07, -0.03, 0.00},
                {-0.09, -0.05, 0.00}})
            m_mlp.InitializeWeights(2, {
                {-0.07, 0.06, 0.03, 0.00},
                {0.01, -0.04, 0.05, 0.00},
                {-0.03, 0.03, -0.07, 0.00}})
            m_mlp.InitializeWeights(3, {
               {0.12, 0.07, 0.08, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub BWMLP1XORSigmoid2441()

            InitXOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.InitializeStruct({2, 4, 4, 1}, addBiasColumn:=True)
            m_mlp.nbIterationsBatch = 4

            m_mlp.nbIterations = 600
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {-0.05, 0.02, 0.00},
                {-0.07, 0.06, 0.00},
                {0.1, -0.06, 0.00},
                {0.07, -0.02, 0.00}})
            m_mlp.InitializeWeights(2, {
                {0.00, -0.02, -0.01, -0.03, 0.00},
                {0.05, 0.08, -0.02, 0.04, 0.00},
                {0.02, 0.05, 0.01, -0.02, 0.00},
                {0.04, -0.09, -0.05, -0.07, 0.00}})
            m_mlp.InitializeWeights(3, {
               {0.05, -0.01, 0.09, -0.01, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.04
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub BWMLP1XORSigmoid25551()

            InitXOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.InitializeStruct({2, 5, 5, 5, 1}, addBiasColumn:=True)
            m_mlp.nbIterationsBatch = 10

            m_mlp.nbIterations = 26000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {-0.01, 0.09, 0.00},
                {0.01, -0.03, 0.00},
                {-0.03, 0.00, 0.00},
                {0.02, -0.08, 0.00},
                {-0.05, 0.08, 0.00}})
            m_mlp.InitializeWeights(2, {
                {0.05, 0.07, 0.05, 0.08, 0.03, 0.00},
                {0.03, -0.01, -0.04, 0.06, 0.09, 0.00},
                {-0.02, 0.02, 0.04, 0.01, 0.04, 0.00},
                {-0.02, 0.03, -0.07, -0.15, 0.02, 0.00},
                {-0.07, -0.09, -0.01, -0.06, 0.01, 0.00}})
            m_mlp.InitializeWeights(3, {
                {-0.03, 0.01, -0.03, 0.01, -0.05, 0.00},
                {0.00, -0.02, 0.00, 0.03, 0.03, 0.00},
                {0.06, -0.1, 0.03, 0.03, 0.01, 0.00},
                {0.00, 0.09, 0.09, 0.01, 0.02, 0.00},
                {0.02, 0.06, -0.08, 0.05, -0.01, 0.00}})
            m_mlp.InitializeWeights(4, {
               {0.01, -0.03, 0.01, -0.05, 0.09, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub BWMLP1XORTanh231()

            InitXOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.InitializeStruct({2, 3, 1}, addBiasColumn:=True)
            m_mlp.nbIterationsBatch = 10

            m_mlp.nbIterations = 3000
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent)

            m_mlp.InitializeWeights(1, {
                {0.00, -0.02, 0.00},
                {0.06, 0.05, 0.00},
                {0.05, 0.00, 0.00}})
            m_mlp.InitializeWeights(2, {
               {0.04, -0.03, -0.01, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.04
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub BWMLP1XORTanh241AdaGrad()

            InitXOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.InitializeStruct({2, 4, 1}, addBiasColumn:=True)
            m_mlp.nbIterationsBatch = 4

            m_mlp.nbIterations = 100
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent)

            m_mlp.InitializeWeights(1, {
                {0.00, -0.08, 0.00},
                {0.05, -0.08, 0.00},
                {-0.04, -0.12, 0.00},
                {0.05, 0.05, 0.00}})
            m_mlp.InitializeWeights(2, {
               {-0.04, 0.06, 0.05, -0.04, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.25
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub BWMLP1XORTanh261()

            InitXOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.InitializeStruct({2, 6, 1}, addBiasColumn:=True)
            m_mlp.nbIterationsBatch = 30

            m_mlp.nbIterations = 700
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent)

            m_mlp.InitializeWeights(1, {
                {0.02, 0.1, 0.00},
                {0.02, 0.1, 0.00},
                {0.07, 0.05, 0.00},
                {-0.11, 0.00, 0.00},
                {-0.05, -0.06, 0.00},
                {0.07, -0.05, 0.00}})
            m_mlp.InitializeWeights(2, {
               {0.00, 0.05, -0.01, -0.02, 0.03, -0.03, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub BWMLP1XORRelu231()

            InitXOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.InitializeStruct({2, 3, 1}, addBiasColumn:=True)
            m_mlp.nbIterationsBatch = 4

            m_mlp.nbIterations = 100
            m_mlp.SetActivationFunction(enumActivationFunction.ReLu)

            m_mlp.InitializeWeights(1, {
                {0.00, 0.14, 0.00},
                {0.07, 0.07, 0.00},
                {0.00, -0.05, 0.00}})
            m_mlp.InitializeWeights(2, {
               {-0.07, 0.01, 0.04, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub BWMLP1XORRelu231b()

            InitXOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.InitializeStruct({2, 3, 1}, addBiasColumn:=True)
            m_mlp.nbIterationsBatch = 4

            m_mlp.nbIterations = 60
            m_mlp.SetActivationFunction(enumActivationFunction.ReLu)

            m_mlp.InitializeWeights(1, {
                {0.00, 0.05, 0.00},
                {-0.01, 0.06, 0.00},
                {-0.04, -0.02, 0.00}})
            m_mlp.InitializeWeights(2, {
               {0.08, -0.03, -0.04, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub BWMLP1XORRelu251()

            InitXOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.InitializeStruct({2, 5, 1}, addBiasColumn:=True)
            m_mlp.nbIterationsBatch = 4

            m_mlp.nbIterations = 30
            m_mlp.SetActivationFunction(enumActivationFunction.ReLu)

            m_mlp.InitializeWeights(1, {
                {-0.13, 0.00, 0.00},
                {0.06, -0.06, 0.00},
                {0.06, -0.05, 0.00},
                {0.03, 0.03, 0.00},
                {-0.07, -0.05, 0.00}})
            m_mlp.InitializeWeights(2, {
               {0.02, 0.02, -0.02, -0.06, 0.07, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArrayXOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub BWMLP2XORSigmoid()

            Init2XOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.InitializeStruct({4, 9, 2}, addBiasColumn:=True)
            m_mlp.nbIterationsBatch = 20

            m_mlp.nbIterations = 3000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {0.07, -0.04, 0.05, -0.04, 0.00},
                {0.01, -0.06, 0.08, 0.05, 0.00},
                {-0.03, -0.08, -0.01, -0.01, 0.00},
                {-0.01, 0.00, 0.05, 0.00, 0.00},
                {0.05, -0.09, -0.03, 0.04, 0.00},
                {0.04, 0.04, -0.05, 0.00, 0.00},
                {-0.03, -0.04, 0.06, 0.08, 0.00},
                {0.00, 0.03, -0.03, 0.01, 0.00},
                {-0.03, -0.11, 0.01, 0.04, 0.00}})
            m_mlp.InitializeWeights(2, {
                {0.02, 0.01, 0.02, 0.01, 0.02, -0.01, 0.03, 0.03, 0.01, 0.00},
                {0.01, 0.04, 0.05, 0.00, 0.00, 0.03, 0.01, -0.01, 0.02, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArray2XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        Public Sub BWMLP2XORTanh()

            Init2XOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.InitializeStruct({4, 9, 2}, addBiasColumn:=True)
            m_mlp.nbIterationsBatch = 20

            m_mlp.nbIterations = 1700
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent)

            m_mlp.InitializeWeights(1, {
                {0.06, -0.03, -0.04, -0.11, 0.00},
                {-0.07, -0.02, 0.06, 0.01, 0.00},
                {0.00, -0.09, 0.01, -0.05, 0.00},
                {0.06, -0.03, 0.02, 0.03, 0.00},
                {-0.08, -0.04, 0.08, -0.03, 0.00},
                {0.03, 0.02, 0.09, -0.06, 0.00},
                {-0.01, -0.02, 0.07, 0.04, 0.00},
                {0.06, 0.00, 0.02, -0.01, 0.00},
                {0.00, 0.1, 0.00, 0.01, 0.00}})
            m_mlp.InitializeWeights(2, {
                {0.02, 0.01, 0.04, 0.03, 0.02, 0.05, 0.03, 0.04, 0.01, 0.00},
                {0.04, 0.01, 0.01, -0.02, -0.05, 0.04, -0.04, -0.03, 0.00, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArray2XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub BWMLP2XORRelu()

            Init2XOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.InitializeStruct({4, 9, 2}, addBiasColumn:=True)
            m_mlp.nbIterationsBatch = 20

            m_mlp.nbIterations = 700
            m_mlp.SetActivationFunction(enumActivationFunction.ReLu)

            m_mlp.InitializeWeights(1, {
                {-0.04, 0.04, 0.07, -0.05, 0.00},
                {-0.05, -0.06, 0.02, -0.04, 0.00},
                {-0.01, -0.07, 0.01, 0.05, 0.00},
                {-0.05, 0.07, -0.01, -0.11, 0.00},
                {0.07, 0.02, 0.02, 0.02, 0.00},
                {0.02, -0.05, 0.07, 0.04, 0.00},
                {0.00, 0.08, 0.01, -0.05, 0.00},
                {-0.01, 0.02, 0.01, -0.07, 0.00},
                {-0.01, 0.02, -0.01, -0.01, 0.00}})
            m_mlp.InitializeWeights(2, {
                {0.00, -0.06, -0.08, 0.04, 0.01, 0.02, 0.03, 0.02, 0.02, 0.00},
                {0.03, 0.00, 0.00, -0.04, -0.03, -0.05, 0.01, 0.05, 0.07, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArray2XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub BWMLP3XORRelu()

            Init3XOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.InitializeStruct({6, 14, 3}, addBiasColumn:=True)
            m_mlp.nbIterationsBatch = 20

            m_mlp.nbIterations = 3000
            m_mlp.SetActivationFunction(enumActivationFunction.ReLu)

            m_mlp.InitializeWeights(1, {
                {0.06, 0.00, -0.03, -0.02, -0.01, -0.03, 0.00},
                {0.11, 0.01, -0.03, 0.04, 0.02, -0.04, 0.00},
                {-0.07, -0.05, 0.02, -0.04, -0.09, 0.07, 0.00},
                {-0.04, 0.06, 0.07, 0.04, 0.04, -0.01, 0.00},
                {-0.02, 0.01, 0.02, -0.03, 0.04, -0.01, 0.00},
                {-0.01, -0.01, -0.05, -0.01, -0.04, 0.04, 0.00},
                {0.05, -0.05, -0.03, 0.02, 0.03, -0.04, 0.00},
                {0.00, 0.06, 0.08, 0.02, 0.05, -0.04, 0.00},
                {0.02, -0.05, -0.04, 0.05, -0.06, 0.00, 0.00},
                {0.02, -0.07, -0.06, 0.00, -0.02, 0.05, 0.00},
                {0.01, 0.00, 0.03, -0.01, -0.05, 0.00, 0.00},
                {0.01, -0.02, 0.05, -0.03, -0.01, -0.01, 0.00},
                {0.07, -0.01, -0.05, -0.02, 0.00, -0.05, 0.00},
                {-0.04, -0.01, -0.04, -0.06, 0.03, 0.01, 0.00}})
            m_mlp.InitializeWeights(2, {
                {0.00, 0.05, -0.01, -0.01, 0.02, -0.03, -0.01, -0.01, 0.00, -0.02, -0.01, 0.01, 0.00, -0.02, 0.00},
                {0.02, 0.00, 0.01, -0.01, -0.01, 0.00, -0.01, 0.02, 0.01, -0.04, 0.01, 0.00, 0.01, -0.01, 0.00},
                {0.01, -0.01, 0.00, 0.04, 0.01, -0.03, -0.03, -0.03, 0.04, 0.04, -0.02, -0.03, 0.00, -0.08, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub BWMLP3XORSigmoidAdam()

            Init3XOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.InitializeStruct({6, 12, 3}, addBiasColumn:=True)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.Adam
            m_mlp.nbIterationsBatch = 100

            m_mlp.nbIterations = 9000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain:=2)

            m_mlp.InitializeWeights(1, {
                {-0.02, 0.02, 0.08, 0.04, 0.04, -0.02, 0.00},
                {-0.05, -0.02, 0.05, -0.04, -0.04, 0.00, 0.00},
                {-0.01, -0.01, 0.01, 0.02, 0.01, 0.03, 0.00},
                {0.05, -0.01, 0.01, 0.04, 0.05, -0.07, 0.00},
                {-0.01, -0.02, 0.03, -0.03, 0.03, -0.1, 0.00},
                {-0.06, 0.06, 0.00, -0.02, 0.03, 0.05, 0.00},
                {-0.03, -0.01, -0.01, 0.05, 0.00, 0.01, 0.00},
                {-0.01, 0.01, 0.00, -0.01, -0.04, -0.02, 0.00},
                {0.03, -0.06, 0.03, -0.03, 0.02, -0.03, 0.00},
                {0.03, 0.01, 0.04, -0.02, 0.02, -0.08, 0.00},
                {0.03, -0.03, 0.03, 0.03, 0.00, -0.01, 0.00},
                {-0.05, 0.01, 0.00, 0.06, 0.06, 0.02, 0.00}})
            m_mlp.InitializeWeights(2, {
                {-0.07, -0.01, -0.03, -0.02, -0.01, 0.04, -0.03, 0.03, 0.04, 0.01, -0.03, 0.01, 0.00},
                {0.00, 0.02, 0.04, 0.05, -0.02, 0.03, -0.01, -0.03, -0.02, -0.02, -0.01, 0.02, 0.00},
                {0.01, -0.04, -0.01, 0.00, -0.03, -0.03, -0.02, -0.01, 0.06, -0.06, -0.02, 0.04, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub BWMLP3XORSigmoid1()

            Init3XOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.InitializeStruct({6, 16, 3}, addBiasColumn:=True)
            m_mlp.nbIterationsBatch = 100

            m_mlp.nbIterations = 8000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {-0.03, -0.05, -0.02, -0.08, -0.02, -0.02, 0.00},
                {-0.06, -0.02, 0.01, -0.02, 0.04, -0.01, 0.00},
                {-0.07, -0.01, 0.02, 0.03, 0.04, 0.03, 0.00},
                {-0.01, -0.01, 0.05, 0.02, 0.00, -0.02, 0.00},
                {0.00, -0.03, -0.01, -0.06, -0.08, -0.05, 0.00},
                {-0.09, 0.02, -0.01, -0.06, -0.04, 0.04, 0.00},
                {-0.04, 0.13, -0.03, -0.05, 0.02, -0.02, 0.00},
                {-0.06, 0.03, -0.02, -0.02, 0.05, 0.04, 0.00},
                {-0.05, -0.02, 0.00, 0.02, -0.02, -0.04, 0.00},
                {0.00, 0.03, 0.04, -0.04, 0.00, 0.03, 0.00},
                {-0.04, 0.05, -0.01, 0.03, 0.00, 0.01, 0.00},
                {-0.01, 0.05, 0.01, 0.01, 0.04, 0.01, 0.00},
                {0.04, -0.01, 0.08, 0.03, -0.05, 0.01, 0.00},
                {0.00, 0.03, 0.02, 0.00, 0.07, 0.03, 0.00},
                {0.00, 0.05, -0.06, -0.02, -0.01, 0.00, 0.00},
                {-0.01, 0.1, 0.02, -0.07, 0.00, -0.07, 0.00}})
            m_mlp.InitializeWeights(2, {
                {0.00, 0.01, -0.01, -0.03, 0.03, -0.03, 0.00, -0.02, -0.01, 0.02, 0.00, 0.04, 0.00, -0.02, -0.04, 0.02, 0.00},
                {0.00, 0.02, 0.01, 0.02, 0.02, 0.00, 0.00, 0.04, 0.03, -0.01, 0.01, 0.04, 0.03, -0.02, 0.00, 0.03, 0.00},
                {-0.02, 0.05, 0.03, 0.00, -0.04, -0.02, 0.00, 0.03, -0.04, 0.02, 0.01, -0.01, 0.02, 0.00, 0.01, 0.01, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub BWMLP3XORSigmoid2()

            Init3XOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.InitializeStruct({6, 14, 3}, addBiasColumn:=True)
            m_mlp.nbIterationsBatch = 30

            m_mlp.nbIterations = 4000
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {0.13, 0.01, 0.01, -0.01, 0.01, -0.06, 0.00},
                {0.04, 0.02, -0.01, -0.03, 0.02, 0.02, 0.00},
                {-0.05, 0.06, 0.03, -0.06, 0.06, 0.01, 0.00},
                {-0.08, 0.03, -0.02, 0.02, 0.01, -0.04, 0.00},
                {0.02, 0.06, 0.03, -0.01, 0.00, -0.05, 0.00},
                {0.01, -0.05, 0.00, 0.00, -0.01, -0.02, 0.00},
                {-0.04, 0.01, 0.03, 0.01, 0.03, 0.04, 0.00},
                {0.04, -0.05, -0.06, -0.03, -0.08, -0.06, 0.00},
                {0.06, -0.04, -0.03, -0.03, 0.02, -0.04, 0.00},
                {-0.01, 0.00, 0.01, 0.02, -0.05, 0.00, 0.00},
                {0.00, 0.03, -0.03, 0.05, 0.00, -0.09, 0.00},
                {0.06, 0.06, 0.04, 0.00, 0.04, 0.03, 0.00},
                {0.03, 0.03, -0.01, 0.06, 0.02, -0.04, 0.00},
                {-0.07, -0.05, -0.05, 0.01, 0.06, -0.02, 0.00}})
            m_mlp.InitializeWeights(2, {
                {-0.06, 0.03, 0.00, -0.03, 0.03, -0.01, -0.03, 0.02, -0.01, 0.04, 0.00, -0.03, 0.02, 0.03, 0.00},
                {-0.02, -0.01, 0.00, 0.02, 0.00, -0.01, -0.01, 0.00, 0.05, -0.03, 0.00, -0.01, 0.05, 0.05, 0.00},
                {-0.02, 0.01, -0.02, -0.02, 0.04, 0.00, 0.01, 0.00, -0.02, -0.01, 0.04, -0.05, -0.01, 0.02, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.01
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub BWMLP3XORTanh()

            Init3XOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.InitializeStruct({6, 20, 3}, addBiasColumn:=True)
            m_mlp.nbIterationsBatch = 200

            m_mlp.nbIterations = 6000
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {0.03, 0.02, 0.04, 0.00, 0.04, -0.09, 0.00},
                {-0.02, -0.01, 0.02, -0.01, 0.08, 0.06, 0.00},
                {-0.04, 0.08, -0.04, -0.03, 0.04, -0.1, 0.00},
                {0.04, 0.03, 0.02, 0.04, 0.00, 0.02, 0.00},
                {-0.03, 0.00, 0.03, 0.05, -0.02, 0.00, 0.00},
                {0.02, 0.01, -0.05, 0.01, 0.00, -0.07, 0.00},
                {0.02, 0.04, -0.04, -0.01, 0.00, 0.01, 0.00},
                {0.03, -0.04, 0.03, -0.02, -0.11, -0.04, 0.00},
                {-0.01, -0.01, 0.03, 0.01, 0.03, 0.03, 0.00},
                {0.05, 0.04, 0.00, -0.01, 0.00, -0.01, 0.00},
                {-0.01, -0.01, 0.01, 0.01, 0.05, 0.04, 0.00},
                {-0.02, 0.01, 0.07, -0.01, -0.04, 0.05, 0.00},
                {0.01, 0.05, -0.02, 0.06, 0.08, 0.04, 0.00},
                {-0.09, 0.03, -0.04, 0.03, -0.04, 0.06, 0.00},
                {0.03, 0.05, 0.02, -0.05, 0.00, 0.1, 0.00},
                {-0.02, 0.01, 0.02, 0.00, -0.02, 0.04, 0.00},
                {-0.01, 0.05, 0.05, -0.07, 0.02, -0.01, 0.00},
                {0.03, -0.01, -0.07, -0.05, 0.02, 0.02, 0.00},
                {0.02, -0.02, -0.04, -0.03, -0.05, -0.12, 0.00},
                {0.01, 0.02, -0.02, -0.06, 0.01, -0.01, 0.00}})
            m_mlp.InitializeWeights(2, {
                {0.05, -0.02, 0.07, -0.03, 0.01, 0.00, 0.05, -0.03, -0.06, 0.01, 0.00, -0.02, 0.01, 0.00, 0.01, -0.01, -0.01, 0.01, 0.01, -0.01, 0.00},
                {0.00, 0.01, 0.00, 0.02, -0.01, 0.01, 0.02, 0.07, -0.04, -0.01, 0.01, 0.02, 0.00, 0.01, 0.05, 0.02, -0.05, -0.03, 0.00, -0.01, 0.00},
                {-0.04, -0.01, -0.05, 0.02, 0.00, 0.01, -0.02, -0.02, 0.04, -0.01, -0.05, -0.03, 0.05, -0.02, -0.02, 0.01, -0.01, 0.05, -0.03, -0.02, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub BWMLP3XORTanhNesterovMomentum1()

            Init3XOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.InitializeStruct({6, 20, 3}, addBiasColumn:=True)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.NesterovMomentum
            m_mlp.nbIterationsBatch = 200

            m_mlp.nbIterations = 6000
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {-0.03, 0.05, -0.07, 0.05, -0.02, -0.01, 0.00},
                {0.01, -0.03, 0.05, -0.01, -0.05, -0.05, 0.00},
                {-0.08, 0.05, -0.03, -0.03, -0.08, 0.00, 0.00},
                {-0.03, 0.01, 0.03, -0.04, -0.04, -0.06, 0.00},
                {-0.01, -0.01, 0.09, 0.04, -0.04, 0.04, 0.00},
                {0.01, -0.06, -0.05, 0.01, 0.00, 0.02, 0.00},
                {0.03, 0.01, -0.05, 0.01, 0.05, 0.06, 0.00},
                {0.04, -0.02, -0.05, -0.03, -0.05, 0.03, 0.00},
                {0.02, 0.06, 0.03, -0.04, -0.03, 0.02, 0.00},
                {0.03, 0.06, -0.02, -0.06, -0.02, 0.01, 0.00},
                {0.03, 0.05, -0.05, 0.01, -0.05, -0.07, 0.00},
                {0.01, 0.03, 0.02, -0.09, -0.08, 0.01, 0.00},
                {-0.03, -0.03, -0.01, -0.05, 0.03, -0.03, 0.00},
                {-0.01, -0.03, -0.06, -0.04, -0.12, -0.04, 0.00},
                {-0.08, -0.05, -0.01, 0.03, -0.01, -0.07, 0.00},
                {0.02, 0.09, 0.03, -0.07, 0.02, -0.01, 0.00},
                {0.06, -0.02, 0.00, 0.04, 0.05, -0.02, 0.00},
                {0.01, 0.02, 0.02, 0.02, -0.05, 0.07, 0.00},
                {-0.08, -0.05, 0.00, 0.00, 0.07, -0.03, 0.00},
                {-0.03, 0.06, 0.04, 0.01, -0.01, 0.09, 0.00}})
            m_mlp.InitializeWeights(2, {
                {-0.02, -0.02, -0.03, -0.02, 0.01, -0.01, 0.00, 0.03, -0.01, -0.04, 0.03, 0.01, 0.01, 0.02, 0.00, -0.04, 0.02, 0.00, -0.02, -0.03, 0.00},
                {0.00, -0.01, 0.04, 0.03, -0.05, 0.00, -0.06, -0.02, 0.00, 0.00, -0.01, -0.03, 0.01, -0.02, 0.01, -0.01, 0.00, -0.01, 0.01, -0.01, 0.00},
                {-0.02, -0.03, 0.01, -0.02, -0.01, -0.01, -0.01, 0.04, 0.05, 0.01, 0.01, 0.01, 0.02, 0.00, -0.01, -0.02, 0.00, -0.02, 0.01, 0.00, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub BWMLP3XORTanhNesterovMomentum2()

            Init3XOR(m_mlp)
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.InitializeStruct({6, 15, 3}, addBiasColumn:=True)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.NesterovMomentum
            m_mlp.nbIterationsBatch = 200

            m_mlp.nbIterations = 7000
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {0.04, -0.04, -0.03, -0.03, 0.09, 0.04, 0.00},
                {-0.03, -0.04, 0.03, 0.1, 0.09, 0.06, 0.00},
                {-0.02, 0.01, -0.01, -0.01, -0.07, 0.00, 0.00},
                {-0.02, 0.01, -0.04, -0.13, 0.01, 0.00, 0.00},
                {-0.04, -0.03, 0.00, -0.04, 0.06, 0.01, 0.00},
                {0.00, 0.06, 0.08, 0.01, 0.03, -0.05, 0.00},
                {0.02, 0.03, 0.07, -0.01, 0.06, -0.09, 0.00},
                {0.05, -0.02, -0.04, -0.03, 0.06, 0.03, 0.00},
                {-0.03, -0.02, -0.06, -0.07, 0.02, 0.01, 0.00},
                {0.01, -0.01, -0.01, -0.03, -0.04, -0.02, 0.00},
                {-0.02, -0.01, -0.05, 0.03, 0.01, 0.02, 0.00},
                {0.02, -0.03, 0.04, -0.01, 0.00, 0.05, 0.00},
                {0.00, 0.06, -0.04, 0.02, -0.09, -0.01, 0.00},
                {0.01, -0.01, 0.05, -0.06, 0.01, -0.01, 0.00},
                {-0.07, -0.02, 0.03, 0.03, -0.08, -0.02, 0.00}})
            m_mlp.InitializeWeights(2, {
                {-0.05, -0.02, -0.04, 0.00, 0.02, 0.03, 0.02, 0.01, -0.02, 0.06, 0.00, 0.01, 0.03, -0.03, 0.00, 0.00},
                {0.00, 0.03, -0.03, 0.02, 0.03, -0.02, 0.03, 0.00, 0.00, -0.01, -0.02, -0.07, 0.02, -0.05, 0.02, 0.00},
                {-0.01, 0.02, 0.06, 0.06, 0.01, 0.00, -0.03, -0.01, 0.01, 0.04, -0.04, -0.01, -0.05, 0.04, 0.00, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedOutput = m_targetArray3XOR
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Dim sOutput = m_mlp.output.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)

            Const expectedLoss# = 0.02
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 2)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

        End Sub

        <TestMethod()>
        Public Sub BWMLPIrisFlowerLogicalTanh()

            InitIrisFlowerLogical4Layers(m_mlp)
            m_mlp.InitializeStruct({4, 22, 8, 3}, addBiasColumn:=True)
            m_mlp.inputArrayTest = m_inputArrayIrisFlowerTest
            m_mlp.targetArrayTest = m_targetArrayIrisFlowerLogicalTest
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.RMSProp
            m_mlp.nbIterationsBatch = 10

            m_mlp.nbIterations = 3500 '3300
            m_mlp.minimalSuccessTreshold = 0.3
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2)

            m_mlp.InitializeWeights(1, {
                {0.04, 0.07, 0.01, -0.06, 0.00},
                {-0.05, -0.03, 0.00, 0.01, 0.00},
                {0.01, 0.01, 0.02, 0.05, 0.00},
                {0.01, 0.02, -0.03, -0.05, 0.00},
                {0.03, 0.05, 0.03, 0.04, 0.00},
                {-0.05, 0.09, -0.1, -0.01, 0.00},
                {0.02, 0.05, 0.01, -0.12, 0.00},
                {-0.03, 0.03, 0.01, -0.08, 0.00},
                {-0.04, -0.04, -0.05, -0.01, 0.00},
                {-0.01, -0.02, 0.01, -0.03, 0.00},
                {-0.1, 0.01, -0.06, 0.06, 0.00},
                {0.01, -0.01, 0.05, 0.00, 0.00},
                {-0.03, 0.01, -0.08, -0.08, 0.00},
                {0.01, 0.01, 0.03, -0.03, 0.00},
                {-0.05, 0.01, 0.02, 0.09, 0.00},
                {-0.02, -0.05, 0.02, 0.00, 0.00},
                {0.00, 0.02, -0.01, 0.02, 0.00},
                {0.05, 0.08, 0.03, 0.06, 0.00},
                {-0.07, 0.00, 0.03, 0.15, 0.00},
                {-0.04, 0.04, -0.05, -0.01, 0.00},
                {-0.09, 0.08, -0.08, -0.13, 0.00},
                {-0.07, -0.04, -0.06, -0.11, 0.00}})
            m_mlp.InitializeWeights(2, {
                {0.01, 0.00, 0.00, -0.01, -0.01, -0.01, -0.03, 0.00, 0.00, -0.01, -0.02, 0.03, -0.01, 0.02, 0.04, -0.05, 0.00, 0.00, -0.01, 0.00, 0.01, 0.00, 0.00},
                {0.00, 0.00, 0.00, 0.00, -0.03, 0.00, -0.03, -0.01, -0.01, 0.01, 0.00, -0.01, 0.00, -0.02, 0.03, -0.01, 0.01, -0.01, -0.02, 0.02, 0.01, 0.03, 0.00},
                {0.03, 0.03, -0.02, 0.01, -0.02, 0.03, -0.02, 0.04, 0.02, -0.04, -0.01, 0.04, 0.01, -0.02, 0.03, 0.01, -0.02, 0.03, 0.01, -0.02, 0.01, -0.01, 0.00},
                {0.01, 0.00, 0.01, 0.03, -0.02, 0.03, -0.02, 0.01, 0.00, 0.02, 0.01, 0.03, 0.01, 0.01, 0.04, 0.00, 0.01, 0.00, 0.02, 0.01, -0.02, 0.03, 0.00},
                {-0.03, 0.00, 0.00, -0.02, 0.03, -0.02, 0.00, 0.01, -0.03, 0.00, 0.00, -0.05, -0.01, -0.01, 0.02, -0.01, -0.01, -0.02, -0.04, 0.02, 0.01, 0.00, 0.00},
                {0.01, -0.02, 0.01, -0.02, 0.01, -0.02, 0.02, 0.02, -0.01, 0.02, -0.01, 0.02, -0.02, 0.01, -0.04, -0.01, 0.02, 0.02, 0.02, 0.00, -0.02, 0.02, 0.00},
                {0.01, 0.00, 0.03, 0.02, -0.03, 0.02, 0.01, 0.00, 0.07, 0.01, -0.02, -0.02, 0.02, -0.02, 0.02, -0.01, -0.03, -0.03, 0.00, 0.01, 0.00, 0.00, 0.00},
                {0.00, -0.03, 0.03, 0.05, -0.02, 0.01, 0.02, -0.03, -0.02, -0.01, -0.03, 0.03, -0.01, -0.01, 0.01, -0.02, 0.02, -0.01, 0.00, 0.00, 0.00, 0.02, 0.00}})
            m_mlp.InitializeWeights(3, {
                {0.00, 0.00, 0.02, 0.02, 0.03, -0.08, -0.01, 0.02, 0.00},
                {0.01, -0.02, -0.02, 0.03, 0.03, -0.04, -0.05, 0.04, 0.00},
                {-0.02, 0.02, 0.02, 0.04, -0.03, -0.06, -0.01, -0.06, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedSuccess# = 0.85
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.176
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            m_mlp.TestAllSamples(
                m_mlp.inputArrayTest, m_mlp.targetArrayTest, nbOutputs:=3)
            Dim expectedSuccessPrediction# = 0.978
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub BWMLPIrisFlowerLogicalSigmoid1()

            InitIrisFlowerLogical4Layers(m_mlp)
            m_mlp.inputArrayTest = m_inputArrayIrisFlowerTest
            m_mlp.targetArrayTest = m_targetArrayIrisFlowerLogicalTest
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.RMSProp
            m_mlp.nbIterationsBatch = 10

            m_mlp.nbIterations = 2000
            m_mlp.minimalSuccessTreshold = 0.3
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {0.00, -0.04, -0.03, 0.01, 0.00},
                {0.05, 0.00, 0.06, -0.08, 0.00},
                {-0.1, 0.04, 0.01, -0.04, 0.00},
                {0.08, 0.02, 0.05, -0.05, 0.00},
                {-0.01, -0.04, 0.00, 0.01, 0.00},
                {0.00, -0.03, -0.03, 0.04, 0.00},
                {-0.08, -0.05, -0.05, 0.01, 0.00},
                {0.02, 0.02, -0.05, -0.05, 0.00},
                {-0.03, 0.04, -0.04, 0.06, 0.00},
                {0.05, 0.04, -0.03, 0.14, 0.00},
                {-0.02, 0.08, 0.02, -0.02, 0.00},
                {0.04, -0.06, 0.02, 0.01, 0.00},
                {-0.1, 0.11, 0.01, -0.03, 0.00},
                {-0.01, -0.03, 0.08, -0.02, 0.00},
                {0.01, -0.08, -0.01, 0.06, 0.00},
                {0.05, 0.03, -0.04, 0.02, 0.00}})
            m_mlp.InitializeWeights(2, {
                {0.02, -0.05, 0.00, 0.02, 0.00, 0.01, 0.01, 0.02, -0.03, -0.02, 0.00, 0.02, 0.00, 0.03, -0.05, -0.01, 0.00},
                {0.01, 0.01, 0.03, 0.04, 0.02, 0.03, 0.02, -0.01, 0.00, 0.05, -0.05, -0.02, 0.00, -0.02, 0.00, 0.02, 0.00},
                {-0.01, 0.03, -0.01, -0.05, 0.00, 0.00, 0.04, 0.01, 0.06, 0.01, 0.01, 0.00, 0.00, 0.02, 0.03, 0.00, 0.00},
                {-0.01, 0.02, 0.03, -0.02, 0.03, 0.02, -0.01, 0.03, 0.1, -0.02, 0.00, -0.06, 0.00, 0.03, -0.01, 0.01, 0.00},
                {0.04, 0.00, -0.01, -0.06, -0.04, 0.00, -0.01, 0.00, -0.04, 0.01, 0.02, 0.02, -0.02, 0.01, 0.03, -0.01, 0.00},
                {0.02, 0.01, 0.00, 0.04, -0.04, 0.00, -0.03, 0.03, 0.00, 0.04, -0.02, -0.01, 0.03, -0.01, -0.04, -0.03, 0.00},
                {-0.06, 0.01, -0.04, -0.02, 0.06, -0.01, 0.03, 0.02, -0.02, 0.00, -0.02, 0.02, 0.02, -0.05, -0.02, 0.04, 0.00},
                {-0.01, 0.06, 0.00, 0.01, 0.00, 0.00, 0.00, 0.02, 0.01, 0.01, 0.00, -0.02, 0.00, 0.01, 0.00, 0.01, 0.00}})
            m_mlp.InitializeWeights(3, {
                {0.01, 0.07, -0.03, 0.00, 0.00, 0.01, 0.03, -0.04, 0.00},
                {-0.01, 0.03, -0.05, -0.02, -0.02, -0.03, 0.02, -0.07, 0.00},
                {-0.02, 0.00, 0.00, 0.02, -0.02, -0.01, -0.07, 0.04, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedSuccess# = 0.811
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.2
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            m_mlp.TestAllSamples(
                m_mlp.inputArrayTest, m_mlp.targetArrayTest, nbOutputs:=3)
            Dim expectedSuccessPrediction# = 0.956
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub BWMLPIrisFlowerLogicalSigmoid2()

            InitIrisFlowerLogical4Layers(m_mlp)
            m_mlp.InitializeStruct({4, 22, 6, 3}, addBiasColumn:=True)
            m_mlp.inputArrayTest = m_inputArrayIrisFlowerTest
            m_mlp.targetArrayTest = m_targetArrayIrisFlowerLogicalTest
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.trainingAlgorithm = enumTrainingAlgorithm.RMSProp
            m_mlp.nbIterationsBatch = 10

            m_mlp.nbIterations = 3000
            m_mlp.minimalSuccessTreshold = 0.3
            m_mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

            m_mlp.InitializeWeights(1, {
                {0.05, -0.02, 0.02, 0.01, 0.00},
                {-0.02, -0.09, 0.03, 0.05, 0.00},
                {0.05, 0.00, 0.02, 0.05, 0.00},
                {0.01, -0.02, -0.05, 0.04, 0.00},
                {0.01, -0.02, 0.00, 0.06, 0.00},
                {0.02, -0.04, 0.04, -0.01, 0.00},
                {-0.02, -0.02, 0.00, -0.04, 0.00},
                {0.03, 0.00, 0.01, 0.00, 0.00},
                {-0.03, -0.01, -0.06, 0.00, 0.00},
                {-0.01, 0.00, -0.01, -0.01, 0.00},
                {-0.07, -0.04, -0.01, 0.04, 0.00},
                {-0.01, -0.07, 0.03, 0.04, 0.00},
                {0.05, -0.06, 0.06, -0.05, 0.00},
                {0.1, -0.03, 0.00, -0.02, 0.00},
                {0.06, -0.06, 0.03, -0.01, 0.00},
                {0.08, -0.03, 0.00, 0.03, 0.00},
                {-0.04, 0.03, -0.1, -0.04, 0.00},
                {-0.09, 0.00, 0.00, 0.05, 0.00},
                {0.02, -0.04, -0.03, -0.03, 0.00},
                {0.00, 0.04, -0.08, 0.04, 0.00},
                {-0.08, 0.00, 0.00, -0.01, 0.00},
                {0.11, -0.01, -0.04, 0.01, 0.00}})
            m_mlp.InitializeWeights(2, {
                {0.01, -0.02, -0.04, 0.02, -0.01, 0.00, -0.02, -0.03, 0.02, -0.03, 0.02, -0.02, 0.03, 0.04, -0.02, 0.01, 0.00, 0.00, 0.01, 0.03, 0.02, -0.02, 0.00},
                {0.00, 0.01, 0.02, 0.04, -0.02, -0.03, 0.04, 0.00, -0.02, 0.04, 0.01, -0.04, 0.03, 0.00, 0.01, 0.02, 0.03, -0.04, 0.01, 0.00, -0.03, -0.02, 0.00},
                {0.01, 0.01, 0.02, 0.01, 0.03, -0.02, 0.05, 0.03, 0.00, -0.02, -0.01, 0.04, 0.01, -0.02, 0.00, -0.03, 0.04, -0.04, 0.01, 0.04, -0.04, 0.03, 0.00},
                {0.01, 0.00, -0.03, 0.02, 0.01, 0.01, 0.02, 0.01, 0.02, 0.02, -0.01, 0.00, 0.02, -0.01, -0.01, 0.01, -0.01, -0.01, 0.05, 0.02, -0.02, 0.00, 0.00},
                {-0.01, -0.02, -0.02, 0.05, 0.03, 0.03, 0.03, -0.03, -0.03, -0.01, 0.00, -0.03, 0.01, -0.02, -0.01, -0.02, 0.01, 0.00, 0.01, 0.00, 0.01, 0.04, 0.00},
                {-0.02, 0.04, 0.03, 0.00, 0.06, -0.01, -0.01, 0.01, 0.01, 0.01, 0.03, 0.01, 0.02, -0.04, 0.01, -0.07, -0.01, 0.02, -0.01, 0.02, -0.01, 0.01, 0.00}})
            m_mlp.InitializeWeights(3, {
                {-0.04, -0.05, 0.01, 0.06, 0.08, 0.03, 0.00},
                {-0.01, 0.00, 0.04, 0.02, 0.00, -0.01, 0.00},
                {0.01, -0.04, -0.03, -0.02, -0.05, 0.00, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedSuccess# = 0.81
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.18
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            m_mlp.TestAllSamples(
                m_mlp.inputArrayTest, m_mlp.targetArrayTest, nbOutputs:=3)
            Dim expectedSuccessPrediction# = 0.956
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub BWMLPIrisFlowerAnalog()

            InitIrisFlowerAnalog4Layers(m_mlp)
            m_mlp.InitializeStruct({4, 20, 10, 1}, addBiasColumn:=True)
            m_mlp.inputArrayTest = m_inputArrayIrisFlowerTest
            m_mlp.targetArrayTest = m_targetArrayIrisFlowerAnalogTest
            m_mlp.Initialize(learningRate:=0.1!)
            m_mlp.nbIterationsBatch = 10

            m_mlp.nbIterations = 4000
            m_mlp.minimalSuccessTreshold = 0.2
            m_mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain:=2.0!)

            m_mlp.InitializeWeights(1, {
                {0.03, -0.01, 0.07, 0.07, 0.00},
                {-0.05, 0.04, 0.07, 0.03, 0.00},
                {-0.07, -0.02, 0.02, -0.03, 0.00},
                {0.02, 0.04, -0.05, 0.00, 0.00},
                {0.02, 0.1, 0.00, 0.09, 0.00},
                {0.04, 0.02, 0.06, 0.03, 0.00},
                {-0.08, 0.00, -0.01, -0.04, 0.00},
                {0.02, 0.01, -0.03, -0.02, 0.00},
                {-0.09, 0.01, 0.06, -0.08, 0.00},
                {0.09, -0.01, 0.04, 0.00, 0.00},
                {0.09, 0.06, 0.01, -0.1, 0.00},
                {0.00, 0.02, -0.01, 0.02, 0.00},
                {0.04, 0.02, 0.02, 0.01, 0.00},
                {-0.02, 0.08, -0.03, -0.1, 0.00},
                {0.01, -0.07, -0.06, 0.01, 0.00},
                {0.09, 0.05, -0.06, 0.02, 0.00},
                {0.13, 0.02, -0.01, -0.07, 0.00},
                {0.04, 0.08, 0.05, -0.03, 0.00},
                {-0.04, -0.03, 0.01, -0.03, 0.00},
                {-0.12, 0.09, 0.06, 0.03, 0.00}})
            m_mlp.InitializeWeights(2, {
                {0.00, 0.01, -0.05, -0.06, 0.01, 0.01, 0.00, 0.00, -0.04, -0.01, 0.02, -0.03, 0.00, -0.03, -0.01, 0.02, -0.01, 0.02, 0.02, 0.00, 0.00},
                {0.00, 0.06, 0.00, -0.01, 0.00, -0.02, 0.03, -0.01, 0.00, 0.06, 0.01, -0.05, 0.04, 0.02, -0.04, -0.02, -0.01, -0.03, 0.02, 0.01, 0.00},
                {0.03, -0.02, -0.02, 0.01, 0.04, -0.01, -0.01, 0.04, -0.01, 0.03, -0.02, 0.04, 0.00, 0.01, -0.01, 0.01, 0.00, -0.03, 0.00, 0.02, 0.00},
                {0.01, 0.03, -0.05, 0.00, -0.01, 0.04, -0.03, 0.01, -0.01, -0.02, -0.01, 0.00, 0.01, 0.01, 0.02, -0.01, 0.00, -0.02, 0.00, -0.01, 0.00},
                {-0.01, 0.03, -0.01, 0.04, 0.02, 0.01, 0.01, -0.01, 0.03, 0.03, 0.00, -0.02, 0.00, 0.01, -0.01, 0.00, -0.03, 0.00, 0.00, -0.02, 0.00},
                {0.03, 0.01, -0.02, 0.01, -0.06, 0.01, -0.04, -0.02, -0.02, -0.01, 0.03, 0.03, 0.03, 0.01, 0.00, 0.00, 0.01, -0.01, 0.02, 0.01, 0.00},
                {0.00, 0.02, 0.00, 0.00, -0.01, 0.02, -0.01, 0.01, 0.01, -0.06, -0.05, 0.01, 0.04, -0.01, -0.06, 0.00, -0.01, 0.00, 0.02, -0.02, 0.00},
                {-0.01, -0.03, 0.02, -0.03, 0.04, 0.04, 0.01, -0.02, 0.02, -0.02, 0.01, 0.03, -0.06, 0.00, 0.02, 0.02, 0.02, 0.00, 0.03, -0.01, 0.00},
                {-0.02, -0.02, 0.00, -0.01, -0.05, 0.02, -0.01, -0.01, -0.02, -0.03, -0.01, -0.03, -0.02, 0.04, 0.01, -0.02, -0.02, 0.02, 0.01, -0.01, 0.00},
                {0.02, 0.03, -0.01, 0.02, -0.02, 0.01, -0.01, -0.01, 0.01, 0.03, -0.02, -0.02, -0.01, 0.00, 0.04, 0.01, 0.00, -0.01, -0.04, 0.03, 0.00}})
            m_mlp.InitializeWeights(3, {
                {0.00, 0.02, 0.03, 0.01, 0.04, 0.01, 0.03, 0.01, 0.02, -0.03, 0.00}})

            m_mlp.Train(learningMode:=enumLearningMode.VectorialBatch)

            Dim expectedSuccess# = 0.75
            Dim success! = m_mlp.successPC
            Dim successRounded# = Math.Round(success, 3)
            Assert.AreEqual(True, successRounded >= expectedSuccess)

            Const expectedLoss# = 0.17
            Dim loss# = m_mlp.averageError
            Dim lossRounded# = Math.Round(loss, 3)
            Assert.AreEqual(True, lossRounded <= expectedLoss)

            m_mlp.TestAllSamples(m_mlp.inputArrayTest, m_mlp.targetArrayTest, nbOutputs:=1)
            Dim expectedSuccessPrediction# = 0.933
            Dim successPrediction! = m_mlp.successPC
            Dim successPredictionRounded# = Math.Round(successPrediction, 3)
            Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

        End Sub

        <TestMethod()>
        Public Sub BWMLPSunspot1Tanh()

            TestMLPSunspot1Tanh(m_mlp, nbIterations:=3000,
                expectedSuccess:=0.1, expectedSuccessPrediction:=0.8, expectedLoss:=0.2,
                learningMode:=enumLearningMode.VectorialBatch)

        End Sub

    End Class

End Namespace