
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode
Imports Microsoft.VisualStudio.TestTools.UnitTesting ' Assert.AreEqual

Module modMLPTest

#Region "Initialization"

#Region "Iris data set"

    ' Iris flower data set
    ' https://en.wikipedia.org/wiki/Iris_flower_data_set
    ' http://archive.ics.uci.edu/ml/datasets/Iris
    ' 1. sepal length in cm
    ' 2. sepal width in cm
    ' 3. petal length in cm
    ' 4. petal width in cm
    ' 5. class:
    '    Analog data set: a single output neuron for all three classes:
    '    0   : -- Iris Setosa
    '    0.5 : -- Iris Versicolour
    '    1   : -- Iris Virginica
    '    Logical data set: one output neuron for each class, so three output neurons:
    '    100 : -- Iris Setosa
    '    010 : -- Iris Versicolour
    '    001 : -- Iris Virginica

    Public ReadOnly m_neuronCountIrisAnalog%() = {4, 16, 16, 1}
    Public ReadOnly m_neuronCountIrisAnalog451%() = {4, 5, 1}
    Public ReadOnly m_neuronCountIrisAnalog4_20_1%() = {4, 20, 1}
    Public ReadOnly m_neuronCountIrisLogical%() = {4, 16, 16, 3}
    Public ReadOnly m_neuronCountIrisLogical443%() = {4, 4, 3}
    Public ReadOnly m_neuronCountIrisLogical4_20_3%() = {4, 20, 3}
    Public ReadOnly m_inputArrayIris!(,) = {
        {5.1, 3.5, 1.4, 0.2},
        {4.9, 3, 1.4, 0.2},
        {4.7, 3.2, 1.3, 0.2},
        {4.6, 3.1, 1.5, 0.2},
        {5, 3.6, 1.4, 0.2},
        {5.4, 3.9, 1.7, 0.4},
        {4.6, 3.4, 1.4, 0.3},
        {5, 3.4, 1.5, 0.2},
        {4.4, 2.9, 1.4, 0.2},
        {4.9, 3.1, 1.5, 0.1},
        {5.4, 3.7, 1.5, 0.2},
        {4.8, 3.4, 1.6, 0.2},
        {4.8, 3, 1.4, 0.1},
        {4.3, 3, 1.1, 0.1},
        {5.8, 4, 1.2, 0.2},
        {5.7, 4.4, 1.5, 0.4},
        {5.4, 3.9, 1.3, 0.4},
        {5.1, 3.5, 1.4, 0.3},
        {5.7, 3.8, 1.7, 0.3},
        {5.1, 3.8, 1.5, 0.3},
        {5.4, 3.4, 1.7, 0.2},
        {5.1, 3.7, 1.5, 0.4},
        {4.6, 3.6, 1, 0.2},
        {5.1, 3.3, 1.7, 0.5},
        {4.8, 3.4, 1.9, 0.2},
        {5, 3, 1.6, 0.2},
        {5, 3.4, 1.6, 0.4},
        {5.2, 3.5, 1.5, 0.2},
        {5.2, 3.4, 1.4, 0.2},
        {4.7, 3.2, 1.6, 0.2},
        {4.8, 3.1, 1.6, 0.2},
        {5.4, 3.4, 1.5, 0.4},
        {5.2, 4.1, 1.5, 0.1},
        {5.5, 4.2, 1.4, 0.2},
        {4.9, 3.1, 1.5, 0.2},
        {5, 3.2, 1.2, 0.2},
        {5.5, 3.5, 1.3, 0.2},
        {4.9, 3.6, 1.4, 0.1},
        {4.4, 3, 1.3, 0.2},
        {5.1, 3.4, 1.5, 0.2},
        {5, 3.5, 1.3, 0.3},
        {4.5, 2.3, 1.3, 0.3},
        {4.4, 3.2, 1.3, 0.2},
        {5, 3.5, 1.6, 0.6},
        {5.1, 3.8, 1.9, 0.4},
        {4.8, 3, 1.4, 0.3},
        {5.1, 3.8, 1.6, 0.2},
        {4.6, 3.2, 1.4, 0.2},
        {5.3, 3.7, 1.5, 0.2},
        {5, 3.3, 1.4, 0.2},
        {7, 3.2, 4.7, 1.4},
        {6.4, 3.2, 4.5, 1.5},
        {6.9, 3.1, 4.9, 1.5},
        {5.5, 2.3, 4, 1.3},
        {6.5, 2.8, 4.6, 1.5},
        {5.7, 2.8, 4.5, 1.3},
        {6.3, 3.3, 4.7, 1.6},
        {4.9, 2.4, 3.3, 1},
        {6.6, 2.9, 4.6, 1.3},
        {5.2, 2.7, 3.9, 1.4},
        {5, 2, 3.5, 1},
        {5.9, 3, 4.2, 1.5},
        {6, 2.2, 4, 1},
        {6.1, 2.9, 4.7, 1.4},
        {5.6, 2.9, 3.6, 1.3},
        {6.7, 3.1, 4.4, 1.4},
        {5.6, 3, 4.5, 1.5},
        {5.8, 2.7, 4.1, 1},
        {6.2, 2.2, 4.5, 1.5},
        {5.6, 2.5, 3.9, 1.1},
        {5.9, 3.2, 4.8, 1.8},
        {6.1, 2.8, 4, 1.3},
        {6.3, 2.5, 4.9, 1.5},
        {6.1, 2.8, 4.7, 1.2},
        {6.4, 2.9, 4.3, 1.3},
        {6.6, 3, 4.4, 1.4},
        {6.8, 2.8, 4.8, 1.4},
        {6.7, 3, 5, 1.7},
        {6, 2.9, 4.5, 1.5},
        {5.7, 2.6, 3.5, 1},
        {5.5, 2.4, 3.8, 1.1},
        {5.5, 2.4, 3.7, 1},
        {5.8, 2.7, 3.9, 1.2},
        {6, 2.7, 5.1, 1.6},
        {5.4, 3, 4.5, 1.5},
        {6, 3.4, 4.5, 1.6},
        {6.7, 3.1, 4.7, 1.5},
        {6.3, 2.3, 4.4, 1.3},
        {5.6, 3, 4.1, 1.3},
        {5.5, 2.5, 4, 1.3},
        {5.5, 2.6, 4.4, 1.2},
        {6.1, 3, 4.6, 1.4},
        {5.8, 2.6, 4, 1.2},
        {5, 2.3, 3.3, 1},
        {5.6, 2.7, 4.2, 1.3},
        {5.7, 3, 4.2, 1.2},
        {5.7, 2.9, 4.2, 1.3},
        {6.2, 2.9, 4.3, 1.3},
        {5.1, 2.5, 3, 1.1},
        {5.7, 2.8, 4.1, 1.3},
        {6.3, 3.3, 6, 2.5},
        {5.8, 2.7, 5.1, 1.9},
        {7.1, 3, 5.9, 2.1},
        {6.3, 2.9, 5.6, 1.8},
        {6.5, 3, 5.8, 2.2},
        {7.6, 3, 6.6, 2.1},
        {4.9, 2.5, 4.5, 1.7},
        {7.3, 2.9, 6.3, 1.8},
        {6.7, 2.5, 5.8, 1.8},
        {7.2, 3.6, 6.1, 2.5},
        {6.5, 3.2, 5.1, 2},
        {6.4, 2.7, 5.3, 1.9},
        {6.8, 3, 5.5, 2.1},
        {5.7, 2.5, 5, 2},
        {5.8, 2.8, 5.1, 2.4},
        {6.4, 3.2, 5.3, 2.3},
        {6.5, 3, 5.5, 1.8},
        {7.7, 3.8, 6.7, 2.2},
        {7.7, 2.6, 6.9, 2.3},
        {6, 2.2, 5, 1.5},
        {6.9, 3.2, 5.7, 2.3},
        {5.6, 2.8, 4.9, 2},
        {7.7, 2.8, 6.7, 2},
        {6.3, 2.7, 4.9, 1.8},
        {6.7, 3.3, 5.7, 2.1},
        {7.2, 3.2, 6, 1.8},
        {6.2, 2.8, 4.8, 1.8},
        {6.1, 3, 4.9, 1.8},
        {6.4, 2.8, 5.6, 2.1},
        {7.2, 3, 5.8, 1.6},
        {7.4, 2.8, 6.1, 1.9},
        {7.9, 3.8, 6.4, 2},
        {6.4, 2.8, 5.6, 2.2},
        {6.3, 2.8, 5.1, 1.5},
        {6.1, 2.6, 5.6, 1.4},
        {7.7, 3, 6.1, 2.3},
        {6.3, 3.4, 5.6, 2.4},
        {6.4, 3.1, 5.5, 1.8},
        {6, 3, 4.8, 1.8},
        {6.9, 3.1, 5.4, 2.1},
        {6.7, 3.1, 5.6, 2.4},
        {6.9, 3.1, 5.1, 2.3},
        {5.8, 2.7, 5.1, 1.9},
        {6.8, 3.2, 5.9, 2.3},
        {6.7, 3.3, 5.7, 2.5},
        {6.7, 3, 5.2, 2.3},
        {6.3, 2.5, 5, 1.9},
        {6.5, 3, 5.2, 2},
        {6.2, 3.4, 5.4, 2.3},
        {5.9, 3, 5.1, 1.8}}

    Public ReadOnly m_targetArrayIrisAnalogUnnormalized!(,) = {
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2},
        {2}}

    Public ReadOnly m_targetArrayIrisAnalog!(,) = {
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {0.5},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1}}

    Public ReadOnly m_targetArrayIrisLogical!(,) = {
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1}}

#End Region

#Region "XOR data set"

    Public ReadOnly m_neuronCountXOR%() = {2, 2, 1}
    Public ReadOnly m_neuronCountXOR231%() = {2, 3, 1} ' With bias
    Public ReadOnly m_neuronCountXOR261%() = {2, 6, 1} ' TensorFlow minimal size
    Public ReadOnly m_neuronCountXOR271%() = {2, 7, 1} ' Keras minimal size
    Public ReadOnly m_neuronCountXOR291%() = {2, 9, 1} ' Keras minimal size for tanh
    Public ReadOnly m_neuronCountXOR2_10_1%() = {2, 10, 1}
    Public ReadOnly m_neuronCountXOR2_16_1%() = {2, 16, 1}

    Public ReadOnly m_neuronCount2XOR%() = {4, 4, 2}
    Public ReadOnly m_neuronCount2XOR452%() = {4, 5, 2}
    Public ReadOnly m_neuronCount2XOR462%() = {4, 6, 2} ' TensorFlow minimal size
    Public ReadOnly m_neuronCount2XOR472%() = {4, 7, 2}
    Public ReadOnly m_neuronCount2XOR482%() = {4, 8, 2}
    Public ReadOnly m_neuronCount2XOR4_10_2%() = {4, 10, 2}
    Public ReadOnly m_neuronCount2XOR4_32_2%() = {4, 32, 2} ' Keras minimal size: stable!
    Public ReadOnly m_neuronCount3XOR%() = {6, 6, 3}
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

    Public ReadOnly m_inputArrayXOR!(,) = {
        {1, 0},
        {0, 0},
        {0, 1},
        {1, 1}}

    Public ReadOnly m_inputArrayXOR90PC!(,) = {
        {0.9!, 0.1!},
        {0.1!, 0.1!},
        {0.1!, 0.9!},
        {0.9!, 0.9!}}

    Public ReadOnly m_inputArrayXOR80PC!(,) = {
        {0.8!, 0.2!},
        {0.2!, 0.2!},
        {0.2!, 0.8!},
        {0.8!, 0.8!}}

    Public ReadOnly m_inputArrayXOR70PC!(,) = {
        {0.7!, 0.3!},
        {0.3!, 0.3!},
        {0.3!, 0.7!},
        {0.7!, 0.7!}}

    Public ReadOnly m_targetArrayXOR!(,) = {
        {1},
        {0},
        {1},
        {0}}

    Public ReadOnly m_inputArray2XOR!(,) = {
        {1, 0, 1, 0},
        {1, 0, 0, 0},
        {1, 0, 0, 1},
        {1, 0, 1, 1},
        {0, 0, 1, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 1},
        {0, 0, 1, 1},
        {0, 1, 1, 0},
        {0, 1, 0, 0},
        {0, 1, 0, 1},
        {0, 1, 1, 1},
        {1, 1, 1, 0},
        {1, 1, 0, 0},
        {1, 1, 0, 1},
        {1, 1, 1, 1}}

    Public ReadOnly m_targetArray2XOR!(,) = {
        {1, 1},
        {1, 0},
        {1, 1},
        {1, 0},
        {0, 1},
        {0, 0},
        {0, 1},
        {0, 0},
        {1, 1},
        {1, 0},
        {1, 1},
        {1, 0},
        {0, 1},
        {0, 0},
        {0, 1},
        {0, 0}}

    Public ReadOnly m_inputArray3XOR!(,) = {
        {1, 0, 1, 0, 1, 0},
        {1, 0, 1, 0, 0, 0},
        {1, 0, 1, 0, 0, 1},
        {1, 0, 1, 0, 1, 1},
        {1, 0, 0, 0, 1, 0},
        {1, 0, 0, 0, 0, 0},
        {1, 0, 0, 0, 0, 1},
        {1, 0, 0, 0, 1, 1},
        {1, 0, 0, 1, 1, 0},
        {1, 0, 0, 1, 0, 0},
        {1, 0, 0, 1, 0, 1},
        {1, 0, 0, 1, 1, 1},
        {1, 0, 1, 1, 1, 0},
        {1, 0, 1, 1, 0, 0},
        {1, 0, 1, 1, 0, 1},
        {1, 0, 1, 1, 1, 1},
        {0, 0, 1, 0, 1, 0},
        {0, 0, 1, 0, 0, 0},
        {0, 0, 1, 0, 0, 1},
        {0, 0, 1, 0, 1, 1},
        {0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1, 1},
        {0, 0, 0, 1, 1, 0},
        {0, 0, 0, 1, 0, 0},
        {0, 0, 0, 1, 0, 1},
        {0, 0, 0, 1, 1, 1},
        {0, 0, 1, 1, 1, 0},
        {0, 0, 1, 1, 0, 0},
        {0, 0, 1, 1, 0, 1},
        {0, 0, 1, 1, 1, 1},
        {0, 1, 1, 0, 1, 0},
        {0, 1, 1, 0, 0, 0},
        {0, 1, 1, 0, 0, 1},
        {0, 1, 1, 0, 1, 1},
        {0, 1, 0, 0, 1, 0},
        {0, 1, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 1},
        {0, 1, 0, 0, 1, 1},
        {0, 1, 0, 1, 1, 0},
        {0, 1, 0, 1, 0, 0},
        {0, 1, 0, 1, 0, 1},
        {0, 1, 0, 1, 1, 1},
        {0, 1, 1, 1, 1, 0},
        {0, 1, 1, 1, 0, 0},
        {0, 1, 1, 1, 0, 1},
        {0, 1, 1, 1, 1, 1},
        {1, 1, 1, 0, 1, 0},
        {1, 1, 1, 0, 0, 0},
        {1, 1, 1, 0, 0, 1},
        {1, 1, 1, 0, 1, 1},
        {1, 1, 0, 0, 1, 0},
        {1, 1, 0, 0, 0, 0},
        {1, 1, 0, 0, 0, 1},
        {1, 1, 0, 0, 1, 1},
        {1, 1, 0, 1, 1, 0},
        {1, 1, 0, 1, 0, 0},
        {1, 1, 0, 1, 0, 1},
        {1, 1, 0, 1, 1, 1},
        {1, 1, 1, 1, 1, 0},
        {1, 1, 1, 1, 0, 0},
        {1, 1, 1, 1, 0, 1},
        {1, 1, 1, 1, 1, 1}}

    Public ReadOnly m_targetArray3XOR!(,) = {
        {1, 1, 1},
        {1, 1, 0},
        {1, 1, 1},
        {1, 1, 0},
        {1, 0, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 0, 0},
        {1, 1, 1},
        {1, 1, 0},
        {1, 1, 1},
        {1, 1, 0},
        {1, 0, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 0, 0},
        {0, 1, 1},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 0},
        {0, 0, 1},
        {0, 0, 0},
        {0, 0, 1},
        {0, 0, 0},
        {0, 1, 1},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 0},
        {0, 0, 1},
        {0, 0, 0},
        {0, 0, 1},
        {0, 0, 0},
        {1, 1, 1},
        {1, 1, 0},
        {1, 1, 1},
        {1, 1, 0},
        {1, 0, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 0, 0},
        {1, 1, 1},
        {1, 1, 0},
        {1, 1, 1},
        {1, 1, 0},
        {1, 0, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 0, 0},
        {0, 1, 1},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 0},
        {0, 0, 1},
        {0, 0, 0},
        {0, 0, 1},
        {0, 0, 0},
        {0, 1, 1},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 0},
        {0, 0, 1},
        {0, 0, 0},
        {0, 0, 1},
        {0, 0, 0}}

    Private Sub InitXOR(mlp As clsMLPGeneric)
        mlp.Initialize(learningRate:=0.01!)
        mlp.inputArray = m_inputArrayXOR
        mlp.targetArray = m_targetArrayXOR
    End Sub

    Private Sub Init2XOR(mlp As clsMLPGeneric)
        mlp.Initialize(learningRate:=0.01!)
        mlp.inputArray = m_inputArray2XOR
        mlp.targetArray = m_targetArray2XOR
    End Sub

    Private Sub Init3XOR(mlp As clsMLPGeneric)
        mlp.Initialize(learningRate:=0.01!)
        mlp.inputArray = m_inputArray3XOR
        mlp.targetArray = m_targetArray3XOR
    End Sub

#End Region

#End Region

#Region "Standard tests"

    Public Sub TestMLP1XORSemiStochastic(mlp As clsMLPGeneric,
        Optional nbIterations% = 8000,
        Optional expectedLoss# = 0.03#,
        Optional learningRate! = 0.05!,
        Optional weightAdjustment! = 0.1!,
        Optional gain! = 2)

        InitXOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)

        mlp.InitializeWeights(1, {
            {-0.75, 0.64, -0.09},
            {0.12, 0.75, -0.63}})
        mlp.InitializeWeights(2, {
            {-0.79, -0.13, 0.58}})

        mlp.Train(enumLearningMode.SemiStochastic)

        Dim expectedOutput = m_targetArrayXOR
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

        Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")

        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP1XORStochastic(mlp As clsMLPGeneric,
        Optional nbIterations% = 25000,
        Optional expectedLoss# = 0.04#,
        Optional learningRate! = 0.05!,
        Optional weightAdjustment! = 0.1!,
        Optional gain! = 2)

        InitXOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)

        mlp.InitializeWeights(1, {
            {-0.75, 0.64, -0.09},
            {0.12, 0.75, -0.63}})
        mlp.InitializeWeights(2, {
            {-0.79, -0.13, 0.58}})

        mlp.Train(enumLearningMode.Stochastic)

        Dim expectedOutput = m_targetArrayXOR
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

        Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")

        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP1XORWithoutBias(mlp As clsMLPGeneric,
        Optional nbIterations% = 40000,
        Optional expectedLoss# = 0.11#,
        Optional learningRate! = 0.1!,
        Optional weightAdjustment! = 0.02!,
        Optional gain! = 1)

        InitXOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=False)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)

        mlp.InitializeWeights(1, {
            {0.42, 0.79},
            {0.55, 0.02}})
        mlp.InitializeWeights(2, {
            {0.51, 0.31}})

        mlp.Train()

        'Dim expectedOutput = m_targetArrayXOR
        Dim expectedOutput = New Double(,) {
            {0.9},
            {0.1},
            {0.9},
            {0.1}}
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

        Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")

        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP1XORWithoutBias231(mlp As clsMLPGeneric,
        Optional nbIterations% = 60000,
        Optional expectedLoss# = 0.11#,
        Optional learningRate! = 0.08!,
        Optional weightAdjustment! = 0.02!,
        Optional gain! = 1)

        InitXOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=False)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)

        mlp.InitializeWeights(1, {
            {0.73, 0.38},
            {0.07, 0.3},
            {0.99, 0.25}})
        mlp.InitializeWeights(2, {
            {1.0, 0.98, 0.61}})

        mlp.Train()

        'Dim expectedOutput = m_targetArrayXOR
        Dim expectedOutput = New Double(,) {
            {0.9},
            {0.1},
            {0.9},
            {0.1}}

        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

        Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")

        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    'Public Sub TestMLP1XORWithoutBias231b(mlp As clsMLPGeneric,
    '    Optional nbIterations% = 60000,
    '    Optional expectedLoss# = 0.03#,
    '    Optional learningRate! = 0.09!,
    '    Optional weightAdjustment! = 0.05!,
    '    Optional gain! = 1,
    '    Optional center! = 2.2!)

    '    InitXOR(mlp)
    '    mlp.Initialize(learningRate, weightAdjustment)
    '    mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=False)

    '    mlp.nbIterations = nbIterations
    '    mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain, center)

    '    mlp.InitializeWeights(1, {
    '        {0.66, 0.53},
    '        {0.65, 0.69},
    '        {0.82, 0.56}})
    '    mlp.InitializeWeights(2, {
    '        {0.62, 0.54, 0.5}})

    '    mlp.Train()

    '    Dim expectedOutput = m_targetArrayXOR
    '    Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix

    '    Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")

    '    Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
    '    Assert.AreEqual(sExpectedOutput, sOutput)

    '    Dim loss! = mlp.ComputeAverageError()
    '    Dim lossRounded# = Math.Round(loss, 2)
    '    Assert.AreEqual(True, lossRounded <= expectedLoss)

    'End Sub

    Public Sub TestMLP1XORELU(mlp As clsMLPGeneric,
            Optional nbIterations% = 600,
            Optional expectedLoss# = 0.01#,
            Optional learningRate! = 0.05!,
            Optional weightAdjustment! = 0.0!,
            Optional gain! = 0.6!,
            Optional center! = 0.3!,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        InitXOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
        mlp.SetActivationFunction(
            enumActivationFunction.ELU, gain, center)

        mlp.InitializeWeights(1, {
            {0.98, 0.15, 0.5},
            {0.11, 0.11, 0.29}})
        mlp.InitializeWeights(2, {
            {0.83, 0.44, 0.06}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArrayXOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

#End Region

#Region "Vectorized tests"
#End Region

#Region "Universal tests"

    Public Sub TestMLP1XOR4Layers(mlp As clsMLPGeneric,
        Optional nbIterations% = 500,
        Optional expectedLoss# = 0.02,
        Optional learningRate! = 0.1!,
        Optional weightAdjustment! = 0.1,
        Optional gain! = 2,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        InitXOR(mlp)
        mlp.weightAdjustment = weightAdjustment
        mlp.learningRate = learningRate
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCountXOR4Layers, addBiasColumn:=True)
        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain)

        mlp.InitializeWeights(1, {
            {-0.2, 0.97, 0.06},
            {0.73, -0.63, -0.21}})
        mlp.InitializeWeights(2, {
            {0.53, 0.58, -0.6},
            {-0.8, -0.59, 0.05}})
        mlp.InitializeWeights(3, {
            {-0.66, -0.55, 0.49}})

        mlp.Train()

        Dim expectedOutput = m_targetArrayXOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP1XOR5Layers(mlp As clsMLPGeneric,
        Optional nbIterations% = 900,
        Optional expectedLoss# = 0.02,
        Optional learningRate! = 0.1!,
        Optional weightAdjustment! = 0.1,
        Optional gain! = 2,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        InitXOR(mlp)
        mlp.weightAdjustment = weightAdjustment
        mlp.learningRate = learningRate
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCountXOR5Layers, addBiasColumn:=True)
        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain)

        mlp.InitializeWeights(1, {
            {0.86, 0.09, -0.48},
            {0.84, -0.01, 0.52}})
        mlp.InitializeWeights(2, {
            {-0.86, 0.24, 0.42},
            {-0.68, 0.25, -0.68}})
        mlp.InitializeWeights(3, {
            {0.44, -0.42, 0.78},
            {0.38, 0.05, 0.91}})
        mlp.InitializeWeights(4, {
            {0.51, 0.11, 0.84}})

        mlp.Train()

        Dim expectedOutput = m_targetArrayXOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP1XORSigmoid(mlp As clsMLPGeneric,
            Optional nbIterations% = 2000,
            Optional expectedLoss# = 0.04#,
            Optional learningRate! = 0.1!,
            Optional weightAdjustment! = 0.1!,
            Optional gain! = 2.5,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        InitXOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
        mlp.SetActivationFunction(
            enumActivationFunction.Sigmoid, gain)

        mlp.InitializeWeights(1, {
            {0.36, -0.84, 0.37},
            {0.19, -0.8, -0.56}})
        mlp.InitializeWeights(2, {
            {0.14, 0.3, 0.93}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArrayXOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        ' XOR at 90%: works
        mlp.TestAllSamples(m_inputArrayXOR90PC, nbOutputs:=1)
        Dim sOutput90PC$ = mlp.output.ToStringWithFormat(dec:="0.00")

        ' XOR at 80%: works
        mlp.TestAllSamples(m_inputArrayXOR80PC, nbOutputs:=1)
        Dim sOutput80PC$ = mlp.output.ToStringWithFormat(dec:="0.00")

        ' XOR at 70%: does not works anymore
        mlp.TestAllSamples(m_inputArrayXOR70PC, nbOutputs:=1)
        Dim sOutput70PC$ = mlp.output.ToStringWithFormat(dec:="0.00")

    End Sub

    Public Sub TestMLP1XORHTangent(mlp As clsMLPGeneric,
            Optional nbIterations% = 2500,
            Optional expectedLoss# = 0.02#,
            Optional learningRate! = 0.05!,
            Optional weightAdjustment! = 0.1!,
            Optional gain! = 2.4,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        InitXOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations ' HTan: works
        mlp.InitializeStruct(m_neuronCountXOR, addBiasColumn:=True)
        mlp.SetActivationFunction(
            enumActivationFunction.HyperbolicTangent, gain)

        mlp.InitializeWeights(1, {
            {-0.59, 0.78, -0.16},
            {0.25, 0.94, -0.17}})
        mlp.InitializeWeights(2, {
            {-0.45, -0.8, -0.37}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArrayXOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP1XORHTangent261(mlp As clsMLPGeneric,
            Optional nbIterations% = 400,
            Optional expectedLoss# = 0.02#,
            Optional learningRate! = 0.2!,
            Optional weightAdjustment! = 0,
            Optional gain! = 2,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        InitXOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations ' HTan: works
        mlp.InitializeStruct(m_neuronCountXOR261, addBiasColumn:=False)
        mlp.SetActivationFunction(
            enumActivationFunction.HyperbolicTangent, gain)

        mlp.InitializeWeights(1, {
            {0.28, 0.43},
            {-0.47, -0.41},
            {0.02, -0.31},
            {0.06, 0.45},
            {0.22, 0.46},
            {-0.13, 0.08}})
        mlp.InitializeWeights(2, {
            {-0.05, 0.19, 0.34, -0.26, -0.38, -0.07}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArrayXOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP2XORSigmoid(mlp As clsMLPGeneric,
            Optional nbIterations% = 200,
            Optional expectedLoss# = 0.03#,
            Optional learningRate! = 0.09!,
            Optional weightAdjustment! = 0.1!,
            Optional gain! = 2.6!,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Init2XOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)
        mlp.SetActivationFunction(
            enumActivationFunction.HyperbolicTangent, gain)

        mlp.InitializeWeights(1, {
            {0.6, 0.52, 0.12, -0.33, -0.48},
            {-0.5, -0.04, 0.39, 0.4, 0.64},
            {-0.66, 0.23, -0.19, 0.32, 0.6},
            {0.21, 0.6, -0.29, -0.7, 0.01}})
        mlp.InitializeWeights(2, {
            {-0.07, -0.69, -0.08, -0.52, 0.48},
            {0.25, -0.37, -0.34, 0.49, -0.65}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArray2XOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP2XORHTangent(mlp As clsMLPGeneric,
            Optional nbIterations% = 400,
            Optional expectedLoss# = 0.02#,
            Optional learningRate! = 0.1!,
            Optional weightAdjustment! = 0.15!,
            Optional gain! = 2,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Init2XOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations ' HTan: works
        mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)
        mlp.SetActivationFunction(
            enumActivationFunction.HyperbolicTangent, gain)

        mlp.InitializeWeights(1, {
            {0.55, 0.15, 0.71, 0.29, 0.25},
            {0.29, 0.53, -0.1, 0.4, -0.67},
            {-0.54, 0.08, 0.27, 0.72, -0.3},
            {0.48, 0.47, -0.67, 0.17, 0.23}})
        mlp.InitializeWeights(2, {
            {-0.32, -0.57, -0.05, 0.55, -0.5},
            {0.28, -0.76, -0.42, 0.21, 0.31}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArray2XOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP2XORHTangent462(mlp As clsMLPGeneric,
        Optional nbIterations% = 700,
        Optional expectedLoss# = 0.01#,
        Optional learningRate! = 0.1!,
        Optional weightAdjustment! = 0,
        Optional gain! = 2,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Init2XOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations ' HTan: works
        mlp.InitializeStruct(m_neuronCount2XOR462, addBiasColumn:=False)
        mlp.SetActivationFunction(
            enumActivationFunction.HyperbolicTangent, gain)

        mlp.InitializeWeights(1, {
            {0.1, -0.37, -0.04, -0.3},
            {-0.15, -0.02, 0.04, 0.18},
            {0.16, -0.45, 0.13, 0.12},
            {-0.28, 0.08, -0.45, -0.15},
            {0.15, 0.18, 0.48, -0.07},
            {0.2, -0.38, 0.24, -0.45}})
        mlp.InitializeWeights(2, {
            {0.5, 0.47, 0.37, 0.32, -0.3, 0.02},
            {0.34, 0.31, -0.12, -0.33, 0.01, -0.31}})

        'mlp.PrintWeights()
        'mlp.printOutput_ = True

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArray2XOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP3XORSigmoid(mlp As clsMLPGeneric,
            Optional nbIterations% = 600,
            Optional expectedLoss# = 0.02#,
            Optional learningRate! = 0.1!,
            Optional weightAdjustment! = 0.1!,
            Optional gain! = 2,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Init3XOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCount3XOR, addBiasColumn:=True)
        mlp.SetActivationFunction(
            enumActivationFunction.Sigmoid, gain)

        mlp.InitializeWeights(1, {
            {-0.45, -0.45, -0.17, -0.29, -0.03, -0.19, 0.58},
            {0.01, 0.47, -0.55, -0.07, -0.46, -0.33, -0.21},
            {0.18, -0.21, -0.43, 0.49, -0.13, 0.6, -0.09},
            {0.43, 0.39, -0.25, 0.49, -0.07, 0.32, 0.38},
            {0.33, 0.55, -0.19, 0.38, 0.09, -0.19, -0.49},
            {-0.12, 0.58, 0.14, -0.02, 0.13, 0.56, -0.43}})

        mlp.InitializeWeights(2, {
            {-0.14, 0.31, -0.49, -0.46, 0.25, 0.32, -0.4},
            {-0.6, 0.43, -0.06, -0.03, -0.18, 0.5, -0.24},
            {-0.17, -0.53, -0.11, -0.55, 0.13, -0.4, 0.29}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArray3XOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP3XORHTangent(mlp As clsMLPGeneric,
            Optional nbIterations% = 1100,
            Optional expectedLoss# = 0.02#,
            Optional learningRate! = 0.1!,
            Optional weightAdjustment! = 0,
            Optional gain! = 1,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Init3XOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCount3XOR673, addBiasColumn:=True)
        mlp.SetActivationFunction(
            enumActivationFunction.HyperbolicTangent, gain)

        mlp.InitializeWeights(1, {
            {-0.01, -0.12, -0.57, -0.02, -0.9, 0.61, 0.32},
            {0.5, 0.08, 0.86, -0.76, -0.91, -0.23, -0.81},
            {-0.13, 0.14, -0.46, 0.08, -0.25, 0.65, 0.45},
            {-0.48, 0.48, -0.47, 0.49, -0.34, -0.76, 0.71},
            {-0.86, 0.05, -0.18, 0.62, 0.35, 0.42, -0.61},
            {0.02, 0.91, -0.29, 0.37, 0.38, -0.34, 0.84},
            {0.2, 0.66, -0.58, -0.71, -0.51, -0.47, 0.39}})
        mlp.InitializeWeights(2, {
            {0.77, 0.49, -0.99, 0.03, -0.09, -0.61, -0.47, 0.09},
            {-0.57, 0.32, 0.26, -0.47, -0.22, -0.18, 0.17, -0.11},
            {0.68, -0.33, 0.79, 0.65, -0.01, 0.7, -0.18, -0.71}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArray3XOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss! = mlp.ComputeAverageError()
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

#End Region

#Region "Iris standard tests"

    Public Sub InitIrisAnalog(mlp As clsMLPGeneric)
        mlp.inputArray = m_inputArrayIris
        mlp.targetArray = m_targetArrayIrisAnalog
        mlp.InitializeStruct(m_neuronCountIrisAnalog4_20_1, addBiasColumn:=True)
    End Sub

    Public Sub InitIrisLogical(mlp As clsMLPGeneric)
        mlp.inputArray = m_inputArrayIris
        mlp.targetArray = m_targetArrayIrisLogical
        mlp.InitializeStruct(m_neuronCountIrisLogical443, addBiasColumn:=True)
    End Sub

    Public Sub TestMLPIrisAnalog(mlp As clsMLPGeneric,
        Optional nbIterations% = 2000,
        Optional expectedSuccess# = 0.953#,
        Optional expectedLoss# = 0.06#,
        Optional learningRate! = 0.1!,
        Optional weightAdjustment! = 0.1!,
        Optional gain! = 1,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        mlp.Initialize(learningRate, weightAdjustment)
        InitIrisAnalog(mlp)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)

        mlp.InitializeWeights(1, {
            {-0.05, 0.43, -0.26, -0.27, -0.01},
            {-0.5, 0.15, -0.11, 0.4, -0.49},
            {0.15, -0.42, 0.19, -0.1, 0.31},
            {-0.07, 0.05, -0.18, -0.2, -0.03},
            {0.48, 0.45, -0.03, -0.28, 0.34},
            {-0.15, -0.39, -0.34, -0.41, -0.25},
            {0.15, -0.25, 0.46, 0.28, 0.41},
            {0.21, -0.23, 0.03, -0.26, 0.13},
            {-0.24, -0.44, -0.42, 0.24, 0.33},
            {0.45, 0.09, -0.43, -0.16, 0.14},
            {-0.39, 0.22, 0.28, -0.25, 0.25},
            {-0.03, -0.48, -0.11, -0.36, 0.39},
            {-0.2, 0.15, -0.14, -0.09, -0.38},
            {0.41, -0.39, -0.34, -0.47, -0.42},
            {0.18, -0.29, 0.42, 0.37, 0.22},
            {-0.11, 0.39, 0.27, -0.46, 0.31},
            {-0.04, -0.37, 0.43, -0.04, 0.21},
            {-0.47, -0.09, -0.26, -0.25, -0.37},
            {-0.4, 0.24, 0.26, 0.26, -0.07},
            {-0.11, -0.08, 0.33, 0.17, 0.08}})
        mlp.InitializeWeights(2, {
            {-0.26, -0.02, -0.41, 0.15, -0.39, -0.47, 0.33, 0.13, 0.25, 0.13, -0.13, 0.35, 0.18, 0.34, 0.02, -0.07, 0.44, -0.09, 0.39, -0.41, 0.36}})

        mlp.minimalSuccessTreshold = 0.2
        mlp.Train(learningMode)

        Dim success! = mlp.successPC
        Dim successRounded# = Math.Round(success, 3)
        Assert.AreEqual(True, successRounded >= expectedSuccess)

        Dim loss! = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 3)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        If mlp.successPC = 1 Then
            Dim expectedOutput = m_targetArrayIrisAnalog
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)
        End If

    End Sub

    Public Sub TestMLPIrisLogical(mlp As clsMLPGeneric,
        Optional nbIterations% = 300,
        Optional expectedSuccess# = 0.969#,
        Optional expectedLoss# = 0.05#,
        Optional learningRate! = 0.1!,
        Optional weightAdjustment! = 0.1!,
        Optional gain! = 1,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        mlp.Initialize(learningRate, weightAdjustment)
        InitIrisLogical(mlp)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)

        mlp.InitializeWeights(1, {
            {0.5, 0.03, 0.08, 0.35, 0.43},
            {-0.1, 0.09, -0.19, 0.11, 0.46},
            {-0.26, -0.39, -0.49, 0.35, -0.27},
            {-0.4, -0.49, 0.43, -0.38, -0.11}})
        mlp.InitializeWeights(2, {
            {-0.36, 0.37, -0.04, -0.1, -0.23},
            {0.2, -0.42, 0.09, 0.23, -0.38},
            {0.21, -0.35, -0.22, 0.01, -0.07}})

        mlp.minimalSuccessTreshold = 0.3
        mlp.Train(learningMode)

        Dim success! = mlp.successPC
        Dim successRounded# = Math.Round(success, 3)
        Assert.AreEqual(True, successRounded >= expectedSuccess)

        Dim loss! = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 3)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        If mlp.successPC = 1 Then
            Dim expectedOutput = m_targetArrayIrisLogical
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)
        End If

    End Sub

    Public Sub MLPGenericIrisTest(mlp As clsMLPGeneric, testName$, Optional nbIterations% = 2000)

        mlp.ShowMessage(testName)

        mlp.nbIterations = nbIterations

        mlp.Initialize(learningRate:=0.1!, weightAdjustment:=0.1!)

        mlp.printOutput_ = True
        mlp.printOutputMatrix = False

        mlp.inputArray = m_inputArrayIris
        mlp.targetArray = m_targetArrayIrisAnalog
        mlp.InitializeStruct({4, 20, 1}, addBiasColumn:=True)

        mlp.SetActivationFunction(enumActivationFunction.Sigmoid)

        mlp.Randomize()

        mlp.PrintParameters()

        WaitForKeyToStart()

        mlp.minimalSuccessTreshold = 0.2
        mlp.Train()

        mlp.ShowMessage(testName & ": Done.")

    End Sub

#End Region

End Module