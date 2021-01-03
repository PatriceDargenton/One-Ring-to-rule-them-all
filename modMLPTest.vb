
Imports Perceptron.Utility ' Matrix
Imports Perceptron.clsMLPGeneric ' enumLearningMode
Imports Microsoft.VisualStudio.TestTools.UnitTesting ' Assert.AreEqual

Module modMLPTest

#Region "Initialization"

#Region "Iris flower data set"

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

    Public ReadOnly m_neuronCountIrisFlowerAnalog%() = {4, 16, 16, 1}
    Public ReadOnly m_neuronCountIrisFlowerAnalog451%() = {4, 5, 1}
    Public ReadOnly m_neuronCountIrisFlowerAnalog4_20_1%() = {4, 20, 1}
    Public ReadOnly m_neuronCountIrisFlowerAnalog4991%() = {4, 9, 9, 1}
    Public ReadOnly m_neuronCountIrisFlowerLogical%() = {4, 16, 16, 3}
    Public ReadOnly m_neuronCountIrisFlowerLogical443%() = {4, 4, 3}
    Public ReadOnly m_neuronCountIrisFlowerLogical453%() = {4, 5, 3}
    Public ReadOnly m_neuronCountIrisFlowerLogical4663%() = {4, 6, 6, 3}
    Public ReadOnly m_neuronCountIrisFlowerLogical4773%() = {4, 7, 7, 3}
    Public ReadOnly m_neuronCountIrisFlowerLogical4883%() = {4, 8, 8, 3}
    Public ReadOnly m_neuronCountIrisFlowerLogical4_16_83%() = {4, 16, 8, 3}
    Public ReadOnly m_neuronCountIrisFlowerLogical4_20_3%() = {4, 20, 3}

    ' https://en.wikipedia.org/wiki/Iris_flower_data_set#External_links
    ' Corrected: Contains two errors which are documented
    ' 4.9, 3.1, 1.5, 0.1 -> 4.9, 3.1, 1.5, 0.2
    ' 4.9, 3.1, 1.5, 0.1 -> 4.9, 3.6, 1.4, 0.1
    Public ReadOnly m_inputArrayIrisFlower!(,) = {
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

    Public ReadOnly m_targetArrayIrisFlowerAnalogUnnormalized!(,) = {
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

    Public ReadOnly m_targetArrayIrisFlowerAnalog!(,) = {
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

    Public ReadOnly m_targetArrayIrisFlowerLogical!(,) = {
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

    ' Dataset split in train set and test set:
    ' https://github.com/vmitt/iris_dataset_prediction_using_tensorflow

    ' https://en.wikipedia.org/wiki/Iris_flower_data_set#External_links
    ' Corrected: Contains two errors which are documented
    Public ReadOnly m_inputArrayIrisFlowerTrain!(,) = {
        {6.4, 2.8, 5.6, 2.2},
        {5.0, 2.3, 3.3, 1.0},
        {4.9, 2.5, 4.5, 1.7},
        {4.9, 3.1, 1.5, 0.1},
        {5.7, 3.8, 1.7, 0.3},
        {4.4, 3.2, 1.3, 0.2},
        {5.4, 3.4, 1.5, 0.4},
        {6.9, 3.1, 5.1, 2.3},
        {6.7, 3.1, 4.4, 1.4},
        {5.1, 3.7, 1.5, 0.4},
        {5.2, 2.7, 3.9, 1.4},
        {6.9, 3.1, 4.9, 1.5},
        {5.8, 4.0, 1.2, 0.2},
        {5.4, 3.9, 1.7, 0.4},
        {7.7, 3.8, 6.7, 2.2},
        {6.3, 3.3, 4.7, 1.6},
        {6.8, 3.2, 5.9, 2.3},
        {7.6, 3.0, 6.6, 2.1},
        {6.4, 3.2, 5.3, 2.3},
        {5.7, 4.4, 1.5, 0.4},
        {6.7, 3.3, 5.7, 2.1},
        {6.4, 2.8, 5.6, 2.1},
        {5.4, 3.9, 1.3, 0.4},
        {6.1, 2.6, 5.6, 1.4},
        {7.2, 3.0, 5.8, 1.6},
        {5.2, 3.5, 1.5, 0.2},
        {5.8, 2.6, 4.0, 1.2},
        {5.9, 3.0, 5.1, 1.8},
        {5.4, 3.0, 4.5, 1.5},
        {6.7, 3.0, 5.0, 1.7},
        {6.3, 2.3, 4.4, 1.3},
        {5.1, 2.5, 3.0, 1.1},
        {6.4, 3.2, 4.5, 1.5},
        {6.8, 3.0, 5.5, 2.1},
        {6.2, 2.8, 4.8, 1.8},
        {6.9, 3.2, 5.7, 2.3},
        {6.5, 3.2, 5.1, 2.0},
        {5.8, 2.8, 5.1, 2.4},
        {5.1, 3.8, 1.5, 0.3},
        {4.8, 3.0, 1.4, 0.3},
        {7.9, 3.8, 6.4, 2.0},
        {5.8, 2.7, 5.1, 1.9},
        {6.7, 3.0, 5.2, 2.3},
        {5.1, 3.8, 1.9, 0.4},
        {4.7, 3.2, 1.6, 0.2},
        {6.0, 2.2, 5.0, 1.5},
        {4.8, 3.4, 1.6, 0.2},
        {7.7, 2.6, 6.9, 2.3},
        {4.6, 3.6, 1.0, 0.2},
        {7.2, 3.2, 6.0, 1.8},
        {5.0, 3.3, 1.4, 0.2},
        {6.6, 3.0, 4.4, 1.4},
        {6.1, 2.8, 4.0, 1.3},
        {5.0, 3.2, 1.2, 0.2},
        {7.0, 3.2, 4.7, 1.4},
        {6.0, 3.0, 4.8, 1.8},
        {7.4, 2.8, 6.1, 1.9},
        {5.8, 2.7, 5.1, 1.9},
        {6.2, 3.4, 5.4, 2.3},
        {5.0, 2.0, 3.5, 1.0},
        {5.6, 2.5, 3.9, 1.1},
        {6.7, 3.1, 5.6, 2.4},
        {6.3, 2.5, 5.0, 1.9},
        {6.4, 3.1, 5.5, 1.8},
        {6.2, 2.2, 4.5, 1.5},
        {7.3, 2.9, 6.3, 1.8},
        {4.4, 3.0, 1.3, 0.2},
        {7.2, 3.6, 6.1, 2.5},
        {6.5, 3.0, 5.5, 1.8},
        {5.0, 3.4, 1.5, 0.2},
        {4.7, 3.2, 1.3, 0.2},
        {6.6, 2.9, 4.6, 1.3},
        {5.5, 3.5, 1.3, 0.2},
        {7.7, 3.0, 6.1, 2.3},
        {6.1, 3.0, 4.9, 1.8},
        {4.9, 3.1, 1.5, 0.2},
        {5.5, 2.4, 3.8, 1.1},
        {5.7, 2.9, 4.2, 1.3},
        {6.0, 2.9, 4.5, 1.5},
        {6.4, 2.7, 5.3, 1.9},
        {5.4, 3.7, 1.5, 0.2},
        {6.1, 2.9, 4.7, 1.4},
        {6.5, 2.8, 4.6, 1.5},
        {5.6, 2.7, 4.2, 1.3},
        {6.3, 3.4, 5.6, 2.4},
        {4.9, 3.6, 1.4, 0.1},
        {6.8, 2.8, 4.8, 1.4},
        {5.7, 2.8, 4.5, 1.3},
        {6.0, 2.7, 5.1, 1.6},
        {5.0, 3.5, 1.3, 0.3},
        {6.5, 3.0, 5.2, 2.0},
        {6.1, 2.8, 4.7, 1.2},
        {5.1, 3.5, 1.4, 0.3},
        {4.6, 3.1, 1.5, 0.2},
        {6.5, 3.0, 5.8, 2.2},
        {4.6, 3.4, 1.4, 0.3},
        {4.6, 3.2, 1.4, 0.2},
        {7.7, 2.8, 6.7, 2.0},
        {5.9, 3.2, 4.8, 1.8},
        {5.1, 3.8, 1.6, 0.2},
        {4.9, 3.0, 1.4, 0.2},
        {4.9, 2.4, 3.3, 1.0},
        {4.5, 2.3, 1.3, 0.3},
        {5.8, 2.7, 4.1, 1.0},
        {5.0, 3.4, 1.6, 0.4},
        {5.2, 3.4, 1.4, 0.2},
        {5.3, 3.7, 1.5, 0.2},
        {5.0, 3.6, 1.4, 0.2},
        {5.6, 2.9, 3.6, 1.3},
        {4.8, 3.1, 1.6, 0.2},
        {6.3, 2.7, 4.9, 1.8},
        {5.7, 2.8, 4.1, 1.3},
        {5.0, 3.0, 1.6, 0.2},
        {6.3, 3.3, 6.0, 2.5},
        {5.0, 3.5, 1.6, 0.6},
        {5.5, 2.6, 4.4, 1.2},
        {5.7, 3.0, 4.2, 1.2},
        {4.4, 2.9, 1.4, 0.2},
        {4.8, 3.0, 1.4, 0.1},
        {5.5, 2.4, 3.7, 1.0}}

    ' Not corrected
    Public ReadOnly m_inputArrayIrisFlowerTrainOriginal!(,) = {
        {6.4, 2.8, 5.6, 2.2},
        {5.0, 2.3, 3.3, 1.0},
        {4.9, 2.5, 4.5, 1.7},
        {4.9, 3.1, 1.5, 0.1},
        {5.7, 3.8, 1.7, 0.3},
        {4.4, 3.2, 1.3, 0.2},
        {5.4, 3.4, 1.5, 0.4},
        {6.9, 3.1, 5.1, 2.3},
        {6.7, 3.1, 4.4, 1.4},
        {5.1, 3.7, 1.5, 0.4},
        {5.2, 2.7, 3.9, 1.4},
        {6.9, 3.1, 4.9, 1.5},
        {5.8, 4.0, 1.2, 0.2},
        {5.4, 3.9, 1.7, 0.4},
        {7.7, 3.8, 6.7, 2.2},
        {6.3, 3.3, 4.7, 1.6},
        {6.8, 3.2, 5.9, 2.3},
        {7.6, 3.0, 6.6, 2.1},
        {6.4, 3.2, 5.3, 2.3},
        {5.7, 4.4, 1.5, 0.4},
        {6.7, 3.3, 5.7, 2.1},
        {6.4, 2.8, 5.6, 2.1},
        {5.4, 3.9, 1.3, 0.4},
        {6.1, 2.6, 5.6, 1.4},
        {7.2, 3.0, 5.8, 1.6},
        {5.2, 3.5, 1.5, 0.2},
        {5.8, 2.6, 4.0, 1.2},
        {5.9, 3.0, 5.1, 1.8},
        {5.4, 3.0, 4.5, 1.5},
        {6.7, 3.0, 5.0, 1.7},
        {6.3, 2.3, 4.4, 1.3},
        {5.1, 2.5, 3.0, 1.1},
        {6.4, 3.2, 4.5, 1.5},
        {6.8, 3.0, 5.5, 2.1},
        {6.2, 2.8, 4.8, 1.8},
        {6.9, 3.2, 5.7, 2.3},
        {6.5, 3.2, 5.1, 2.0},
        {5.8, 2.8, 5.1, 2.4},
        {5.1, 3.8, 1.5, 0.3},
        {4.8, 3.0, 1.4, 0.3},
        {7.9, 3.8, 6.4, 2.0},
        {5.8, 2.7, 5.1, 1.9},
        {6.7, 3.0, 5.2, 2.3},
        {5.1, 3.8, 1.9, 0.4},
        {4.7, 3.2, 1.6, 0.2},
        {6.0, 2.2, 5.0, 1.5},
        {4.8, 3.4, 1.6, 0.2},
        {7.7, 2.6, 6.9, 2.3},
        {4.6, 3.6, 1.0, 0.2},
        {7.2, 3.2, 6.0, 1.8},
        {5.0, 3.3, 1.4, 0.2},
        {6.6, 3.0, 4.4, 1.4},
        {6.1, 2.8, 4.0, 1.3},
        {5.0, 3.2, 1.2, 0.2},
        {7.0, 3.2, 4.7, 1.4},
        {6.0, 3.0, 4.8, 1.8},
        {7.4, 2.8, 6.1, 1.9},
        {5.8, 2.7, 5.1, 1.9},
        {6.2, 3.4, 5.4, 2.3},
        {5.0, 2.0, 3.5, 1.0},
        {5.6, 2.5, 3.9, 1.1},
        {6.7, 3.1, 5.6, 2.4},
        {6.3, 2.5, 5.0, 1.9},
        {6.4, 3.1, 5.5, 1.8},
        {6.2, 2.2, 4.5, 1.5},
        {7.3, 2.9, 6.3, 1.8},
        {4.4, 3.0, 1.3, 0.2},
        {7.2, 3.6, 6.1, 2.5},
        {6.5, 3.0, 5.5, 1.8},
        {5.0, 3.4, 1.5, 0.2},
        {4.7, 3.2, 1.3, 0.2},
        {6.6, 2.9, 4.6, 1.3},
        {5.5, 3.5, 1.3, 0.2},
        {7.7, 3.0, 6.1, 2.3},
        {6.1, 3.0, 4.9, 1.8},
        {4.9, 3.1, 1.5, 0.1},
        {5.5, 2.4, 3.8, 1.1},
        {5.7, 2.9, 4.2, 1.3},
        {6.0, 2.9, 4.5, 1.5},
        {6.4, 2.7, 5.3, 1.9},
        {5.4, 3.7, 1.5, 0.2},
        {6.1, 2.9, 4.7, 1.4},
        {6.5, 2.8, 4.6, 1.5},
        {5.6, 2.7, 4.2, 1.3},
        {6.3, 3.4, 5.6, 2.4},
        {4.9, 3.1, 1.5, 0.1},
        {6.8, 2.8, 4.8, 1.4},
        {5.7, 2.8, 4.5, 1.3},
        {6.0, 2.7, 5.1, 1.6},
        {5.0, 3.5, 1.3, 0.3},
        {6.5, 3.0, 5.2, 2.0},
        {6.1, 2.8, 4.7, 1.2},
        {5.1, 3.5, 1.4, 0.3},
        {4.6, 3.1, 1.5, 0.2},
        {6.5, 3.0, 5.8, 2.2},
        {4.6, 3.4, 1.4, 0.3},
        {4.6, 3.2, 1.4, 0.2},
        {7.7, 2.8, 6.7, 2.0},
        {5.9, 3.2, 4.8, 1.8},
        {5.1, 3.8, 1.6, 0.2},
        {4.9, 3.0, 1.4, 0.2},
        {4.9, 2.4, 3.3, 1.0},
        {4.5, 2.3, 1.3, 0.3},
        {5.8, 2.7, 4.1, 1.0},
        {5.0, 3.4, 1.6, 0.4},
        {5.2, 3.4, 1.4, 0.2},
        {5.3, 3.7, 1.5, 0.2},
        {5.0, 3.6, 1.4, 0.2},
        {5.6, 2.9, 3.6, 1.3},
        {4.8, 3.1, 1.6, 0.2},
        {6.3, 2.7, 4.9, 1.8},
        {5.7, 2.8, 4.1, 1.3},
        {5.0, 3.0, 1.6, 0.2},
        {6.3, 3.3, 6.0, 2.5},
        {5.0, 3.5, 1.6, 0.6},
        {5.5, 2.6, 4.4, 1.2},
        {5.7, 3.0, 4.2, 1.2},
        {4.4, 2.9, 1.4, 0.2},
        {4.8, 3.0, 1.4, 0.1},
        {5.5, 2.4, 3.7, 1.0}}

    Public ReadOnly m_targetArrayIrisFlowerLogicalTrain!(,) = {
        {0, 0, 1},
        {0, 1, 0},
        {0, 0, 1},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {0, 0, 1},
        {0, 1, 0},
        {1, 0, 0},
        {0, 1, 0},
        {0, 1, 0},
        {1, 0, 0},
        {1, 0, 0},
        {0, 0, 1},
        {0, 1, 0},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {1, 0, 0},
        {0, 0, 1},
        {0, 0, 1},
        {1, 0, 0},
        {0, 0, 1},
        {0, 0, 1},
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
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
        {1, 0, 0},
        {1, 0, 0},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {1, 0, 0},
        {1, 0, 0},
        {0, 0, 1},
        {1, 0, 0},
        {0, 0, 1},
        {1, 0, 0},
        {0, 0, 1},
        {1, 0, 0},
        {0, 1, 0},
        {0, 1, 0},
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 0},
        {0, 0, 1},
        {0, 0, 1},
        {0, 0, 1},
        {0, 1, 0},
        {0, 0, 1},
        {1, 0, 0},
        {0, 0, 1},
        {0, 0, 1},
        {1, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {1, 0, 0},
        {0, 0, 1},
        {0, 0, 1},
        {1, 0, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 0, 1},
        {1, 0, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 0, 1},
        {1, 0, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {1, 0, 0},
        {0, 0, 1},
        {0, 1, 0},
        {1, 0, 0},
        {1, 0, 0},
        {0, 0, 1},
        {1, 0, 0},
        {1, 0, 0},
        {0, 0, 1},
        {0, 1, 0},
        {1, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {1, 0, 0},
        {0, 1, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {1, 0, 0},
        {0, 0, 1},
        {0, 1, 0},
        {1, 0, 0},
        {0, 0, 1},
        {1, 0, 0},
        {0, 1, 0},
        {0, 1, 0},
        {1, 0, 0},
        {1, 0, 0},
        {0, 1, 0}}

    Public ReadOnly m_inputArrayIrisFlowerTest!(,) = {
        {5.9, 3.0, 4.2, 1.5},
        {6.9, 3.1, 5.4, 2.1},
        {5.1, 3.3, 1.7, 0.5},
        {6.0, 3.4, 4.5, 1.6},
        {5.5, 2.5, 4.0, 1.3},
        {6.2, 2.9, 4.3, 1.3},
        {5.5, 4.2, 1.4, 0.2},
        {6.3, 2.8, 5.1, 1.5},
        {5.6, 3.0, 4.1, 1.3},
        {6.7, 2.5, 5.8, 1.8},
        {7.1, 3.0, 5.9, 2.1},
        {4.3, 3.0, 1.1, 0.1},
        {5.6, 2.8, 4.9, 2.0},
        {5.5, 2.3, 4.0, 1.3},
        {6.0, 2.2, 4.0, 1.0},
        {5.1, 3.5, 1.4, 0.2},
        {5.7, 2.6, 3.5, 1.0},
        {4.8, 3.4, 1.9, 0.2},
        {5.1, 3.4, 1.5, 0.2},
        {5.7, 2.5, 5.0, 2.0},
        {5.4, 3.4, 1.7, 0.2},
        {5.6, 3.0, 4.5, 1.5},
        {6.3, 2.9, 5.6, 1.8},
        {6.3, 2.5, 4.9, 1.5},
        {5.8, 2.7, 3.9, 1.2},
        {6.1, 3.0, 4.6, 1.4},
        {5.2, 4.1, 1.5, 0.1},
        {6.7, 3.1, 4.7, 1.5},
        {6.7, 3.3, 5.7, 2.5},
        {6.4, 2.9, 4.3, 1.3}}

    Public ReadOnly m_targetArrayIrisFlowerLogicalTest!(,) = {
        {0, 1, 0},
        {0, 0, 1},
        {1, 0, 0},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {1, 0, 0},
        {0, 0, 1},
        {0, 1, 0},
        {0, 0, 1},
        {0, 0, 1},
        {1, 0, 0},
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 0},
        {1, 0, 0},
        {0, 1, 0},
        {1, 0, 0},
        {1, 0, 0},
        {0, 0, 1},
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0},
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
        {0, 1, 0}}

    Public ReadOnly m_targetArrayIrisFlowerAnalogTrain!(,) = {
        {1},
        {0.5},
        {1},
        {0},
        {0},
        {0},
        {0},
        {1},
        {0.5},
        {0},
        {0.5},
        {0.5},
        {0},
        {0},
        {1},
        {0.5},
        {1},
        {1},
        {1},
        {0},
        {1},
        {1},
        {0},
        {1},
        {1},
        {0},
        {0.5},
        {1},
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
        {0},
        {0},
        {1},
        {1},
        {1},
        {0},
        {0},
        {1},
        {0},
        {1},
        {0},
        {1},
        {0},
        {0.5},
        {0.5},
        {0},
        {0.5},
        {1},
        {1},
        {1},
        {1},
        {0.5},
        {0.5},
        {1},
        {1},
        {1},
        {0.5},
        {1},
        {0},
        {1},
        {1},
        {0},
        {0},
        {0.5},
        {0},
        {1},
        {1},
        {0},
        {0.5},
        {0.5},
        {0.5},
        {1},
        {0},
        {0.5},
        {0.5},
        {0.5},
        {1},
        {0},
        {0.5},
        {0.5},
        {0.5},
        {0},
        {1},
        {0.5},
        {0},
        {0},
        {1},
        {0},
        {0},
        {1},
        {0.5},
        {0},
        {0},
        {0.5},
        {0},
        {0.5},
        {0},
        {0},
        {0},
        {0},
        {0.5},
        {0},
        {1},
        {0.5},
        {0},
        {1},
        {0},
        {0.5},
        {0.5},
        {0},
        {0},
        {0.5}}

    Public ReadOnly m_targetArrayIrisFlowerAnalogTest!(,) = {
        {0.5},
        {1},
        {0},
        {0.5},
        {0.5},
        {0.5},
        {0},
        {1},
        {0.5},
        {1},
        {1},
        {0},
        {1},
        {0.5},
        {0.5},
        {0},
        {0.5},
        {0},
        {0},
        {1},
        {0},
        {0.5},
        {1},
        {0.5},
        {0.5},
        {0.5},
        {0},
        {0.5},
        {1},
        {0.5}}

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

        Dim loss# = mlp.averageError
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

        Dim loss# = mlp.averageError
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

        Dim loss# = mlp.averageError
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

        Dim loss# = mlp.averageError
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

    '    Dim loss# = mlp.averageError
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

        Dim loss# = mlp.averageError
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

        Dim loss# = mlp.averageError
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

        Dim loss# = mlp.averageError
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

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        ' XOR at 90%: works
        mlp.TestAllSamples(m_inputArrayXOR90PC)
        Dim sOutput90PC$ = mlp.output.ToStringWithFormat(dec:="0.00")

        ' XOR at 80%: works
        mlp.TestAllSamples(m_inputArrayXOR80PC)
        Dim sOutput80PC$ = mlp.output.ToStringWithFormat(dec:="0.00")

        ' XOR at 70%: does not works anymore
        mlp.TestAllSamples(m_inputArrayXOR70PC)
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

        Dim loss# = mlp.averageError
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

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP2XORSigmoid(mlp As clsMLPGeneric,
            Optional nbIterations% = 5000,
            Optional expectedLoss# = 0.03#,
            Optional learningRate! = 0.1!,
            Optional weightAdjustment! = 0.1!,
            Optional gain! = 1.0!,
            Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        Init2XOR(mlp)
        mlp.Initialize(learningRate, weightAdjustment)
        mlp.nbIterations = nbIterations
        mlp.InitializeStruct(m_neuronCount2XOR, addBiasColumn:=True)
        mlp.SetActivationFunction(
            enumActivationFunction.Sigmoid, gain)

        mlp.InitializeWeights(1, {
            {-0.05, -0.28, -0.14, -0.42, -0.47},
            {0.41, 0.15, 0.05, 0.04, -0.46},
            {-0.23, 0.29, -0.19, 0.31, -0.37},
            {-0.4, -0.44, -0.29, -0.03, 0.05}})
        mlp.InitializeWeights(2, {
            {-0.2, 0.17, -0.35, 0.46, 0.26},
            {0.16, 0.35, 0.4, -0.11, -0.42}})

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArray2XOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

    Public Sub TestMLP2XORHTangent2(mlp As clsMLPGeneric,
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

        Dim loss# = mlp.averageError
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

        Dim loss# = mlp.averageError
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

        mlp.Train(learningMode)

        Dim expectedOutput = m_targetArray2XOR

        Dim sOutput$ = mlp.output.ToStringWithFormat(dec:="0.0")
        Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
        Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
        Assert.AreEqual(sExpectedOutput, sOutput)

        Dim loss# = mlp.averageError
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

        Dim loss# = mlp.averageError
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

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 2)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

    End Sub

#End Region

#Region "Iris flower standard tests"

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

    Public Sub TestMLPIrisFlowerAnalogTanh(mlp As clsMLPGeneric,
        Optional nbIterations% = 200,
        Optional expectedSuccess# = 0.967#,
        Optional expectedSuccessPrediction# = 0.967#,
        Optional expectedLoss# = 0.031#,
        Optional learningRate! = 0.1!,
        Optional weightAdjustment! = 0.1!,
        Optional gain! = 2,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        mlp.Initialize(learningRate, weightAdjustment)
        InitIrisFlowerAnalog4Layers(mlp)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain)

        mlp.InitializeWeights(1, {
            {0.1, 0.31, 0.24, -0.45, -0.2},
            {-0.28, -0.38, -0.26, -0.18, 0.32},
            {0.39, -0.05, 0.27, -0.03, -0.09},
            {-0.45, -0.15, 0.46, -0.33, 0.15},
            {0.28, -0.23, -0.5, -0.03, -0.09},
            {-0.03, 0.12, -0.44, 0.09, 0.27},
            {-0.19, 0.21, 0.4, -0.46, 0.41},
            {0.23, -0.3, 0.27, 0.07, -0.2},
            {0.25, 0.26, -0.36, 0.1, 0.32}})
        mlp.InitializeWeights(2, {
            {-0.31, 0.2, -0.1, 0.28, -0.02, -0.11, -0.31, -0.23, 0.32, 0.04},
            {0.24, 0.08, 0.39, 0.25, -0.32, 0.22, -0.45, 0.08, 0.23, 0.37},
            {-0.07, -0.17, -0.34, -0.32, 0.1, -0.34, -0.3, -0.48, 0.01, 0.37},
            {0.37, 0.09, -0.22, 0.41, 0.16, 0.32, -0.21, 0.43, 0.02, 0.14},
            {-0.28, -0.14, 0.49, 0.12, 0.38, 0.33, 0.48, -0.38, -0.47, 0.21},
            {0.06, 0.37, 0.45, 0.26, -0.3, -0.42, 0.12, -0.21, 0.29, -0.47},
            {-0.44, 0.3, -0.47, 0.37, 0.21, -0.2, -0.03, 0.1, 0.02, 0.4},
            {0.25, 0.15, -0.29, -0.26, -0.29, -0.36, -0.17, 0.04, -0.19, 0.27},
            {-0.32, -0.42, 0.49, 0.3, -0.2, 0.5, 0.15, 0.01, 0.21, 0.08}})
        mlp.InitializeWeights(3, {
            {-0.3, -0.5, -0.36, 0.07, -0.08, -0.17, -0.38, 0.4, 0.08, -0.14}})

        mlp.minimalSuccessTreshold = 0.2
        mlp.Train(learningMode)

        Dim success! = mlp.successPC
        Dim successRounded# = Math.Round(success, 3)
        Assert.AreEqual(True, successRounded >= expectedSuccess)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 3)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        If mlp.successPC = 1 Then
            Dim expectedOutput = m_targetArrayIrisFlowerAnalogTest
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)
        End If

        mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
            m_targetArrayIrisFlowerAnalogTest, nbOutputs:=1)
        Dim successPrediction! = mlp.successPC
        Dim successPredictionRounded# = Math.Round(successPrediction, 3)
        Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

    End Sub

    Public Sub TestMLPIrisFlowerAnalogTanh2(mlp As clsMLPGeneric,
        Optional nbIterations% = 50,
        Optional expectedSuccess# = 0.983#,
        Optional expectedSuccessPrediction# = 0.967#,
        Optional expectedLoss# = 0.06#,
        Optional learningRate! = 0.1!,
        Optional weightAdjustment! = 0.1!,
        Optional gain! = 2,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        mlp.Initialize(learningRate, weightAdjustment)
        InitIrisFlowerAnalog4Layers(mlp)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain)

        mlp.InitializeWeights(1, {
            {-0.79, -0.65, 0.35, 0.73, -0.75},
            {-0.6, 0.12, 0.27, -0.99, -0.08},
            {0.39, -0.48, 0.76, -0.42, -0.83},
            {-0.71, -0.64, -0.42, -0.55, 0.49},
            {0.19, 0.56, 0.24, -0.74, 0.93},
            {0.08, 0.56, -0.55, -0.81, 0.63},
            {0.47, 0.35, -0.18, -0.51, -0.3},
            {0.75, -0.97, 0.6, 0.43, 0.54},
            {0.4, 0.3, 0.47, -0.41, 0.75}})
        mlp.InitializeWeights(2, {
            {-0.48, 0.53, -0.18, 0.14, 0.97, 0.43, -0.11, 0.55, -0.3, 0.66},
            {-0.37, 0.63, 0.91, 0.86, 0.66, 0.31, -0.8, 0.41, 0.73, -0.44},
            {0.49, 0.15, -0.14, 0.34, 0.04, 0.45, -0.24, 0.75, 0.55, 0.99},
            {-0.93, -0.06, -0.67, -0.88, 0.09, -0.43, 0.36, 0.45, -0.49, 0.15},
            {-0.03, 0.47, 0.87, 0.88, -0.65, 0.88, 0.09, -0.76, 0.02, -0.78},
            {0.49, 0.59, -0.87, 0.71, -0.65, -0.81, 0.87, 0.18, -0.09, -0.99},
            {-0.91, 0.62, -0.02, -0.3, 0.31, 0.9, -0.62, -0.38, -0.66, -0.92},
            {0.91, 0.74, 0.1, -0.7, -0.04, 0.88, -0.04, 0.64, 0.54, 0.81},
            {-0.11, 0.73, 0.11, -0.03, 0.8, 0.63, -0.95, -0.92, 0.79, -0.9}})
        mlp.InitializeWeights(3, {
            {-0.78, -0.38, -0.43, 0.16, -0.28, 0.4, -0.7, 0.76, 0.56, -0.95}})

        mlp.minimalSuccessTreshold = 0.2
        mlp.Train(learningMode)

        Dim success! = mlp.successPC
        Dim successRounded# = Math.Round(success, 3)
        Assert.AreEqual(True, successRounded >= expectedSuccess)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 3)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        If mlp.successPC = 1 Then
            Dim expectedOutput = m_targetArrayIrisFlowerAnalogTest
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)
        End If

        mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
            m_targetArrayIrisFlowerAnalogTest, nbOutputs:=1)
        Dim successPrediction! = mlp.successPC
        Dim successPredictionRounded# = Math.Round(successPrediction, 3)
        Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

    End Sub

    Public Sub TestMLPIrisFlowerAnalogSigmoid(mlp As clsMLPGeneric,
        Optional nbIterations% = 150,
        Optional expectedSuccess# = 0.958#,
        Optional expectedSuccessPrediction# = 0.967#,
        Optional expectedLoss# = 0.072#,
        Optional learningRate! = 0.2!,
        Optional weightAdjustment! = 0!,
        Optional gain! = 1.1!,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        mlp.Initialize(learningRate, weightAdjustment)
        InitIrisFlowerAnalog4Layers(mlp)
        mlp.InitializeStruct({4, 12, 8, 1}, addBiasColumn:=True)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)

        mlp.InitializeWeights(1, {
            {0.35, -0.16, -0.09, 0.22, -0.47},
            {-0.46, 0.33, 0.21, 0.43, 0.1},
            {0.47, 0.19, -0.21, -0.42, 0.22},
            {0.18, 0.16, 0.47, 0.45, -0.38},
            {0.02, 0.36, 0.31, 0.46, -0.45},
            {-0.23, 0.17, -0.19, -0.49, -0.42},
            {0.36, 0.16, -0.2, 0.09, 0.4},
            {-0.06, 0.46, 0.23, 0.42, 0.45},
            {-0.04, -0.25, -0.24, -0.03, 0.17},
            {0.03, 0.41, -0.16, -0.13, 0.13},
            {0.27, 0.05, -0.09, 0.08, -0.17},
            {-0.02, 0.09, 0.26, 0.48, 0.27}})
        mlp.InitializeWeights(2, {
            {-0.35, -0.1, 0.43, 0.02, -0.39, -0.47, 0.49, 0.32, -0.26, 0.2, -0.27, 0.09, 0.5},
            {0.16, -0.23, 0.1, -0.15, -0.21, 0.02, -0.13, -0.17, 0.44, -0.12, -0.19, -0.19, -0.25},
            {0.22, -0.24, -0.08, 0.35, -0.3, 0.25, -0.35, 0.29, -0.44, -0.18, 0.24, -0.14, 0.15},
            {0.04, -0.09, 0.02, -0.37, 0.43, 0.04, 0.47, -0.09, 0.42, -0.44, 0.38, -0.25, -0.03},
            {-0.04, -0.09, 0.32, -0.04, 0.05, -0.29, 0.3, 0.27, -0.23, -0.26, -0.1, -0.11, -0.02},
            {-0.06, -0.3, 0.1, 0.45, 0.37, 0.49, 0.14, 0.48, 0.45, 0.32, 0.3, -0.06, 0.28},
            {-0.16, 0.34, 0.3, -0.3, 0.05, 0.1, 0.23, -0.21, 0.24, 0.47, 0.09, 0.27, 0.03},
            {0.06, 0.38, -0.23, -0.33, -0.38, 0.24, -0.06, -0.45, 0.03, -0.29, -0.19, -0.39, -0.49}})
        mlp.InitializeWeights(3, {
            {-0.25, 0.33, -0.02, -0.05, -0.49, 0.16, -0.39, -0.07, -0.04}})

        mlp.minimalSuccessTreshold = 0.2

        mlp.Train(learningMode)

        Dim success! = mlp.successPC
        Dim successRounded# = Math.Round(success, 3)
        Assert.AreEqual(True, successRounded >= expectedSuccess)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 3)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        If mlp.successPC = 1 Then
            Dim expectedOutput = m_targetArrayIrisFlowerAnalogTest
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)
        End If

        mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
            m_targetArrayIrisFlowerAnalogTest, nbOutputs:=1)
        Dim successPrediction! = mlp.successPC
        Dim successPredictionRounded# = Math.Round(successPrediction, 3)
        Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

    End Sub

    Public Sub TestMLPIrisFlowerAnalogGaussian(mlp As clsMLPGeneric,
        Optional nbIterations% = 100,
        Optional expectedSuccess# = 0.967#,
        Optional expectedSuccessPrediction# = 0.933#,
        Optional expectedLoss# = 0.07#,
        Optional learningRate! = 0.2!,
        Optional weightAdjustment! = 0.2!,
        Optional gain! = 0.2,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        mlp.Initialize(learningRate, weightAdjustment)
        InitIrisFlowerAnalog4Layers(mlp)

        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Gaussian, gain)

        mlp.InitializeWeights(1, {
            {-0.49, -0.39, 0.24, 0.25, 0.07},
            {-0.21, -0.42, -0.37, -0.09, 0.2},
            {-0.03, -0.27, 0.26, 0.23, 0.06},
            {0.25, -0.31, -0.05, -0.1, 0.17},
            {-0.19, -0.42, 0.37, 0.18, 0.38},
            {-0.12, -0.22, 0.35, -0.42, 0.06},
            {-0.26, -0.48, 0.41, -0.19, 0.21},
            {0.49, -0.12, -0.24, -0.48, -0.15},
            {0.27, 0.13, 0.2, 0.4, -0.04}})
        mlp.InitializeWeights(2, {
            {-0.48, 0.11, 0.18, -0.08, -0.4, -0.46, -0.45, 0.18, 0.23, 0.19},
            {-0.47, 0.37, -0.43, 0.19, 0.41, 0.3, -0.22, -0.17, -0.35, -0.24},
            {-0.05, -0.17, -0.06, -0.48, 0.07, -0.13, 0.43, -0.08, -0.46, 0.39},
            {0.18, -0.12, 0.47, -0.28, 0.36, 0.26, 0.1, -0.08, 0.48, 0.02},
            {-0.31, -0.15, -0.32, 0.12, 0.18, -0.39, -0.19, 0.06, -0.39, 0.05},
            {-0.2, 0.06, 0.14, -0.48, 0.19, -0.33, 0.17, 0.35, 0.23, -0.4},
            {0.13, 0.13, 0.33, -0.49, 0.15, 0.41, 0.35, 0.33, -0.35, -0.3},
            {0.36, -0.15, 0.12, -0.43, -0.4, -0.35, 0.32, -0.16, -0.04, -0.43},
            {-0.13, -0.19, -0.1, -0.12, 0.32, -0.17, -0.3, -0.34, 0.09, 0.43}})
        mlp.InitializeWeights(3, {
            {0.07, -0.25, -0.08, 0.06, 0.22, -0.16, 0.11, -0.48, -0.29, -0.05}})

        mlp.minimalSuccessTreshold = 0.2

        mlp.Train(learningMode)

        Dim success! = mlp.successPC
        Dim successRounded# = Math.Round(success, 3)
        Assert.AreEqual(True, successRounded >= expectedSuccess)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 3)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        If mlp.successPC = 1 Then
            Dim expectedOutput = m_targetArrayIrisFlowerAnalogTest
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)
        End If

        mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
            m_targetArrayIrisFlowerAnalogTest, nbOutputs:=1)
        Dim successPrediction! = mlp.successPC
        Dim successPredictionRounded# = Math.Round(successPrediction, 3)
        Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

    End Sub

    Public Sub TestMLPIrisFlowerLogicalTanh(mlp As clsMLPGeneric,
        Optional nbIterations% = 1500,
        Optional expectedSuccess# = 0.994#,
        Optional expectedSuccessPrediction# = 0.978#,
        Optional expectedLoss# = 0.025#,
        Optional learningRate! = 0.1!,
        Optional weightAdjustment! = 0.1!,
        Optional gain! = 2,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        ' 97.8% prediction, 99.4% learning with 1500 iterations

        mlp.Initialize(learningRate, weightAdjustment)
        InitIrisFlowerLogical4Layers(mlp)

        mlp.minimalSuccessTreshold = 0.3
        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.HyperbolicTangent, gain)

        mlp.InitializeWeights(1, {
            {0.74, -0.46, -0.07, -0.78, -0.76},
            {-0.51, 0.88, 0.34, 0.85, -0.28},
            {0.17, 1.04, -0.68, -0.62, 0.07},
            {0.69, -0.86, -0.69, 0.05, 0.52},
            {-0.63, 0.84, 0.63, 0.05, -0.69},
            {-0.65, -0.58, -0.78, -0.27, 0.72},
            {-0.3, -0.15, 0.2, -1.04, 0.85},
            {-0.11, 0.68, 0.54, -0.51, -0.97},
            {-0.04, 0.58, -0.79, 0.58, 0.82},
            {1.12, 0.19, 0.17, -0.32, -0.73},
            {0.53, 0.42, -0.68, -0.65, 0.79},
            {0.6, 0.06, 0.71, -0.73, -0.75},
            {-0.76, 0.49, -0.32, -0.66, 0.78},
            {-0.75, -0.35, -0.65, -0.26, -0.89},
            {-0.92, -0.47, 0.76, -0.31, -0.47},
            {-0.28, 0.78, 1.11, 0.16, -0.08}})
        mlp.InitializeWeights(2, {
            {0.18, 0.22, 0.22, -0.56, -0.52, 0.15, -0.22, -0.05, -0.45, 0.57, 0.48, 0.34, 0.18, -0.24, -0.42, 0.19, 0.03},
            {0.59, 0.41, 0.4, -0.14, -0.18, -0.42, -0.16, -0.35, -0.03, 0.55, 0.3, 0.4, 0.07, -0.56, 0.09, -0.15, 0.09},
            {-0.16, -0.21, -0.47, -0.26, -0.56, -0.17, 0.4, 0.47, -0.51, -0.15, -0.43, -0.19, 0.2, 0.01, -0.31, 0.19, 0.44},
            {0.47, -0.48, 0.32, -0.55, -0.26, 0.04, -0.24, 0.39, 0.09, 0.25, 0.09, 0.18, 0.06, 0.59, -0.2, -0.11, -0.55},
            {0.5, 0.37, 0.06, -0.31, 0.52, -0.03, 0.59, 0.01, -0.27, 0.34, -0.33, 0.44, 0.38, -0.07, 0.0, -0.4, 0.24},
            {-0.45, -0.1, 0.2, -0.02, -0.64, 0.43, 0.43, -0.01, -0.04, 0.24, -0.35, -0.37, 0.05, -0.1, 0.44, 0.63, 0.1},
            {0.1, -0.07, -0.45, 0.13, 0.48, 0.19, 0.28, 0.09, -0.53, 0.28, -0.45, 0.49, 0.29, -0.28, -0.5, -0.04, 0.39},
            {0.3, 0.5, 0.33, 0.12, 0.43, -0.58, 0.3, 0.3, -0.05, 0.36, 0.28, -0.28, 0.28, -0.18, -0.09, 0.52, -0.31}})
        mlp.InitializeWeights(3, {
            {0.43, -0.53, -0.53, -0.5, -0.28, 0.32, 0.47, 0.5, -0.57},
            {-0.86, 0.38, 0.12, 0.42, -0.16, -0.58, 0.53, 0.36, 0.34},
            {0.59, -0.53, 0.72, 0.35, -0.11, -0.22, 0.55, -0.05, 0.56}})

        mlp.Train()

        Dim success! = mlp.successPC
        Dim successRounded# = Math.Round(success, 3)
        Assert.AreEqual(True, successRounded >= expectedSuccess)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 3)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        If mlp.successPC = 1 AndAlso mlp.minimalSuccessTreshold <= 0.05! Then
            Dim expectedOutput = m_targetArrayIrisFlowerLogicalTrain
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)
        End If

        mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
            m_targetArrayIrisFlowerLogicalTest, nbOutputs:=3)
        Dim successPrediction! = mlp.successPC
        Dim successPredictionRounded# = Math.Round(successPrediction, 3)
        Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

    End Sub

    Public Sub TestMLPIrisFlowerLogicalSigmoid(mlp As clsMLPGeneric,
        Optional nbIterations% = 1000,
        Optional expectedSuccess# = 0.986#,
        Optional expectedSuccessPrediction# = 0.978#,
        Optional expectedLoss# = 0.058#,
        Optional learningRate! = 0.1!,
        Optional weightAdjustment! = 0.1!,
        Optional gain! = 1,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        ' 97.8% prediction, 98.6% learning with 200 iterations

        mlp.Initialize(learningRate, weightAdjustment)
        InitIrisFlowerLogical4Layers(mlp)

        mlp.minimalSuccessTreshold = 0.3
        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Sigmoid, gain)

        mlp.InitializeWeights(1, {
            {0.86, -0.47, 0.71, -0.61, 0.37},
            {-0.88, -0.19, -0.68, 0.74, 0.36},
            {-0.03, -0.78, -0.69, 0.32, -0.87},
            {-0.23, -0.1, -0.26, -1.09, 0.8},
            {0.59, 0.82, 0.81, 0.52, 0.14},
            {-0.71, -0.56, 0.99, 0.04, -0.4},
            {-0.57, -0.44, 0.75, -0.82, 0.46},
            {-0.19, -0.41, 0.7, -0.86, 0.73},
            {-0.78, -0.33, -0.52, -0.4, -0.9},
            {-0.41, 0.03, -0.91, 0.41, 0.9},
            {-0.54, -0.49, 0.61, -0.62, 0.82},
            {0.89, -0.29, -0.98, 0.33, 0.04},
            {1.22, -0.36, 0.49, 0.3, 0.09},
            {-0.5, -0.69, -0.74, 0.77, 0.29},
            {-1.0, 0.2, -0.34, -0.85, 0.3},
            {-0.24, 0.66, 0.42, 0.61, -0.96}})
        mlp.InitializeWeights(2, {
            {0.15, -0.38, -0.24, 0.08, -0.57, 0.41, -0.01, 0.22, -0.36, 0.4, -0.54, -0.29, -0.31, 0.01, -0.49, 0.33, 0.27},
            {0.26, 0.39, 0.49, -0.44, -0.13, -0.44, -0.55, 0.39, 0.27, 0.53, 0.06, 0.19, -0.15, -0.28, 0.33, -0.14, -0.15},
            {0.43, 0.17, -0.43, -0.48, 0.08, 0.19, 0.28, -0.05, 0.47, -0.49, 0.13, 0.43, -0.17, 0.4, -0.42, 0.35, 0.23},
            {0.44, 0.28, 0.24, -0.37, -0.27, -0.48, -0.27, 0.25, -0.21, 0.43, 0.45, -0.46, 0.32, 0.07, -0.23, 0.36, 0.35},
            {0.15, -0.11, 0.23, 0.22, -0.46, -0.46, -0.45, 0.24, -0.05, -0.52, -0.08, 0.54, -0.16, -0.56, -0.09, 0.42, 0.21},
            {-0.21, 0.63, 0.19, -0.33, 0.2, 0.17, 0.49, -0.24, -0.29, 0.32, 0.4, -0.11, -0.5, -0.02, -0.2, -0.61, -0.01},
            {-0.28, 0.23, -0.4, -0.52, 0.34, 0.16, 0.11, -0.06, 0.31, 0.48, 0.33, 0.37, 0.48, 0.41, 0.02, -0.33, -0.4},
            {-0.33, -0.26, -0.35, 0.13, 0.56, 0.34, 0.36, -0.18, 0.44, -0.12, -0.19, -0.27, -0.19, 0.14, -0.5, 0.47, 0.46}})
        mlp.InitializeWeights(3, {
            {0.54, 0.39, 0.32, 0.22, 0.06, -0.27, -0.77, 0.52, 0.66},
            {0.32, -0.66, -0.02, -0.23, 0.6, -0.33, -0.46, 0.74, -0.39},
            {-0.62, 0.45, 0.54, -0.3, 0.6, -0.22, 0.26, 0.62, -0.37}})

        mlp.Train(learningMode)

        Dim success! = mlp.successPC
        Dim successRounded# = Math.Round(success, 3)
        Assert.AreEqual(True, successRounded >= expectedSuccess)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 3)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        If mlp.successPC = 1 AndAlso mlp.minimalSuccessTreshold <= 0.05! Then
            Dim expectedOutput = m_targetArrayIrisFlowerLogicalTrain
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)
        End If

        mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
            m_targetArrayIrisFlowerLogicalTest, nbOutputs:=3)
        Dim successPrediction! = mlp.successPC
        Dim successPredictionRounded# = Math.Round(successPrediction, 3)
        Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

    End Sub

    Public Sub TestMLPIrisFlowerLogicalGaussian(mlp As clsMLPGeneric,
        Optional nbIterations% = 100,
        Optional expectedSuccess# = 0.953#,
        Optional expectedSuccessPrediction# = 0.967#,
        Optional expectedLoss# = 0.06#,
        Optional learningRate! = 0.2!,
        Optional weightAdjustment! = 0.2!,
        Optional gain! = 0.1,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        ' 96.7% prediction, 95.3% learning with 100 iterations

        mlp.Initialize(learningRate, weightAdjustment)
        InitIrisFlowerLogical4Layers(mlp)
        mlp.InitializeStruct({4, 9, 9, 3}, addBiasColumn:=True)

        mlp.minimalSuccessTreshold = 0.3
        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Gaussian, gain)

        mlp.InitializeWeights(1, {
            {-0.15, -0.26, 0.3, -0.18, -0.5},
            {0.1, 0.29, -0.41, -0.16, -0.39},
            {0.17, -0.38, 0.1, -0.2, -0.13},
            {0.4, 0.36, 0.35, 0.22, 0.38},
            {-0.16, 0.23, 0.2, -0.11, -0.09},
            {-0.32, 0.21, -0.43, -0.4, 0.47},
            {-0.31, 0.48, 0.16, 0.22, 0.3},
            {0.21, 0.11, -0.15, -0.02, 0.38},
            {0.41, 0.44, 0.24, -0.12, 0.03}})
        mlp.InitializeWeights(2, {
            {0.34, 0.48, -0.17, -0.06, 0.44, 0.11, 0.33, -0.03, 0.41, -0.43},
            {0.12, 0.11, -0.41, -0.07, -0.43, -0.48, -0.27, -0.36, -0.01, -0.36},
            {0.44, 0.08, 0.21, 0.16, -0.21, -0.16, -0.26, 0.5, -0.07, 0.3},
            {0.47, -0.48, 0.38, -0.16, 0.23, 0.24, 0.16, -0.47, 0.08, -0.35},
            {-0.32, 0.31, 0.22, -0.41, -0.5, -0.23, -0.42, -0.09, 0.36, 0.18},
            {0.29, -0.25, -0.26, 0.09, -0.24, -0.23, 0.16, -0.35, -0.07, -0.24},
            {0.09, -0.02, -0.02, -0.23, 0.15, 0.09, -0.29, -0.41, -0.16, -0.22},
            {0.07, -0.39, 0.41, 0.49, 0.26, 0.27, 0.49, 0.07, -0.21, -0.26},
            {-0.23, 0.33, -0.36, 0.25, -0.5, 0.21, 0.28, -0.21, -0.42, -0.04}})
        mlp.InitializeWeights(3, {
            {-0.42, 0.1, -0.18, 0.06, 0.21, -0.16, -0.05, -0.06, -0.09, -0.44},
            {0.49, -0.44, 0.08, -0.39, 0.18, -0.16, -0.26, 0.33, -0.01, -0.21},
            {-0.41, 0.36, -0.31, 0.5, 0.26, -0.16, 0.22, -0.23, 0.44, 0.31}})

        mlp.Train(learningMode)

        Dim success! = mlp.successPC
        Dim successRounded# = Math.Round(success, 3)
        Assert.AreEqual(True, successRounded >= expectedSuccess)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 3)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        If mlp.successPC = 1 AndAlso mlp.minimalSuccessTreshold <= 0.05! Then
            Dim expectedOutput = m_targetArrayIrisFlowerLogicalTrain
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)
        End If

        mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
            m_targetArrayIrisFlowerLogicalTest, nbOutputs:=3)
        Dim successPrediction! = mlp.successPC
        Dim successPredictionRounded# = Math.Round(successPrediction, 3)
        Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

    End Sub

    Public Sub TestMLPIrisFlowerLogicalSinus(mlp As clsMLPGeneric,
        Optional nbIterations% = 200,
        Optional expectedSuccess# = 0.933#,
        Optional expectedSuccessPrediction# = 0.978#,
        Optional expectedLoss# = 0.1#,
        Optional learningRate! = 0.05!,
        Optional weightAdjustment! = 0.05!,
        Optional gain! = 0.9,
        Optional learningMode As enumLearningMode = enumLearningMode.Defaut)

        ' 97.8% prediction, 93.3% learning with 200 iterations

        mlp.Initialize(learningRate, weightAdjustment)
        InitIrisFlowerLogical4Layers(mlp)
        mlp.InitializeStruct({4, 9, 9, 3}, addBiasColumn:=True)

        mlp.minimalSuccessTreshold = 0.3
        mlp.nbIterations = nbIterations
        mlp.SetActivationFunction(enumActivationFunction.Sinus, gain)

        mlp.InitializeWeights(1, {
            {0.07, -0.14, 0.17, -0.35, -0.03},
            {-0.32, -0.28, -0.27, -0.19, -0.29},
            {-0.02, -0.28, -0.45, 0.32, -0.44},
            {0.07, 0.39, -0.36, -0.28, -0.27},
            {-0.21, 0.4, -0.49, -0.17, 0.24},
            {-0.3, -0.16, -0.45, -0.23, -0.12},
            {0.12, -0.12, -0.38, -0.25, 0.42},
            {0.49, -0.5, -0.12, -0.02, -0.11},
            {0.05, -0.24, -0.46, 0.09, -0.47}})
        mlp.InitializeWeights(2, {
            {-0.16, 0.23, -0.33, -0.37, 0.3, -0.26, 0.12, -0.22, -0.05, 0.16},
            {0.32, 0.23, 0.18, -0.48, 0.4, -0.24, 0.5, 0.44, -0.39, 0.1},
            {0.18, 0.19, 0.31, 0.14, 0.39, -0.36, 0.34, 0.26, 0.39, -0.15},
            {0.25, -0.21, -0.35, -0.03, 0.36, 0.1, 0.38, -0.14, -0.04, -0.37},
            {0.5, -0.41, 0.17, 0.09, -0.39, -0.24, -0.18, -0.14, 0.08, -0.36},
            {0.05, -0.18, 0.43, 0.49, -0.15, 0.15, 0.42, 0.03, -0.26, 0.16},
            {-0.09, 0.36, -0.11, -0.4, 0.41, 0.03, 0.08, -0.29, -0.34, -0.21},
            {-0.13, 0.13, -0.02, 0.48, 0.1, 0.08, -0.48, -0.28, 0.04, 0.13},
            {0.32, -0.02, -0.32, 0.26, 0.31, -0.08, -0.14, -0.35, -0.38, -0.29}})
        mlp.InitializeWeights(3, {
            {0.18, -0.14, -0.37, 0.3, 0.22, -0.36, 0.2, 0.07, 0.18, 0.08},
            {0.18, -0.38, -0.3, -0.22, 0.27, 0.42, 0.35, 0.45, -0.11, 0.27},
            {0.12, 0.2, 0.49, 0.1, 0.35, 0.43, 0.18, 0.13, -0.21, -0.01}})

        mlp.Train(learningMode)

        Dim success! = mlp.successPC
        Dim successRounded# = Math.Round(success, 3)
        Assert.AreEqual(True, successRounded >= expectedSuccess)

        Dim loss# = mlp.averageError
        Dim lossRounded# = Math.Round(loss, 3)
        Assert.AreEqual(True, lossRounded <= expectedLoss)

        If mlp.successPC = 1 AndAlso mlp.minimalSuccessTreshold <= 0.05! Then
            Dim expectedOutput = m_targetArrayIrisFlowerLogicalTrain
            Dim expectedMatrix As Matrix = expectedOutput ' Single(,) -> Matrix
            Dim sOutput = mlp.output.ToStringWithFormat(dec:="0.0")
            Dim sExpectedOutput = expectedMatrix.ToStringWithFormat(dec:="0.0")
            Assert.AreEqual(sExpectedOutput, sOutput)
        End If

        mlp.TestAllSamples(m_inputArrayIrisFlowerTest,
            m_targetArrayIrisFlowerLogicalTest, nbOutputs:=3)
        Dim successPrediction! = mlp.successPC
        Dim successPredictionRounded# = Math.Round(successPrediction, 3)
        Assert.AreEqual(True, successPredictionRounded >= expectedSuccessPrediction)

    End Sub

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

End Module