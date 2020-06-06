
Imports Perceptron.Util ' Matrix

Namespace DLFramework

    Public Enum TensorOperations
        Addition
        Negation
        Substraction
        Multiplication
        Sumatory
        Transpose
        MatrixMultiplication
        Expand
        Other
        None
    End Enum

    Public Class Tensor

        Private Shared idCount% = 0
        Private m_data As Matrix
        Private m_creators As List(Of Tensor)
        Private m_childrens As Dictionary(Of Integer, Integer)
        Private m_creationOperation As TensorOperations
        Private m_gradient As Tensor
        Private m_autoGrad As Boolean
        Private m_id%
        Private m_arguments As List(Of Object)
        Private m_backwardCallback As Action(Of Tensor, Tensor, List(Of Tensor))

        Public Property Data As Matrix
            Get
                Return Me.m_data
            End Get
            Set(value As Matrix)
                Me.m_data = value
            End Set
        End Property

        Public ReadOnly Property Creators As List(Of Tensor)
            Get
                Return Me.m_creators
            End Get
        End Property

        Public ReadOnly Property CreationOperation As TensorOperations
            Get
                Return Me.m_creationOperation
            End Get
        End Property

        Public Property Gradient As Tensor
            Get
                Return Me.m_gradient
            End Get
            Set(value As Tensor)
                Me.m_gradient = value
            End Set
        End Property

        Public ReadOnly Property AutoGrad As Boolean
            Get
                Return Me.m_autoGrad
            End Get
        End Property

        Public ReadOnly Property Id%
            Get
                Return Me.m_id
            End Get
        End Property

        Public Property Childrens As Dictionary(Of Integer, Integer)
            Get
                Return Me.m_childrens
            End Get
            Set(value As Dictionary(Of Integer, Integer))
                Me.m_childrens = value
            End Set
        End Property

        Public ReadOnly Property Arguments As List(Of Object)
            Get
                Return Me.m_arguments
            End Get
        End Property

        Public Sub New(data As Matrix,
            Optional autoGrad As Boolean = False,
            Optional creators As List(Of Tensor) = Nothing,
            Optional creationOperation As TensorOperations = TensorOperations.None,
            Optional arguments As List(Of Object) = Nothing,
            Optional backwardCallback As Action(
                Of Tensor, Tensor, List(Of Tensor)) = Nothing)

            Me.m_data = data
            Me.m_autoGrad = autoGrad
            Me.m_gradient = Nothing
            Me.m_arguments = arguments
            Me.m_backwardCallback = backwardCallback
            Me.m_id = idCount
            idCount += 1
            Me.m_creators = creators
            Me.m_creationOperation = creationOperation
            Me.m_childrens = New Dictionary(Of Integer, Integer)()

            If Me.Creators IsNot Nothing Then
                For Each creator In Me.Creators
                    If creator.m_childrens.ContainsKey(Id) Then
                        creator.m_childrens(Id) += 1
                    Else
                        creator.m_childrens.Add(Id, 1)
                    End If
                Next
            End If

        End Sub

        Private Function allChildrenGradsAccountedFor() As Boolean
            For Each child In m_childrens
                If child.Value <> 0 Then Return False
            Next
            Return True
        End Function

        Public Sub Backward(gradient As Tensor,
            Optional gradientOrigin As Tensor = Nothing)

            If Not Me.m_autoGrad Then Return

            If gradient Is Nothing Then _
                gradient = New Tensor(Matrix.Ones(m_data.x, m_data.y))

            If gradientOrigin IsNot Nothing Then
                If m_childrens(gradientOrigin.Id) = 0 Then _
                    Throw New ArgumentException("Cannot backprop more than once")
                m_childrens(gradientOrigin.Id) -= 1
            End If

            If Me.m_gradient Is Nothing Then
                Me.m_gradient = gradient
            Else
                Me.m_gradient = Tensor.Add(Me.m_gradient, gradient)
            End If

            If Me.m_creators IsNot Nothing AndAlso
                (allChildrenGradsAccountedFor() OrElse gradientOrigin Is Nothing) Then

                Select Case m_creationOperation
                    Case TensorOperations.None
                    Case TensorOperations.Addition
                        AdditionTensorOperation()
                    Case TensorOperations.Negation
                        NegationTensorOperation()
                    Case TensorOperations.Substraction
                        SubstractionTensorOperation()
                    Case TensorOperations.Multiplication
                        MultiplicationTensorOperation()
                    Case TensorOperations.Sumatory
                        SumatoryTensorOperation()
                    Case TensorOperations.Transpose
                        TransposeTensorOperation()
                    Case TensorOperations.MatrixMultiplication
                        MatrixMultiplicationTensorOperation()
                    Case TensorOperations.Expand
                        ExpandTensorOperation()
                    Case TensorOperations.Other
                        m_backwardCallback(Me, Me.m_gradient, Me.m_creators)
                    Case Else
                        Throw New ArgumentException(
                            "Invalid Creation operation: {CreationOperation}")
                End Select
            End If

        End Sub

        Public Overrides Function ToString$()
            Return Me.m_data.ToString()
        End Function

        Private Sub CheckCreatorsThrow(creatorNumber%)
            If Me.m_creators Is Nothing Then _
                Throw New ArgumentException("Creators can not be null")
            If Me.m_creators.Count <> creatorNumber Then _
                Throw New ArgumentException("Creator count must be 2 not {Creators.Count}")
        End Sub

        Private Sub CheckArgumentsThrow(argumentsNumber%)
            If Me.m_arguments Is Nothing Then
                Throw New ArgumentException("Arguments are null")
            End If
            If Me.m_arguments.Count <> argumentsNumber Then
                Throw New ArgumentException(
                    "Number of arguments must be {argumentsNumber}")
            End If
        End Sub

        Private Function CheckCreators(creatorNumber%) As Boolean
            If Me.m_creators Is Nothing Then Return False
            If Me.m_creators.Count <> creatorNumber Then Return False
            Return True
        End Function

        Private Sub SumatoryTensorOperation()

            CheckCreatorsThrow(1)
            CheckArgumentsThrow(1)
            Dim dimension = CType(m_arguments(0), AxisZero)

            Dim copies% = 0
            If dimension = AxisZero.horizontal Then
                copies = CInt(Me.m_creators(0).m_data.y)
            Else
                copies = CInt(Me.m_creators(0).m_data.x)
            End If

            Me.m_creators(0).Backward(Tensor.Expand(Me.m_gradient, dimension, copies))

        End Sub

        Private Sub MatrixMultiplicationTensorOperation()
            Me.m_creators(0).Backward(
                Tensor.MatMul(m_gradient, Tensor.Transp(Me.m_creators(1))))
            Me.m_creators(1).Backward(
                Tensor.Transp(Tensor.MatMul(Tensor.Transp(Me.m_gradient), Me.m_creators(0))))
        End Sub

        Private Sub TransposeTensorOperation()
            CheckCreatorsThrow(1)
            Me.m_creators(0).Backward(Tensor.Transp(Me.m_gradient))
        End Sub

        Private Sub MultiplicationTensorOperation()
            CheckCreatorsThrow(2)
            Me.m_creators(0).Backward(Tensor.Mul(Me.m_gradient, Me.m_creators(1)), Me)
            Me.m_creators(1).Backward(Tensor.Mul(Me.m_gradient, Me.m_creators(0)), Me)
        End Sub

        Private Sub SubstractionTensorOperation()
            CheckCreatorsThrow(2)
            Me.m_creators(0).Backward(Me.m_gradient, Me)
            Me.m_creators(1).Backward(Tensor.Neg(Me.m_gradient), Me)
        End Sub

        Private Sub NegationTensorOperation()
            CheckCreatorsThrow(1)
            Me.m_creators(0).Backward(Tensor.Neg(Me.m_gradient), Me)
        End Sub

        Private Sub AdditionTensorOperation()
            CheckCreatorsThrow(2)
            Me.m_creators(0).Backward(Me.m_gradient, Me)
            Me.m_creators(1).Backward(Me.m_gradient, Me)
        End Sub

        Private Sub ExpandTensorOperation()
            CheckCreatorsThrow(1)
            CheckArgumentsThrow(1)
            Dim dimension = CType(Me.m_arguments(0), AxisZero)
            Me.m_creators(0).Backward(Tensor.Sum(Me.m_gradient, dimension))
        End Sub

        Public Shared Function Expand(A As Tensor, axis0 As AxisZero, copies%) As Tensor

            Dim m As New Matrix()

            If axis0 = AxisZero.horizontal Then
                m = Matrix.Zeros(A.m_data.x, copies)
                Matrix.MatrixLoop(Sub(i, j)
                                      m(i, j) = A.m_data(i, 0)
                                  End Sub, A.m_data.x, copies)
            ElseIf axis0 = AxisZero.vertical Then
                m = Matrix.Zeros(copies, A.m_data.y)
                Matrix.MatrixLoop(Sub(i, j)
                                      m(i, j) = A.m_data(0, j)
                                  End Sub, copies, A.m_data.y)
            End If

            If A.m_autoGrad Then
                Dim Creators = New List(Of Tensor)() From {A}
                Dim Argument = New List(Of Object)() From {axis0}
                Return New Tensor(m, autoGrad:=True, creators:=Creators,
                     creationOperation:=TensorOperations.Expand, arguments:=Argument)
            End If

            Return New Tensor(m)

        End Function

        Public Shared Function Neg(A As Tensor) As Tensor

            If A.m_autoGrad Then
                Dim Creators = New List(Of Tensor)() From {A}
                Return New Tensor(A.m_data * -1.0F, autoGrad:=True,
                    creators:=Creators, creationOperation:=TensorOperations.Negation)
            End If
            Return New Tensor(A.m_data * -1.0F)

        End Function

        Public Shared Function Add(A As Tensor, B As Tensor) As Tensor

            If A.m_autoGrad AndAlso B.m_autoGrad Then
                Dim Creators = New List(Of Tensor)() From {A, B}
                Return New Tensor(A.m_data + B.m_data, autoGrad:=True,
                    creators:=Creators, creationOperation:=TensorOperations.Addition)
            End If
            Return New Tensor(A.m_data + B.m_data)

        End Function

        Public Shared Function Substract(A As Tensor, B As Tensor) As Tensor

            If A.m_autoGrad AndAlso B.m_autoGrad Then
                Dim Creators = New List(Of Tensor)() From {A, B}
                Return New Tensor(A.m_data - B.m_data, autoGrad:=True,
                    creators:=Creators, creationOperation:=TensorOperations.Substraction)
            End If
            Return New Tensor(A.m_data - B.m_data)

        End Function

        Public Shared Function Mul(A As Tensor, B As Tensor) As Tensor

            If A.m_autoGrad AndAlso B.m_autoGrad Then
                Dim Creators = New List(Of Tensor)() From {A, B}
                Return New Tensor(Matrix.DeltaMult(A.m_data, B.m_data),
                    autoGrad:=True, creators:=Creators,
                    creationOperation:=TensorOperations.Multiplication)
            End If
            Return New Tensor(Matrix.DeltaMult(A.m_data, B.m_data))

        End Function

        Public Shared Function Sum(A As Tensor, axis0 As AxisZero) As Tensor

            If A.m_autoGrad Then
                Dim Argument = New List(Of Object)() From {axis0}
                Dim Creators = New List(Of Tensor)() From {A}
                Return New Tensor(A.m_data.Sumatory(axis0), autoGrad:=True,
                    creators:=Creators, creationOperation:=TensorOperations.Sumatory,
                    arguments:=Argument)
            End If
            Return New Tensor(A.m_data.Sumatory(axis0))

        End Function

        Public Shared Function Transp(A As Tensor) As Tensor

            If A.m_autoGrad Then
                Dim Creators = New List(Of Tensor)() From {A}
                Return New Tensor(A.m_data.T, autoGrad:=True, creators:=Creators,
                    creationOperation:=TensorOperations.Transpose)
            End If
            Return New Tensor(A.m_data.T)

        End Function

        Public Shared Function MatMul(A As Tensor, B As Tensor) As Tensor

            If A.m_autoGrad AndAlso B.m_autoGrad Then
                Dim Creators = New List(Of Tensor)() From {A, B}
                Return New Tensor(Matrix.MatMult(A.m_data, B.m_data),
                    autoGrad:=True, creators:=Creators,
                    creationOperation:=TensorOperations.MatrixMultiplication)
            End If
            Return New Tensor(Matrix.MatMult(A.m_data, B.m_data))

        End Function

    End Class

End Namespace