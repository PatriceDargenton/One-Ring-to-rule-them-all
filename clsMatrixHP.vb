
' From https://github.com/HectorPulido/Vectorized-multilayer-neural-network : C# -> VB .NET conversion
'  and for Axis, AxisZero and Item (this function) from:
' https://github.com/HectorPulido/Machine-learning-Framework-Csharp : C# -> VB .NET conversion

Imports System.Text ' StringBuilder
Imports System.Threading.Tasks ' Parallel.For

Namespace Utility

    ' Hector Pulido implementation's : 12 sec
    ' Nikos Labiris implementation's : 7.4 sec : 1.5 times faster
#Const Implementation = 0 ' 0 : Off, 1 : On

    ' From https://github.com/HectorPulido/Machine-learning-Framework-Csharp 
    Public Enum Axis
        horizontal
        vertical
    End Enum

    Public Enum AxisZero
        horizontal = 0
        vertical = 1
        none = -1
    End Enum

#If Implementation Then

    Public Class Matrix

        Private data#(,)

#Region "Properties"

        ' From https://github.com/HectorPulido/Machine-learning-Framework-Csharp (double this[int i, int j])
        Default Public Property Item#(r%, c%)
            Get
                Return Me.data(r, c)
            End Get
            Set(value#)
                Me.data(r, c) = value
            End Set
        End Property

        Public Property matrixP As Double(,)
            Get
                Return CType(Me.data.Clone(), Double(,))
            End Get
            Set(value As Double(,))
                Me.data = value
            End Set
        End Property

        Public ReadOnly Property isDefined As Boolean
            Get
                Return Not IsNothing(Me.data)
            End Get
        End Property

        ''' <summary>
        ''' Rows
        ''' </summary>
        Public ReadOnly Property r%
            Get
                Return Me.data.GetLength(0)
            End Get
        End Property

        ''' <summary>
        ''' Columns
        ''' </summary>
        Public ReadOnly Property c%
            Get
                Return Me.data.GetLength(1)
            End Get
        End Property

        ''' <summary>
        ''' Transpose
        ''' </summary>
        Public ReadOnly Property T As Matrix
            Get
                Return Transpose(Me)
            End Get
        End Property

        Public ReadOnly Property Size As Matrix
            Get
                Return (New Double(,) {{Me.r, Me.c}})
            End Get
        End Property

        Public ReadOnly Property Abs As Matrix
            Get
                Return Abs_()
            End Get
        End Property

        Public ReadOnly Property Average#
            Get
                Return Average_()
            End Get
        End Property

#End Region

#Region "Constructors"

        Public Sub New() ' Constructor used by Tensor
        End Sub

        Public Sub New(rows%, columns%)
            Me.data = New Double(rows - 1, columns - 1) {}
        End Sub

        Public Sub New(matrix0#(,))
            Me.data = matrix0
        End Sub

        Public Sub New(matrix0!(,))
            Dim rows = matrix0.GetLength(0)
            Dim columns = matrix0.GetLength(1)
            Me.data = New Double(rows - 1, columns - 1) {}
            For i = 0 To rows - 1
                For j = 0 To columns - 1
                    Me.data(i, j) = matrix0(i, j)
                Next
            Next
        End Sub

        ' From clsMatrix: https://github.com/PatriceDargenton/Matrix-MultiLayerPerceptron

        ''' <summary>
        ''' Create a matrix object from an array of Single
        ''' </summary>
        Public Shared Function FromArraySingle(inputs!()) As Matrix
            Dim m As New Matrix(inputs.Length, 1)
            For i = 0 To inputs.Length - 1
                m.data(i, 0) = inputs(i)
            Next
            Return m
        End Function

#End Region

#Region "Operators"

        ' Implicit conversion operator #(,) -> Matrix
        Public Shared Widening Operator CType(matrix0#(,)) As Matrix
            Return New Matrix(matrix0)
        End Operator

        ' Implicit conversion operator Matrix -> #(,)
        Public Shared Widening Operator CType(matrix0 As Matrix) As Double(,)
            Return matrix0.matrixP
        End Operator

        ' Implicit conversion operator !(,) -> Matrix
        Public Shared Widening Operator CType(matrix0!(,)) As Matrix
            Return New Matrix(matrix0)
        End Operator

        Public Shared Operator +(m1 As Matrix, m2 As Matrix) As Matrix
            Dim m As Matrix = MatSum(m1, m2)
            Return m
        End Operator

        'Public Shared Operator +(m2 As Matrix, m1#) As Matrix
        '    Dim m As Matrix = MatdoubleSum(m1, m2)
        '    Return m
        'End Operator

        Public Shared Operator -(m1 As Matrix, m2 As Matrix) As Matrix
            Dim m As Matrix = MatSum(m1, m2, neg:=True)
            Return m
        End Operator

        Public Shared Operator -(m2 As Matrix, m1#) As Matrix
            Dim m As Matrix = MatdoubleSum(-m1, m2)
            Return m
        End Operator

        Public Shared Operator *(m2 As Matrix, m1#) As Matrix
            Dim m As Matrix = MatdoubleMult(m2, m1)
            Return m
        End Operator

        Public Shared Operator *(m1 As Matrix, m2 As Matrix) As Matrix
            If m1.r = m2.r AndAlso m1.c = m2.c Then
                Dim m3 As Matrix = DeltaMult(m1, m2)
                Return m3
            End If
            Dim m4 As Matrix = MatMult(m1, m2)
            Return m4
        End Operator

        Public Shared Operator /(m2 As Matrix, m1#) As Matrix
            Return MatdoubleMult(m2, 1 / m1)
        End Operator

        Public Shared Operator ^(m2 As Matrix, m1#) As Matrix
            Return Pow(m2, m1)
        End Operator

#End Region

#Region "Public shared operations"

        Public Shared Function Zeros(r%, c%) As Matrix

            Dim zeros0 = New Double(r - 1, c - 1) {}
            MatrixLoop((Sub(i, j) zeros0(i, j) = 0), r, c)
            Dim m As Matrix = zeros0
            Return m

        End Function

        Public Shared Function Ones(r%, c%) As Matrix

            Dim ones0 = New Double(r - 1, c - 1) {}
            MatrixLoop((Sub(i, j) ones0(i, j) = 1), r, c)
            Dim m As Matrix = ones0
            Return m

        End Function

        Public Shared Function Identy(x%) As Matrix

            Dim identy0 = New Double(x - 1, x - 1) {}

            MatrixLoop(
                (Sub(i, j)
                     If i = j Then
                         identy0(i, j) = 1
                     Else
                         identy0(i, j) = 0
                     End If
                 End Sub), x, x)

            Dim m As Matrix = identy0
            Return m

        End Function

        Public Shared Function Transpose(m As Matrix) As Matrix

            Dim mT = New Double(m.c - 1, m.r - 1) {}
            'MatrixLoop((Sub(i, j) mT(j, i) = m.matrixP(i, j)), m.r, m.c)
            MatrixLoop((Sub(i, j) mT(j, i) = m.data(i, j)), m.r, m.c)
            Dim result As Matrix = mT
            Return result

        End Function

        ''' <summary>
        ''' Transpose and multiply this transposed matrix by m
        ''' </summary>
        Public Shared Function TransposeAndMultiply1(original As Matrix, m As Matrix) As Matrix
            Dim original_t As Matrix = Transpose(original)
            Dim result As Matrix = original_t * m
            Return result
        End Function

        ''' <summary>
        ''' Transpose and multiply a matrix m by this transposed one
        ''' </summary>
        Public Shared Function TransposeAndMultiply2(original As Matrix, m As Matrix) As Matrix
            Dim original_t As Matrix = Transpose(original)
            Dim result As Matrix = m * original_t
            Return result
        End Function

        ''' <summary>
        ''' Subtract 2 matrices (the first as an array of Single) and return a new matrix
        ''' </summary>
        Public Shared Function SubtractFromArraySingle(a_array!(), b As Matrix) As Matrix
            Dim a As Matrix = FromArraySingle(a_array)
            Dim result As Matrix = a - b
            Return result
        End Function

        ''' <summary>
        ''' Multiply matrices a and b, add matrix c,
        '''  and apply a function to every element of the result
        ''' </summary>
        Public Shared Function MultiplyAddAndMap(
            a As Matrix, b As Matrix, c As Matrix,
            lambdaFct As Func(Of Double, Double)) As Matrix

            Dim d As Matrix = MatMult(a, b)
            d += c
            d = Map(d, lambdaFct)

            Return d

        End Function

        ''' <summary>
        ''' Multiply matrices a and b, and apply a function to every element of the result
        ''' </summary>
        Public Shared Function MultiplyAndMap(a As Matrix, b As Matrix,
            lambdaFct As Func(Of Double, Double)) As Matrix

            Dim d As Matrix = MatMult(a, b)
            d = Map(d, lambdaFct)

            Return d

        End Function

        ''' <summary>
        ''' Apply a function to each element of the array
        ''' </summary>
        Public Shared Function Map(m As Matrix, lambdaFct As Func(Of Double, Double)) As Matrix

            Dim c As New Matrix(m.r, m.c)
            For i = 0 To m.r - 1
                For j = 0 To m.c - 1
                    c.data(i, j) = lambdaFct.Invoke(m.data(i, j))
                Next
            Next
            Return c

        End Function

        Public Shared Function Sumatory(m As Matrix,
            Optional dimension As AxisZero = AxisZero.none) As Matrix

            Dim output#(,)

            If dimension = AxisZero.none Then
                output = New Double(0, 0) {}
            ElseIf dimension = AxisZero.horizontal Then
                output = New Double(m.r - 1, 0) {}
            ElseIf dimension = AxisZero.vertical Then
                output = New Double(0, m.c - 1) {}
            Else
                Throw New ArgumentException("The dimension must be -1, 0 or 1")
            End If

            'If dimension = AxisZero.none Then
            '    MatrixLoop((Sub(i, j) output(0, 0) += m.matrixP(i, j)), m.r, m.c)
            'ElseIf dimension = AxisZero.horizontal Then
            '    MatrixLoop((Sub(i, j) output(i, 0) += m.matrixP(i, j)), m.r, m.c)
            'ElseIf dimension = AxisZero.vertical Then
            '    MatrixLoop((Sub(i, j) output(0, j) += m.matrixP(i, j)), m.r, m.c)
            'End If
            If dimension = AxisZero.none Then
                MatrixLoop((Sub(i, j) output(0, 0) += m.data(i, j)), m.r, m.c)
            ElseIf dimension = AxisZero.horizontal Then
                MatrixLoop((Sub(i, j) output(i, 0) += m.data(i, j)), m.r, m.c)
            ElseIf dimension = AxisZero.vertical Then
                MatrixLoop((Sub(i, j) output(0, j) += m.data(i, j)), m.r, m.c)
            End If

            Dim result As Matrix = output
            Return result

        End Function

        Public Shared Function Pow(m2 As Matrix, m1#) As Matrix

            Dim output = New Double(m2.r - 1, m2.c - 1) {}
            'MatrixLoop((Sub(i, j) output(i, j) = Math.Pow(m2.matrixP(i, j), m1)), m2.r, m2.c)
            MatrixLoop((Sub(i, j) output(i, j) = Math.Pow(m2.data(i, j), m1)), m2.r, m2.c)
            Dim result As Matrix = output
            Return result

        End Function

        Public Shared Function Dot(m1 As Matrix, m2 As Matrix) As Matrix
            Return m1 * m2.T
        End Function

        Public Shared Sub MatrixLoop(e As Action(Of Integer, Integer), r%, c%)

            For i = 0 To r - 1
                For j = 0 To c - 1
                    e(i, j)
                Next
            Next

            ' Not stable in this loop:
            'Parallel.For(0, r,
            '    Sub(i)
            '        Parallel.For(0, c,
            '            Sub(j)
            '                e(i, j)
            '            End Sub)
            '    End Sub)

        End Sub

#End Region

#Region "Public operations"

        Public Function AddColumn(m2 As Matrix) As Matrix

            'If Me.matrixP Is Nothing Then Throw New ArgumentException("Matrix can not be null")
            If Me.data Is Nothing Then Throw New ArgumentException("Matrix can not be null")
            If m2.c <> 1 OrElse m2.r <> r Then Throw New ArgumentException("Invalid dimensions")

            Dim newMatrix = New Double(Me.r - 1, Me.c + 1 - 1) {}
            'Dim m = Me.matrixP
            For i = 0 To Me.r - 1
                'newMatrix(i, 0) = m2.matrixP(i, 0)
                newMatrix(i, 0) = m2.data(i, 0)
            Next
            'MatrixLoop((Sub(i, j) newMatrix(i, j + 1) = m(i, j)), Me.r, Me.c)
            MatrixLoop((Sub(i, j) newMatrix(i, j + 1) = Me.data(i, j)), Me.r, Me.c)

            Return newMatrix

        End Function

        'Public Function AddRow(m2 As Matrix) As Matrix

        '    'If Me.matrixP Is Nothing Then Throw New ArgumentException("Matrix can not be null")
        '    If Me.data Is Nothing Then Throw New ArgumentException("Matrix can not be null")
        '    If m2.r <> 1 OrElse m2.c <> Me.c Then Throw New ArgumentException("Invalid dimensions")

        '    Dim newMatrix = New Double(Me.r + 1 - 1, Me.c - 1) {}
        '    Dim m = Me.matrixP
        '    For j = 0 To Me.c - 1
        '        'newMatrix(0, j) = m2.matrixP(0, j)
        '        newMatrix(0, j) = m2.data(0, j)
        '    Next
        '    MatrixLoop((Sub(i, j) newMatrix(i + 1, j) = m(i, j)), Me.r, Me.c)

        '    Return newMatrix

        'End Function

        Public Function Sumatory(Optional dimension As AxisZero = AxisZero.none) As Matrix
            Return Sumatory(Me, dimension)
        End Function

        ''' <summary>
        ''' Cut matrix from r1, c1 to r2, c2
        ''' </summary>
        Public Function Slice(r1%, c1%, r2%, c2%) As Matrix

            'If Me.matrixP Is Nothing Then Throw New ArgumentException("Matrix can not be null")
            If Me.data Is Nothing Then Throw New ArgumentException("Matrix can not be null")

            If r1 >= r2 OrElse c1 >= c2 OrElse r1 < 0 OrElse
               r2 < 0 OrElse c1 < 0 OrElse c2 < 0 Then _
               Throw New ArgumentException("Dimensions are not valid")

            Dim slice0 = New Double(r2 - r1 - 1, c2 - c1 - 1) {}
            For i = r1 To r2 - 1
                For j = c1 To c2 - 1
                    'slice0(i - r1, j - c1) = Me.matrixP(i, j)
                    slice0(i - r1, j - c1) = Me.data(i, j)
                Next
            Next

            Return slice0

        End Function

        'Public Function Slice(r%, c%) As Matrix
        '    Return Slice(0, 0, r, c)
        'End Function

        'Public Function Pow(m1#) As Matrix
        '    Return Pow(Me, m1)
        'End Function

        'Public Function Dot(m As Matrix) As Matrix
        '    Return Dot(Me, m)
        'End Function

#End Region

#Region "Private operations"

        Private Shared Function MatdoubleSum(m1#, m2 As Matrix) As Matrix

            Dim a#(,) = m2
            Dim b = New Double(m2.r - 1, m2.c - 1) {}
            MatrixLoop((Sub(i, j) b(i, j) = a(i, j) + m1), b.GetLength(0), b.GetLength(1))
            Dim m As Matrix = b
            Return m

        End Function

        Private Shared Function MatSum(m1 As Matrix, m2 As Matrix,
            Optional neg As Boolean = False) As Matrix

            If m1.r <> m2.r OrElse m1.c <> m2.c Then _
                Throw New ArgumentException("Matrix must have the same dimensions")

            Dim a#(,) = m1
            Dim b#(,) = m2
            Dim c = New Double(m1.r - 1, m2.c - 1) {}

            MatrixLoop(
                (Sub(i, j)
                     If Not neg Then
                         c(i, j) = a(i, j) + b(i, j)
                     Else
                         c(i, j) = a(i, j) - b(i, j)
                     End If
                 End Sub), c.GetLength(0), c.GetLength(1))

            Dim m As Matrix = c
            Return m

        End Function

        Private Shared Function MatdoubleMult(m2 As Matrix, m1#) As Matrix

            Dim a#(,) = m2
            Dim b = New Double(m2.r - 1, m2.c - 1) {}
            MatrixLoop((Sub(i, j) b(i, j) = a(i, j) * m1), b.GetLength(0), b.GetLength(1))
            Dim m As Matrix = b
            Return m

        End Function

        Private Shared Function MatMult(m1 As Matrix, m2 As Matrix) As Matrix

            If m1.c <> m2.r Then _
                Throw New ArgumentException("Matrix must have compatible dimensions")

            Dim n = m1.r
            Dim m = m1.c
            Dim p = m2.c
            Dim a#(,) = m1
            Dim b#(,) = m2
            Dim c = New Double(n - 1, p - 1) {}

            MatrixLoop(
                (Sub(i, j)
                     Dim sum# = 0
                     For k = 0 To m - 1
                         sum += a(i, k) * b(k, j)
                     Next
                     c(i, j) = sum
                 End Sub), n, p)

            Dim result As Matrix = c
            Return result

        End Function

        Private Shared Function DeltaMult(m1 As Matrix, m2 As Matrix) As Matrix

            If m1.r <> m2.r OrElse m1.c <> m2.c Then _
                Throw New ArgumentException("Matrix must have the same dimensions")

            Dim output = New Double(m1.r - 1, m2.c - 1) {}
            'MatrixLoop((Sub(i, j) output(i, j) = m1.matrixP(i, j) * m2.matrixP(i, j)), m1.r, m2.c)
            MatrixLoop((Sub(i, j) output(i, j) = m1.data(i, j) * m2.data(i, j)), m1.r, m2.c)
            Dim m As Matrix = output
            Return m

        End Function

        ''' <summary>
        ''' Compute absolute values of a matrix
        ''' </summary>
        Private Function Abs_() As Matrix

            Dim d#(,) = Me
            'MatrixLoop((Sub(i, j) d(i, j) = Math.Abs(m.matrixP(i, j))), m.r, m.c)
            MatrixLoop((Sub(i, j) d(i, j) = Math.Abs(Me.data(i, j))), Me.r, Me.c)
            Dim result As Matrix = d
            Return result

        End Function

        ''' <summary>
        ''' Compute average value of the matrix
        ''' </summary>
        Private Function Average_#()

            Dim d# = 0
            'MatrixLoop((Sub(i, j) d += m.matrixP(i, j)), m.r, m.c)
            MatrixLoop((Sub(i, j) d += Me.data(i, j)), Me.r, Me.c)
            Dim aver# = d / (Me.r * Me.c)
            Return aver

        End Function

#End Region

#Region "Exports"

        ''' <summary>
        ''' Override <c>ToString()</c> method to pretty-print the matrix
        ''' </summary>
        Public Overrides Function ToString$()
            Return ToStringWithFormat()
        End Function

        Public Function ToStringWithFormat$(Optional dec$ = format2Dec)

            Dim sb As New StringBuilder()
            sb.AppendLine("{")
            For i = 0 To Me.r - 1
                sb.Append(" {")
                For j = 0 To Me.c - 1
                    Dim strVal$ = Me.data(i, j).ToString(dec).ReplaceCommaByDot()
                    sb.Append(strVal)
                    If j < Me.c - 1 Then sb.Append(", ")
                Next
                sb.Append("}")
                If i < Me.r - 1 Then sb.Append("," & vbLf)
            Next
            sb.Append("}")

            Dim s$ = sb.ToString
            Return s

        End Function

        ''' <summary>
        ''' Convert whole Matrix object to array
        ''' </summary>
        Public Function ToArraySingle() As Single()

            Dim array!() = New Single(Me.data.Length - 1) {}
            Dim k = 0
            For i = 0 To Me.r - 1
                For j = 0 To Me.c - 1
                    array(k) = CSng(Me.data(i, j))
                    k += 1
                Next
            Next
            Return array

        End Function

#End Region

#Region "Miscellaneous"

        'Public Sub SetValue(r%, c%, value#)

        '    'If Me.matrixP Is Nothing Then Throw New ArgumentException("Matrix can not be null")
        '    If Me.data Is Nothing Then Throw New ArgumentException("Matrix can not be null")
        '    Me.data(r, c) = value

        'End Sub

        Public Function GetValue#(r%, c%)

            'If Me.matrixP Is Nothing Then Throw New ArgumentException("Matrix can not be null")
            If Me.data Is Nothing Then Throw New ArgumentException("Matrix can not be null")
            'Return Me.matrixP(r, c)
            Return Me.data(r, c)

        End Function

        Public Function GetRow(r%) As Matrix

            'If Me.matrixP Is Nothing Then Throw New ArgumentException("Matrix can not be null")
            If Me.data Is Nothing Then Throw New ArgumentException("Matrix can not be null")
            Dim row = New Double(0, Me.c - 1) {}

            For j = 0 To Me.c - 1
                'row(0, j) = Me.matrixP(r, j)
                row(0, j) = Me.data(r, j)
            Next

            Return row

        End Function

        Public Function GetColumn(c%) As Matrix

            'If Me.matrixP Is Nothing Then Throw New ArgumentException("Matrix can not be null")
            If Me.data Is Nothing Then Throw New ArgumentException("Matrix can not be null")
            Dim column = New Double(Me.r - 1, 0) {}

            For i = 0 To Me.r - 1
                'column(i, 0) = Me.matrixP(i, c)
                column(i, 0) = Me.data(i, c)
            Next

            Return column

        End Function

        Public Shared Function Randomize(r%, c%, rnd As Random,
            Optional minValue! = 0, Optional maxValue! = 1) As Matrix

            Dim random_ = New Double(r - 1, c - 1) {}
            'MatrixLoop((Sub(i, j) random_(i, j) = rnd.NextDouble), x, y)
            MatrixLoop((Sub(i, j) random_(i, j) =
                Math.Round(rnd.NextDouble(minValue, maxValue),
                clsMLPGeneric.roundWeights)), r, c)
            Dim m As Matrix = random_
            Return m

        End Function

        Public Sub Randomize(rnd As Random,
            Optional minValue! = 0, Optional maxValue! = 1)

            MatrixLoop((Sub(i, j) Me.data(i, j) =
                Math.Round(rnd.NextDouble(minValue, maxValue),
                clsMLPGeneric.roundWeights)), Me.r, Me.c)

        End Sub

#End Region

    End Class

#End If

End Namespace