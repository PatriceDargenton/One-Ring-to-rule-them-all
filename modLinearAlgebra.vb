
' From https://github.com/HectorPulido/Vectorized-multilayer-neural-network : C# -> VB .NET conversion
'  and for Axis, AxisZero and Item (this function) from:
' https://github.com/HectorPulido/Machine-learning-Framework-Csharp : C# -> VB .NET conversion

Imports System.Text ' StringBuilder

Namespace Util

    ' From https://github.com/HectorPulido/Machine-learning-Framework-Csharp 
    Public Enum Axis
        horizontal
        vertical
    End Enum

    Public Enum AxisZero
        horizontal
        vertical
        none
    End Enum

    Public Structure Matrix

        Private _matrix#(,)

        Public Property matrix As Double(,)
            Get
                Return CType(_matrix.Clone, Double(,))
            End Get
            Set(value As Double(,))
                _matrix = value
            End Set
        End Property

        Public ReadOnly Property isDefined As Boolean
            Get
                Return Not IsNothing(_matrix)
            End Get
        End Property

        Public ReadOnly Property x%
            Get
                Return _matrix.GetLength(0)
            End Get
        End Property

        Public ReadOnly Property y%
            Get
                Return _matrix.GetLength(1)
            End Get
        End Property

        Public ReadOnly Property T As Matrix
            Get
                Return Transpose(Me)
            End Get
        End Property

        Public ReadOnly Property size As Matrix
            Get
                Return (New Double(,) {{x, y}})
            End Get
        End Property

        Public ReadOnly Property abs As Matrix
            Get
                Return Abs_(Me)
            End Get
        End Property

        Public ReadOnly Property average#
            Get
                Return Average_(Me)
            End Get
        End Property

        Public Sub New(sizex%, sizey%)
            _matrix = New Double(sizex - 1, sizey - 1) {}
        End Sub

        Public Sub New(matrix#(,))
            _matrix = matrix
        End Sub

        Public Sub New(matrix!(,))
            Dim x = matrix.GetLength(0)
            Dim y = matrix.GetLength(1)
            _matrix = New Double(x - 1, y - 1) {}
            For i = 0 To x - 1
                For j = 0 To y - 1
                    _matrix(i, j) = matrix(i, j)
                Next
            Next
        End Sub

        ' Implicit conversion operator !(,) -> Matrix
        Public Shared Widening Operator CType(matrix!(,)) As Matrix
            Return New Matrix(matrix)
        End Operator

        ' Implicit conversion operator #(,) -> Matrix
        Public Shared Widening Operator CType(matrix#(,)) As Matrix
            Return New Matrix(matrix)
        End Operator

        ' Implicit conversion operator Matrix -> #(,)
        Public Shared Widening Operator CType(matrix As Matrix) As Double(,)
            Return matrix.matrix
        End Operator

        Public Sub SetValue(x%, y%, value#)

            If matrix Is Nothing Then Throw New ArgumentException("Matrix can not be null")
            _matrix(x, y) = value

        End Sub

        Public Function GetValue#(x%, y%)

            If matrix Is Nothing Then Throw New ArgumentException("Matrix can not be null")
            Return matrix(x, y)

        End Function

        ' From https://github.com/HectorPulido/Machine-learning-Framework-Csharp (double this[int i, int j])
        Default Public Property Item#(i%, j%)
            Get
                Return _matrix(i, j)
            End Get
            Set(value#)
                _matrix(i, j) = value
            End Set
        End Property

        ''' <summary>
        ''' Cut matrix from x1, y1 to x2, y2
        ''' </summary>
        Public Function Slice(x1%, y1%, x2%, y2%) As Matrix

            If matrix Is Nothing Then Throw New ArgumentException("Matrix can not be null")

            If x1 > x2 OrElse y1 > y2 OrElse x1 < 0 OrElse
               x2 < 0 OrElse y1 < 0 OrElse y2 < 0 Then _
               Throw New ArgumentException("Dimensions are not valid")

            Dim slice0 = New Double(x2 - x1 - 1, y2 - y1 - 1) {}
            For i = x1 To x2 - 1
                For j = y1 To y2 - 1
                    slice0(i - x1, j - y1) = matrix(i, j)
                Next
            Next

            Return slice0

        End Function

        Public Function Slice(x%, y%) As Matrix
            Return Slice(0, 0, x, y)
        End Function

        Public Function GetRow(x%) As Matrix

            If matrix Is Nothing Then Throw New ArgumentException("Matrix can not be null")
            Dim row = New Double(0, y - 1) {}

            For j = 0 To y - 1
                row(0, j) = matrix(x, j)
            Next

            Return row

        End Function

        Public Function GetColumn(y%) As Matrix

            If matrix Is Nothing Then Throw New ArgumentException("Matrix can not be null")
            Dim column = New Double(x - 1, 0) {}

            For i = 0 To x - 1
                column(i, 0) = matrix(i, y)
            Next

            Return column

        End Function

        Public Function AddColumn(m2 As Matrix) As Matrix

            If matrix Is Nothing Then Throw New ArgumentException("Matrix can not be null")
            If m2.y <> 1 OrElse m2.x <> x Then Throw New ArgumentException("Invalid dimensions")

            Dim newMatrix = New Double(x - 1, y + 1 - 1) {}
            Dim m = matrix
            For i = 0 To x - 1
                newMatrix(i, 0) = m2.matrix(i, 0)
            Next
            MatrixLoop((Sub(i, j) newMatrix(i, j + 1) = m(i, j)), x, y)

            Return newMatrix

        End Function

        Public Function AddRow(m2 As Matrix) As Matrix

            If matrix Is Nothing Then Throw New ArgumentException("Matrix can not be null")
            If m2.x <> 1 OrElse m2.y <> y Then Throw New ArgumentException("Invalid dimensions")

            Dim newMatrix = New Double(x + 1 - 1, y - 1) {}
            Dim m = matrix
            For j = 0 To y - 1
                newMatrix(0, j) = m2.matrix(0, j)
            Next
            MatrixLoop((Sub(i, j) newMatrix(i + 1, j) = m(i, j)), x, y)

            Return newMatrix

        End Function

        Public Overrides Function ToString$()
            Return ToStringWithFormat()
        End Function

        Public Function ToStringWithFormat$(Optional dec$ = format2Dec)

            Dim sb As New StringBuilder()
            sb.AppendLine("{")
            For i = 0 To x - 1
                sb.Append(" {")
                For j = 0 To y - 1
                    Dim sVal$ = matrix(i, j).ToString(dec).ReplaceCommaByDot()
                    sb.Append(sVal)
                    If j < y - 1 Then sb.Append(", ")
                Next
                sb.Append("}")
                If i < x - 1 Then sb.Append("," & vbLf)
            Next
            sb.Append("}")
            Return sb.ToString()

        End Function

        Public Shared Function Zeros(x%, y%) As Matrix

            Dim zeros0 = New Double(x - 1, y - 1) {}
            MatrixLoop((Sub(i, j) zeros0(i, j) = 0), x, y)
            Return zeros0

        End Function

        Public Shared Function Ones(x%, y%) As Matrix

            Dim ones0 = New Double(x - 1, y - 1) {}
            MatrixLoop((Sub(i, j) ones0(i, j) = 1), x, y)
            Return ones0

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

            Return identy0

        End Function

        Public Shared Function Random(x%, y%, r As Random,
            Optional minValue! = 0, Optional maxValue! = 1) As Matrix

            Dim random_ = New Double(x - 1, y - 1) {}
            'MatrixLoop((Sub(i, j) random_(i, j) = r.NextDouble), x, y)
            MatrixLoop((Sub(i, j) random_(i, j) =
                Math.Round(r.NextDouble(minValue, maxValue),
                clsMLPGeneric.roundWeights)), x, y)
            Return random_

        End Function

        Public Shared Function Transpose(m As Matrix) As Matrix

            Dim mT = New Double(m.y - 1, m.x - 1) {}
            MatrixLoop((Sub(i, j) mT(j, i) = m.matrix(i, j)), m.x, m.y)
            Return mT

        End Function

        Public Shared Operator +(m1 As Matrix, m2 As Matrix) As Matrix
            Return MatSum(m1, m2)
        End Operator

        Public Shared Operator +(m2 As Matrix, m1#) As Matrix
            Return MatdoubleSum(m1, m2)
        End Operator

        Public Shared Function MatdoubleSum(m1#, m2 As Matrix) As Matrix

            Dim a#(,) = m2
            Dim b = New Double(m2.x - 1, m2.y - 1) {}
            MatrixLoop((Sub(i, j) b(i, j) = a(i, j) + m1), b.GetLength(0), b.GetLength(1))
            Return b

        End Function

        Public Shared Function MatSum(m1 As Matrix, m2 As Matrix, Optional neg As Boolean = False) As Matrix

            If m1.x <> m2.x OrElse m1.y <> m2.y Then _
                Throw New ArgumentException("Matrix must have the same dimensions")

            Dim a#(,) = m1
            Dim b#(,) = m2
            Dim c = New Double(m1.x - 1, m2.y - 1) {}

            MatrixLoop(
                (Sub(i, j)
                     If Not neg Then
                         c(i, j) = a(i, j) + b(i, j)
                     Else
                         c(i, j) = a(i, j) - b(i, j)
                     End If
                 End Sub), c.GetLength(0), c.GetLength(1))

            Return c

        End Function

        Public Shared Operator -(m1 As Matrix, m2 As Matrix) As Matrix
            Return MatSum(m1, m2, neg:=True)
        End Operator

        Public Shared Operator -(m2 As Matrix, m1#) As Matrix
            Return MatdoubleSum(-m1, m2)
        End Operator

        Public Shared Operator *(m2 As Matrix, m1#) As Matrix
            Return MatdoubleMult(m2, m1)
        End Operator

        Public Shared Operator *(m1 As Matrix, m2 As Matrix) As Matrix
            If m1.x = m2.x AndAlso m1.y = m2.y Then Return DeltaMult(m1, m2)
            Return MatMult(m1, m2)
        End Operator

        Public Shared Function MatdoubleMult(m2 As Matrix, m1#) As Matrix

            Dim a#(,) = m2
            Dim b = New Double(m2.x - 1, m2.y - 1) {}
            MatrixLoop((Sub(i, j) b(i, j) = a(i, j) * m1), b.GetLength(0), b.GetLength(1))
            Return b

        End Function

        Public Shared Function MatMult(m1 As Matrix, m2 As Matrix) As Matrix

            If m1.y <> m2.x Then _
                Throw New ArgumentException("Matrix must have compatible dimensions")

            Dim n = m1.x
            Dim m = m1.y
            Dim p = m2.y
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

            Return c

        End Function

        Public Shared Function DeltaMult(m1 As Matrix, m2 As Matrix) As Matrix

            If m1.x <> m2.x OrElse m1.y <> m2.y Then _
                Throw New ArgumentException("Matrix must have the same dimensions")

            Dim output = New Double(m1.x - 1, m2.y - 1) {}
            MatrixLoop((Sub(i, j) output(i, j) = m1.matrix(i, j) * m2.matrix(i, j)), m1.x, m2.y)
            Return output

        End Function

        Public Shared Operator /(m2 As Matrix, m1#) As Matrix
            Return MatdoubleMult(m2, 1 / m1)
        End Operator

        Public Shared Operator ^(m2 As Matrix, m1#) As Matrix
            Return Pow(m2, m1)
        End Operator

        Public Shared Function Pow(m2 As Matrix, m1#) As Matrix

            Dim output = New Double(m2.x - 1, m2.y - 1) {}
            MatrixLoop((Sub(i, j) output(i, j) = Math.Pow(m2.matrix(i, j), m1)), m2.x, m2.y)
            Return output

        End Function

        Public Function Pow(m1#) As Matrix
            Return Pow(Me, m1)
        End Function

        Public Shared Function Sumatory(m As Matrix, Optional dimension% = -1) As Matrix

            Dim output#(,)

            If dimension = -1 Then
                output = New Double(0, 0) {}
            ElseIf dimension = 0 Then
                output = New Double(m.x - 1, 0) {}
            ElseIf dimension = 1 Then
                output = New Double(0, m.y - 1) {}
            Else
                Throw New ArgumentException("The dimension must be -1, 0 or 1")
            End If

            If dimension = -1 Then
                MatrixLoop((Sub(i, j) output(0, 0) += m.matrix(i, j)), m.x, m.y)
            ElseIf dimension = 0 Then
                MatrixLoop((Sub(i, j) output(i, 0) += m.matrix(i, j)), m.x, m.y)
            ElseIf dimension = 1 Then
                MatrixLoop((Sub(i, j) output(0, j) += m.matrix(i, j)), m.x, m.y)
            End If

            Return output

        End Function

        Public Function Sumatory(Optional dimension% = -1) As Matrix
            Return Sumatory(Me, dimension)
        End Function

        Public Function Dot(m2 As Matrix) As Matrix
            Return Dot(Me, m2)
        End Function

        Public Shared Function Dot(m1 As Matrix, m2 As Matrix) As Matrix
            Return m1 * m2.T
        End Function

        Public Function Abs_(m As Matrix) As Matrix

            Dim d#(,) = m
            MatrixLoop((Sub(i, j) d(i, j) = Math.Abs(m.matrix(i, j))), m.x, m.y)
            Return d

        End Function

        Public Function Average_#(m As Matrix)

            Dim d# = 0
            MatrixLoop((Sub(i, j) d += m.matrix(i, j)), m.x, m.y)
            Dim aver# = d / (m.x * m.y)
            Return aver

        End Function

        Public Shared Sub MatrixLoop(e As Action(Of Integer, Integer), x%, y%)

            For i = 0 To x - 1
                For j = 0 To y - 1
                    e(i, j)
                Next
            Next

        End Sub

        ''' <summary>
        ''' Apply a function to each element of the array
        ''' </summary>
        Public Shared Function Map(m As Matrix, lambdaFct As Func(Of Double, Double)) As Matrix

            Dim c As New Matrix(m.x, m.y)

            For i = 0 To m.x - 1
                For j = 0 To m.y - 1
                    c._matrix(i, j) = lambdaFct.Invoke(m._matrix(i, j))
                Next
            Next

            Return c

        End Function

        ''' <summary>
        ''' Convert whole Matrix object to array
        ''' </summary>
        Public Function ToArray() As Double()

            Dim array#() = New Double(Me._matrix.Length - 1) {}

            Dim k = 0
            For i = 0 To Me.x - 1
                For j = 0 To Me.y - 1
                    array(k) = Me._matrix(i, j)
                    k += 1
                Next
            Next

            Return array

        End Function

        ''' <summary>
        ''' Convert whole Matrix object to array
        ''' </summary>
        Public Function ToArraySingle() As Single()

            Dim array!() = New Single(Me._matrix.Length - 1) {}

            Dim k = 0
            For i = 0 To Me.x - 1
                For j = 0 To Me.y - 1
                    array(k) = CSng(Me._matrix(i, j))
                    k += 1
                Next
            Next

            Return array

        End Function

    End Structure

End Namespace
