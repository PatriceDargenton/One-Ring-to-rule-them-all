
' Matrix implementation using Math.Net
' https://numerics.mathdotnet.com/Matrix.html
' <package id="MathNet.Numerics" version="4.12.0" targetFramework="net472" />

Imports System.Text ' StringBuilder

Namespace Utility

#Const Implementation = 0 ' 0 : Off, 1 : On

#If Implementation Then

    Public Class Matrix : Implements ICloneable

        Private Function IClone() As Object Implements ICloneable.Clone
            Dim m As Matrix = New Matrix(Me)
            Return m
        End Function

        Public Function Clone() As Matrix
            Dim m As Matrix = DirectCast(Me.IClone(), Matrix)
            Return m
        End Function

        Private m_matrix As MathNet.Numerics.LinearAlgebra.Matrix(Of Double)

#Region "Properties"

        Default Public Property Item#(r%, c%)
            Get
                Return Me.m_matrix(r, c)
            End Get
            Set(value#)
                Me.m_matrix(r, c) = value
            End Set
        End Property

        Public Property matrixP As Double(,)
            Get
                Return Me.m_matrix.ToArray
            End Get
            Set(doubleArray As Double(,))

                Dim rows = doubleArray.GetLength(0)
                Dim columns = doubleArray.GetLength(1)
                Constructor(rows, columns)
                For i = 0 To rows - 1
                    For j = 0 To columns - 1
                        Me.m_matrix(i, j) = doubleArray(i, j)
                    Next
                Next

            End Set
        End Property

        Public ReadOnly Property isDefined As Boolean
            Get
                Return Not IsNothing(Me.m_matrix)
            End Get
        End Property

        ''' <summary>
        ''' Rows
        ''' </summary>
        Public ReadOnly Property r%
            Get
                Return Me.m_matrix.RowCount
            End Get
        End Property

        ''' <summary>
        ''' Columns
        ''' </summary>
        Public ReadOnly Property c%
            Get
                Return Me.m_matrix.ColumnCount
            End Get
        End Property

        ''' <summary>
        ''' Transpose
        ''' </summary>
        Public ReadOnly Property T As Matrix
            Get
                Return Transpose_(Me)
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
            Constructor(rows, columns)
        End Sub

        Public Sub New(doubleArray#(,))
            Dim rows = doubleArray.GetLength(0)
            Dim columns = doubleArray.GetLength(1)
            Constructor(rows, columns)
            For i = 0 To rows - 1
                For j = 0 To columns - 1
                    Me.m_matrix(i, j) = doubleArray(i, j)
                Next
            Next
        End Sub

        Public Sub New(matrix0!(,))
            Dim rows = matrix0.GetLength(0)
            Dim columns = matrix0.GetLength(1)
            Constructor(rows, columns)
            For i = 0 To rows - 1
                For j = 0 To columns - 1
                    Me.m_matrix(i, j) = matrix0(i, j)
                Next
            Next
        End Sub

        Public Sub New(m As MathNet.Numerics.LinearAlgebra.Matrix(Of Double))
            Me.m_matrix = m
        End Sub

        Private Sub Constructor(rows%, columns%)
            Me.m_matrix = MathNet.Numerics.LinearAlgebra.Matrix(Of Double).Build.DenseIdentity(rows, columns)
        End Sub

        ''' <summary>
        ''' Create a matrix object from an array of Single
        ''' </summary>
        Public Shared Function FromArraySingle(inputs!()) As Matrix
            Dim m As New Matrix(inputs.Length, 1)
            For i = 0 To inputs.Length - 1
                m.m_matrix(i, 0) = inputs(i)
            Next
            Return m
        End Function

#End Region

#Region "Operators"

        ' Implicit conversion operator #(,) -> Matrix
        Public Shared Widening Operator CType(doubleArray#(,)) As Matrix
            Return New Matrix(doubleArray)
        End Operator

        ' Implicit conversion operator Matrix -> #(,)
        Public Shared Widening Operator CType(matrix0 As Matrix) As Double(,)
            Return matrix0.matrixP
        End Operator

        ' Implicit conversion operator !(,) -> Matrix
        Public Shared Widening Operator CType(singleArray!(,)) As Matrix
            Return New Matrix(singleArray)
        End Operator

        Public Shared Operator +(m1 As Matrix, m2 As Matrix) As Matrix

            Dim m1plusm2 As Matrix = m1.Clone()
            m1plusm2.m_matrix += m2.m_matrix
            Return m1plusm2

        End Operator

        Public Shared Operator -(m1 As Matrix, m2 As Matrix) As Matrix

            Dim m1minusm2 As Matrix = m1.Clone()
            m1minusm2.m_matrix -= m2.m_matrix
            Return m1minusm2

        End Operator

        Public Shared Operator -(m2 As Matrix, m1#) As Matrix

            Dim m As Matrix = m2.Clone()
            m.m_matrix = m.m_matrix.Add(-m1)
            Return m

        End Operator

        Public Shared Operator *(m2 As Matrix, m1#) As Matrix

            Dim m As Matrix = m2.Clone()
            m.m_matrix *= m1
            Return m

        End Operator

        Public Shared Operator *(m1 As Matrix, m2 As Matrix) As Matrix

            If m1.r = m2.r AndAlso m1.c = m2.c Then

                Dim m1multm2 As Matrix = m1.Clone()
                m1multm2.Multiply(m2)
                Return m1multm2

            End If

            Return New Matrix(m1.m_matrix * m2.m_matrix)

        End Operator

#End Region

#Region "Public shared operations"

        Public Shared Function Zeros(r%, c%) As Matrix

            Dim MNMatrix = MathNet.Numerics.LinearAlgebra.Matrix(Of Double).Build.DenseIdentity(r, c)
            MNMatrix.Clear()
            Dim result As Matrix = New Matrix(MNMatrix)
            Return result

        End Function

        Public Shared Function Ones(r%, c%) As Matrix

            Dim m = Zeros(r, c)
            m.m_matrix += 1
            Return m

        End Function

        ''' <summary>
        ''' Transpose a matrix
        ''' </summary>
        Private Shared Function Transpose_(m As Matrix) As Matrix
            Return New Matrix(m.m_matrix.Transpose())
        End Function

        ''' <summary>
        ''' Transpose and multiply this transposed matrix by m
        ''' </summary>
        Public Shared Function TransposeAndMultiply1(original As Matrix, m As Matrix) As Matrix
            Return New Matrix(original.T.m_matrix * m.m_matrix)
        End Function

        ''' <summary>
        ''' Transpose and multiply a matrix m by this transposed one
        ''' </summary>
        Public Shared Function TransposeAndMultiply2(original As Matrix, m As Matrix) As Matrix
            Return New Matrix(m.m_matrix * original.T.m_matrix)
        End Function

        ''' <summary>
        ''' Subtract 2 matrices (the first as an array of Single) and return a new matrix
        ''' </summary>
        Public Shared Function SubtractFromArraySingle(a_array!(), b As Matrix) As Matrix
            Dim a As Matrix = FromArraySingle(a_array)
            Return New Matrix(a.m_matrix - b.m_matrix)
        End Function

        ''' <summary>
        ''' Multiply matrices a and b, add matrix c,
        '''  and apply a function to every element of the result
        ''' </summary>
        Public Shared Function MultiplyAddAndMap(
            a As Matrix, b As Matrix, c As Matrix,
            lambdaFct As Func(Of Double, Double)) As Matrix

            Dim d As New Matrix(a.m_matrix * b.m_matrix)
            d.m_matrix += c.m_matrix
            d.Map(lambdaFct)
            Return d

        End Function

        ''' <summary>
        ''' Multiply matrices a and b, and apply a function to every element of the result
        ''' </summary>
        Public Shared Function MultiplyAndMap(a As Matrix, b As Matrix,
            lambdaFct As Func(Of Double, Double)) As Matrix

            Dim d As New Matrix(a.m_matrix * b.m_matrix)
            d.Map(lambdaFct)
            Return d

        End Function

        ''' <summary>
        ''' Apply a function to each element of the array
        ''' </summary>
        Public Shared Function Map(m As Matrix, lambdaFct As Func(Of Double, Double)) As Matrix

            Dim c As New Matrix(m.r, m.c)
            For i = 0 To m.r - 1
                For j = 0 To m.c - 1
                    c.m_matrix(i, j) = lambdaFct.Invoke(m.m_matrix(i, j))
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

            If dimension = AxisZero.none Then
                MatrixLoop((Sub(i, j) output(0, 0) += m.m_matrix(i, j)), m.r, m.c)
            ElseIf dimension = AxisZero.horizontal Then
                MatrixLoop((Sub(i, j) output(i, 0) += m.m_matrix(i, j)), m.r, m.c)
            ElseIf dimension = AxisZero.vertical Then
                MatrixLoop((Sub(i, j) output(0, j) += m.m_matrix(i, j)), m.r, m.c)
            End If

            Dim result As Matrix = output
            Return result

        End Function

        Public Shared Sub MatrixLoop(e As Action(Of Integer, Integer), r%, c%)

            For i = 0 To r - 1
                For j = 0 To c - 1
                    e(i, j)
                Next
            Next

        End Sub

#End Region

#Region "Public operations"

        Public Function AddColumn(m2 As Matrix) As Matrix

            If m2.c <> 1 OrElse m2.r <> Me.r Then Throw New ArgumentException("Invalid dimensions")

            Dim newMatrix = New Double(Me.r - 1, Me.c + 1 - 1) {}
            For i = 0 To Me.r - 1
                newMatrix(i, 0) = m2.m_matrix(i, 0)
            Next
            MatrixLoop((Sub(i, j) newMatrix(i, j + 1) = Me.m_matrix(i, j)), r, c)

            Dim result As Matrix = newMatrix
            Return result

        End Function

        ''' <summary>
        ''' Apply a function to every element of the array
        ''' </summary>
        Public Sub Map(lambdaFct As Func(Of Double, Double))

            For i = 0 To Me.r - 1
                For j = 0 To Me.c - 1
                    Me.m_matrix(i, j) = lambdaFct.Invoke(Me.m_matrix(i, j))
                Next
            Next

        End Sub

        Public Function Sumatory(Optional dimension As AxisZero = AxisZero.none) As Matrix
            Return Sumatory(Me, dimension)
        End Function

        ''' <summary>
        ''' Cut matrix from r1, c1 to r2, c2
        ''' </summary>
        Public Function Slice(r1%, c1%, r2%, c2%) As Matrix

            If r1 >= r2 OrElse c1 >= c2 OrElse r1 < 0 OrElse
               r2 < 0 OrElse c1 < 0 OrElse c2 < 0 Then _
               Throw New ArgumentException("Dimensions are not valid")

            Dim slice0 = New Double(r2 - r1 - 1, c2 - c1 - 1) {}
            For i = r1 To r2 - 1
                For j = c1 To c2 - 1
                    slice0(i - r1, j - c1) = Me.m_matrix(i, j)
                Next
            Next

            Dim m As Matrix = slice0
            Return m

        End Function

#End Region

#Region "Private operations"

        ''' <summary>
        ''' Hadamard product (element-wise multiplication):
        ''' Multiply each element of the array with each element of the given array
        ''' </summary>
        Private Overloads Sub Multiply(m As Matrix)

            For i = 0 To Me.r - 1
                For j = 0 To Me.c - 1
                    Me.m_matrix(i, j) *= m.m_matrix(i, j)
                Next
            Next

        End Sub

        ''' <summary>
        ''' Compute absolute values of a matrix
        ''' </summary>
        Private Function Abs_() As Matrix

            Dim c As New Matrix(Me.r, Me.c)
            For i = 0 To Me.r - 1
                For j = 0 To Me.c - 1
                    c.m_matrix(i, j) = Math.Abs(Me.m_matrix(i, j))
                Next
            Next
            Return c

        End Function

        ''' <summary>
        ''' Compute average value of the matrix
        ''' </summary>
        Private Function Average_#()

            Dim nbElements% = Me.r * Me.c
            Dim sum# = 0
            For i = 0 To Me.r - 1
                For j = 0 To Me.c - 1
                    sum += Me.m_matrix(i, j)
                Next
            Next

            Dim average1# = 0
            If nbElements <= 1 Then
                average1 = sum
            Else
                average1 = sum / nbElements
            End If

            Return average1

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
                    Dim strVal$ = Me.m_matrix(i, j).ToString(dec).ReplaceCommaByDot()
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
        ''' Convert whole Matrix object to an array of Single
        ''' </summary>
        Public Function ToArraySingle() As Single()

            Dim length = Me.m_matrix.RowCount * Me.m_matrix.ColumnCount
            Dim array!() = New Single(length - 1) {}
            Dim k = 0
            For i = 0 To Me.r - 1
                For j = 0 To Me.c - 1
                    array(k) = CSng(Me.m_matrix(i, j))
                    k += 1
                Next
            Next
            Return array

        End Function

        ''' <summary>
        ''' Convert whole Matrix object to an array of Double
        ''' </summary>
        Public Function ToArray() As Double()

            Dim length = Me.m_matrix.RowCount * Me.m_matrix.ColumnCount
            Dim array#() = New Double(length - 1) {}
            Dim k = 0
            For i = 0 To Me.r - 1
                For j = 0 To Me.c - 1
                    array(k) = Me.m_matrix(i, j)
                    k += 1
                Next
            Next
            Return array

        End Function

#End Region

#Region "Miscellaneous"

        Public Function GetValue#(r%, c%)
            Return Me.m_matrix(r, c)
        End Function

        Public Function GetRow(r%) As Matrix

            Dim row = New Double(0, Me.c - 1) {}
            For j = 0 To Me.c - 1
                row(0, j) = Me.m_matrix(r, j)
            Next
            Return row

        End Function

        Public Function GetColumn(c%) As Matrix

            Dim column = New Double(Me.r - 1, 0) {}
            For i = 0 To Me.r - 1
                column(i, 0) = Me.m_matrix(i, c)
            Next
            Return column

        End Function

        Public Shared Function Randomize(r%, c%, rnd As Random,
            Optional minValue! = -0.5!, Optional maxValue! = 0.5!) As Matrix

            Dim random_ = New Double(r - 1, c - 1) {}
            'MatrixLoop((Sub(i, j) random_(i, j) = rnd.NextDouble), x, y)
            MatrixLoop((Sub(i, j) random_(i, j) =
                Math.Round(rnd.NextDouble(minValue, maxValue),
                clsMLPGeneric.nbRoundingDigits)), r, c)
            Dim m As Matrix = random_
            Return m

        End Function

        Public Sub Randomize(rnd As Random,
            Optional minValue! = -0.5!, Optional maxValue! = 0.5!)

            MatrixLoop((Sub(i, j) Me.m_matrix(i, j) =
                Math.Round(rnd.NextDouble(minValue, maxValue),
                clsMLPGeneric.nbRoundingDigits)), Me.r, Me.c)

        End Sub

        ''' <summary>
        ''' Set 1 or 0 for each value of the matrix whether it is inferior
        '''  to the threshold, and return a new matrix
        ''' </summary>
        Public Function Threshold(minThreshold!) As Matrix

            Dim c As New Matrix(Me.r, Me.c)

            For i = 0 To c.r - 1
                For j = 0 To c.c - 1
                    c.m_matrix(i, j) = CDbl(IIf(Math.Abs(Me.m_matrix(i, j)) <= minThreshold, 1.0#, 0.0#))
                Next
            Next

            Return c

        End Function

#End Region

    End Class

#End If

End Namespace