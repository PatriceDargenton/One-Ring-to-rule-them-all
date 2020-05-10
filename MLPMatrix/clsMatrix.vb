
' From: https://github.com/nlabiris/perceptrons : C# -> VB .NET conversion

Imports System.Text ' StringBuilder
Imports System.Threading.Tasks ' Parallel.For (for previous Visual Studio)

Namespace MatrixMLP

    ''' <summary>
    ''' Contains matrix operations
    ''' </summary>
    Class Matrix

        ''' <summary>
        ''' Rows
        ''' </summary>
        Private m_rows% ' ReadOnly

        ''' <summary>
        ''' Columns
        ''' </summary>
        Private m_cols% ' ReadOnly 

        ''' <summary>
        ''' Array data
        ''' </summary>
        Private data#(,)

        ''' <summary>
        ''' Random number generator
        ''' </summary>
        Private Shared rng As New Random

        ''' <summary>
        ''' Rows
        ''' </summary>
        Public ReadOnly Property Rows%
            Get
                Return Me.m_rows
            End Get
        End Property

        ''' <summary>
        ''' Columns
        ''' </summary>
        Public ReadOnly Property Cols%
            Get
                Return Me.m_cols
            End Get
        End Property

        ''' <summary>
        ''' Constructor
        ''' </summary>
        Public Sub New(rows%, cols%)

            Me.m_rows = rows
            Me.m_cols = cols

            Me.data = New Double(rows - 1, cols - 1) {}

        End Sub

        ''' <summary>
        ''' Create a matrix object from an array
        ''' </summary>
        Public Shared Function FromArray(inputs#()) As Matrix

            Dim m As New Matrix(inputs.Length, 1)

            For i As Integer = 0 To inputs.Length - 1
                m.data(i, 0) = inputs(i)
            Next

            Return m

        End Function

        ''' <summary>
        ''' Create a matrix object from an array of Single
        ''' </summary>
        Public Shared Function FromArraySingle(inputs!()) As Matrix

            Dim m As New Matrix(inputs.Length, 1)

            For i As Integer = 0 To inputs.Length - 1
                m.data(i, 0) = inputs(i)
            Next

            Return m

        End Function

        Public Sub New(matrix#(,))
            Me.data = matrix
            Me.m_rows = Me.data.GetLength(0)
            Me.m_cols = Me.data.GetLength(1)
        End Sub

        Public Sub New(matrix!(,))
            Me.m_rows = matrix.GetLength(0)
            Me.m_cols = matrix.GetLength(1)
            ReDim Me.data(0 To Me.m_rows - 1, 0 To Me.m_cols - 1)
            For i As Integer = 0 To Me.m_rows - 1
                For j As Integer = 0 To Me.m_cols - 1
                    Me.data(i, j) = matrix(i, j)
                Next
            Next
        End Sub

        ' Implicit conversion operator #(,) -> Matrix
        Public Shared Widening Operator CType(matrix(,) As Double) As Matrix
            Return New Matrix(matrix)
        End Operator

        ' Implicit conversion operator Matrix -> #(,)
        Public Shared Widening Operator CType(matrix As Matrix) As Double(,)
            Return matrix.data
        End Operator

        ' Implicit conversion operator !(,) -> Matrix
        Public Shared Widening Operator CType(matrix(,) As Single) As Matrix
            Return New Matrix(matrix)
        End Operator

        ''' <summary>
        ''' Convert the first vector of the matrix to array
        ''' </summary>
        Public Function ToVectorArray() As Double()

            Dim array#() = New Double(Me.data.GetLength(0) - 1) {}

            For i As Integer = 0 To array.Length - 1
                array(i) = Me.data(i, 0)
            Next

            Return array

        End Function

        ''' <summary>
        ''' Convert the first vector of the matrix to array of Single
        ''' </summary>
        Public Function ToVectorArraySingle() As Single()

            Dim array!() = New Single(Me.data.GetLength(0) - 1) {}

            For i As Integer = 0 To array.Length - 1
                array(i) = CSng(Me.data(i, 0))
            Next

            Return array

        End Function

        ''' <summary>
        ''' Fill matrix with random data
        ''' </summary>
        Public Sub Randomize(minValue#, maxValue#)

            Parallel.For(0, Me.Rows,
                Sub(i)
                    Parallel.For(0, Me.Cols,
                        Sub(j)
                            Dim r# = rng.NextDouble(minValue, maxValue)
                            Dim rounded# = Math.Round(r, clsMLPGeneric.roundWeights)
                            Me.data(i, j) = rounded
                        End Sub)
                End Sub)

        End Sub

        ''' <summary>
        ''' Override <c>ToString()</c> method to pretty-print the matrix
        ''' </summary>
        Public Overrides Function ToString$()
            Return ToStringWithFormat()
        End Function

        Public Function ToStringWithFormat$(Optional dec$ = format2Dec)

            Dim sb As New StringBuilder
            sb.AppendLine("{")
            For i As Integer = 0 To Me.m_rows - 1
                sb.Append(" {")
                For j As Integer = 0 To Me.m_cols - 1
                    Dim strVal$ = Me.data(i, j).ToString(dec).ReplaceCommaByDot()
                    sb.Append(strVal)
                    If j < Me.m_cols - 1 Then sb.Append(", ")
                Next
                sb.Append("}")
                If i < Me.m_rows - 1 Then sb.Append("," & vbLf)
            Next
            sb.Append("}")

            Dim s$ = sb.ToString
            Return s

        End Function

        Public Shared Operator +(m1 As Matrix, m2 As Matrix) As Matrix
            Dim m1plusm2 As Matrix = m1
            m1plusm2.Add(m2)
            Return m1plusm2
        End Operator

        Public Shared Operator -(m1 As Matrix, m2 As Matrix) As Matrix
            Dim m1minusm2 As Matrix = m1
            m1minusm2.Subtract(m2)
            Return m1minusm2
        End Operator

        Public Shared Operator *(m1 As Matrix, m2 As Matrix) As Matrix
            Dim m1xm2 As Matrix = m1
            m1xm2.Multiply(m2)
            Return m1xm2
        End Operator

        ''' <summary>
        ''' Add a number to each element of the array
        ''' </summary>
        Public Overloads Sub Add(n%)

            For i As Integer = 0 To Me.m_rows - 1
                For j As Integer = 0 To Me.m_cols - 1
                    Me.data(i, j) += n
                Next
            Next

        End Sub

        ''' <summary>
        ''' Add each element of the matrices
        ''' </summary>
        Public Overloads Sub Add(m As Matrix)

            For i As Integer = 0 To Me.m_rows - 1
                For j As Integer = 0 To Me.m_cols - 1
                    Me.data(i, j) += m.data(i, j)
                Next
            Next

        End Sub

        ''' <summary>
        ''' Subtract a number to each element of the array
        ''' </summary>
        Public Overloads Sub Subtract(n%)

            For i As Integer = 0 To Me.m_rows - 1
                For j As Integer = 0 To Me.m_cols - 1
                    Me.data(i, j) -= n
                Next
            Next

        End Sub

        ''' <summary>
        ''' Subtract each element of the matrices
        ''' </summary>
        Public Overloads Sub Subtract(m As Matrix)

            For i As Integer = 0 To Me.m_rows - 1
                For j As Integer = 0 To Me.m_cols - 1
                    Me.data(i, j) -= m.data(i, j)
                Next
            Next

        End Sub

        ''' <summary>
        ''' Subtract 2 matrices and return a new matrix
        ''' </summary>
        Public Overloads Shared Function Subtract(a As Matrix, b As Matrix) As Matrix

            Dim c As New Matrix(a.Rows, a.Cols)

            For i As Integer = 0 To c.Rows - 1
                For j As Integer = 0 To c.Cols - 1
                    c.data(i, j) = a.data(i, j) - b.data(i, j)
                Next
            Next

            Return c

        End Function

        ''' <summary>
        ''' Subtract 2 matrices (the first as an array of Single) and return a new matrix
        ''' </summary>
        Public Overloads Shared Function SubtractFromArraySingle(a_array!(), b As Matrix) As Matrix
            Dim a As Matrix = Matrix.FromArraySingle(a_array)
            Dim c As Matrix = Matrix.Subtract(a, b)
            Return c
        End Function

        ''' <summary>
        ''' Scalar product: Multiply each element of the array with the given number
        ''' </summary>
        Public Overloads Sub Multiply(n#)

            For i As Integer = 0 To Me.m_rows - 1
                For j As Integer = 0 To Me.m_cols - 1
                    Me.data(i, j) *= n
                Next
            Next

        End Sub

        ''' <summary>
        ''' Hadamard product (element-wise multiplication):
        ''' Multiply each element of the array with each element of the given array
        ''' </summary>
        Public Overloads Sub Multiply(m As Matrix)

            For i As Integer = 0 To Me.m_rows - 1
                For j As Integer = 0 To Me.m_cols - 1
                    Me.data(i, j) *= m.data(i, j)
                Next
            Next

        End Sub

        ''' <summary>
        ''' Matrix product
        ''' </summary>
        Public Overloads Shared Function Multiply(a As Matrix, b As Matrix) As Matrix

            If a.Cols <> b.Rows Then
                Throw New Exception("Columns of A must match columns of B")
            End If

            Dim c As New Matrix(a.Rows, b.Cols)

            For i As Integer = 0 To c.Rows - 1
                For j As Integer = 0 To c.Cols - 1
                    Dim sum# = 0
                    For k As Integer = 0 To a.Cols - 1
                        sum += a.data(i, k) * b.data(k, j)
                    Next
                    c.data(i, j) = sum
                Next
            Next

            Return c

        End Function

        ''' <summary>
        ''' Matrix product (the first as an array)
        ''' </summary>
        Public Overloads Shared Function MultiplyFromArray(a_array#(), b As Matrix) As Matrix
            Dim a As Matrix = Matrix.FromArray(a_array)
            Dim c As Matrix = Matrix.Multiply(b, a)
            Return c
        End Function

        ''' <summary>
        ''' Multiply matrices a and b, add matrix c,
        '''  and apply a function to every element of the result
        ''' </summary>
        Public Overloads Shared Function MultiplyAddAndMap(
            a As Matrix, b As Matrix, c As Matrix,
            lambdaFct As Func(Of Double, Double)) As Matrix

            Dim d As Matrix = Matrix.Multiply(a, b)
            d.Add(c)
            d.Map(lambdaFct)

            Return d

        End Function

        ''' <summary>
        ''' Multiply matrices a and b, and apply a function to every element of the result
        ''' </summary>
        Public Overloads Shared Function MultiplyAndMap(
            a As Matrix, b As Matrix,
            lambdaFct As Func(Of Double, Double)) As Matrix

            Dim d As Matrix = Matrix.Multiply(a, b)
            d.Map(lambdaFct)

            Return d

        End Function

        ''' <summary>
        ''' Compute absolute values of a matrix
        ''' </summary>
        Public Overloads Function Abs() As Matrix

            Dim c As New Matrix(Me.m_rows, Me.m_cols)

            For i As Integer = 0 To Me.m_rows - 1
                For j As Integer = 0 To Me.m_cols - 1
                    c.data(i, j) = Math.Abs(Me.data(i, j))
                Next
            Next

            Return c

        End Function

        ''' <summary>
        ''' Compute average value of the matrix
        ''' </summary>
        Public Overloads Function Average#()

            Dim nbElements% = Me.m_rows * Me.m_cols
            Dim sum# = 0
            For i As Integer = 0 To Me.m_rows - 1
                For j As Integer = 0 To Me.m_cols - 1
                    sum += Me.data(i, j)
                Next
            Next

            Dim average_# = 0
            If nbElements <= 1 Then
                average_ = sum
            Else
                average_ = sum / nbElements
            End If

            Return average_

        End Function

        ''' <summary>
        ''' Transpose a matrix
        ''' </summary>
        Public Overloads Shared Function Transpose(m As Matrix) As Matrix

            Dim c As New Matrix(m.Cols, m.Rows)

            For i As Integer = 0 To m.Rows - 1
                For j As Integer = 0 To m.Cols - 1
                    c.data(j, i) = m.data(i, j)
                Next
            Next

            Return c

        End Function

        ''' <summary>
        ''' Transpose the matrix
        ''' </summary>
        Public Overloads Sub Transpose()

            For i As Integer = 0 To Me.Rows - 1
                For j As Integer = 0 To Me.Cols - 1
                    Me.data(j, i) = Me.data(i, j)
                Next
            Next

        End Sub

        ''' <summary>
        ''' Transpose and multiply this transposed matrix by m
        ''' </summary>
        Public Overloads Shared Function TransposeAndMultiply1(
            original As Matrix, m As Matrix) As Matrix
            Dim original_t As Matrix = Matrix.Transpose(original)
            Dim result As Matrix = Matrix.Multiply(original_t, m)
            Return result
        End Function

        ''' <summary>
        ''' Transpose and multiply a matrix m by this transposed one
        ''' </summary>
        Public Overloads Shared Function TransposeAndMultiply2(
            original As Matrix, m As Matrix) As Matrix
            Dim original_t As Matrix = Matrix.Transpose(original)
            Dim result As Matrix = Matrix.Multiply(m, original_t)
            Return result
        End Function

        ''' <summary>
        ''' Apply a function to every element of the array
        ''' </summary>
        Public Sub Map(lambdaFct As Func(Of Double, Double))

            For i As Integer = 0 To Me.Rows - 1
                For j As Integer = 0 To Me.Cols - 1
                    Me.data(i, j) = lambdaFct.Invoke(Me.data(i, j))
                Next
            Next

        End Sub

        ''' <summary>
        ''' Apply a function to every element of the array
        ''' </summary>
        Public Shared Function Map(m As Matrix, lambdaFct As Func(Of Double, Double)) As Matrix

            Dim c As New Matrix(m.Rows, m.Cols)

            For i As Integer = 0 To m.Rows - 1
                For j As Integer = 0 To m.Cols - 1
                    c.data(i, j) = lambdaFct.Invoke(m.data(i, j))
                Next
            Next

            Return c

        End Function

    End Class

End Namespace