
' From: https://github.com/nlabiris/perceptrons : C# -> VB .NET conversion

Imports System.Text ' StringBuilder
Imports System.Threading.Tasks ' Parallel.For

Namespace Utility

#Const Implementation = 1 ' 0 : Off, 1 : On

#If Implementation Then

    Public Class Matrix : Implements ICloneable

        Const parallelLoop As Boolean = True
        Const parallelMinSize = 64

        Shared rowMax%, columnMax%

        Private Function IClone() As Object Implements ICloneable.Clone
            Dim m As Matrix = New Matrix(Me)
            Return m
        End Function

        Public Function Clone() As Matrix
            Dim m As Matrix = DirectCast(Me.IClone(), Matrix)
            Return m
        End Function

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
            Me.data = New Double(rows - 1, columns - 1) {}
        End Sub

        Public Sub New(doubleArray#(,))
            Me.data = doubleArray
        End Sub

        Public Sub New(singleArray!(,))
            Dim rows = singleArray.GetLength(0)
            Dim columns = singleArray.GetLength(1)
            ReDim Me.data(rows - 1, columns - 1)
            For i = 0 To rows - 1
                For j = 0 To columns - 1
                    Me.data(i, j) = singleArray(i, j)
                Next
            Next
        End Sub

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
        Public Shared Widening Operator CType(doubleArray#(,)) As Matrix
            Return New Matrix(doubleArray)
        End Operator

        ' Implicit conversion operator Matrix -> #(,)
        Public Shared Widening Operator CType(matrix0 As Matrix) As Double(,)
            'Return matrix0.data
            Return matrix0.matrixP
        End Operator

        ' Implicit conversion operator !(,) -> Matrix
        Public Shared Widening Operator CType(singleArray!(,)) As Matrix
            Return New Matrix(singleArray)
        End Operator

        Public Shared Operator +(m1 As Matrix, m2 As Matrix) As Matrix

            Dim m1plusm2 As Matrix = m1.Clone()
            m1plusm2.Add(m2)
            Return m1plusm2

        End Operator

        Public Shared Operator -(m1 As Matrix, m2 As Matrix) As Matrix

            Dim m1minusm2 As Matrix = m1.Clone()
            m1minusm2.Subtract(m2)
            Return m1minusm2

        End Operator

        Public Shared Operator -(m2 As Matrix, m1#) As Matrix

            Dim m As Matrix = m2.Clone()
            m.Subtract(m1)
            Return m

        End Operator

        Public Shared Operator *(m2 As Matrix, m1#) As Matrix

            Dim m As Matrix = m2.Clone()
            m.Multiply(m1)
            Return m

        End Operator

        Public Shared Operator *(m1 As Matrix, m2 As Matrix) As Matrix

            If m1.r = m2.r AndAlso m1.c = m2.c Then

                Dim m1multm2 As Matrix = m1.Clone()
                m1multm2.Multiply(m2)
                Return m1multm2

            End If

            Dim m As Matrix = Multiply(m1, m2)
            Return m

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

        ''' <summary>
        ''' Transpose a matrix
        ''' </summary>
        Private Shared Function Transpose_(m As Matrix) As Matrix

            Dim c As New Matrix(m.c, m.r)
            For i = 0 To m.r - 1
                For j = 0 To m.c - 1
                    c.data(j, i) = m.data(i, j)
                Next
            Next
            Return c

        End Function

        ''' <summary>
        ''' Transpose and multiply this transposed matrix by m
        ''' </summary>
        Public Shared Function TransposeAndMultiply1(original As Matrix, m As Matrix) As Matrix
            'Dim original_t As Matrix = Transpose(original)
            Dim result As Matrix = Multiply(original.T, m)
            Return result
        End Function

        ''' <summary>
        ''' Transpose and multiply a matrix m by this transposed one
        ''' </summary>
        Public Shared Function TransposeAndMultiply2(
            original As Matrix, m As Matrix) As Matrix
            'Dim original_t As Matrix = Transpose(original)
            Dim result As Matrix = Multiply(m, original.T)
            Return result
        End Function

        ''' <summary>
        ''' Subtract 2 matrices (the first as an array of Single) and return a new matrix
        ''' </summary>
        Public Shared Function SubtractFromArraySingle(a_array!(), b As Matrix) As Matrix
            Dim a As Matrix = FromArraySingle(a_array)
            Dim result As Matrix = Subtract(a, b)
            Return result
        End Function

        ''' <summary>
        ''' Multiply matrices a and b, add matrix c,
        '''  and apply a function to every element of the result
        ''' </summary>
        Public Shared Function MultiplyAddAndMap(
            a As Matrix, b As Matrix, c As Matrix,
            lambdaFct As Func(Of Double, Double)) As Matrix

            Dim d As Matrix = Multiply(a, b)
            d.Add(c)
            d.Map(lambdaFct)
            Return d

        End Function

        ''' <summary>
        ''' Multiply matrices a and b, and apply a function to every element of the result
        ''' </summary>
        Public Shared Function MultiplyAndMap(a As Matrix, b As Matrix,
            lambdaFct As Func(Of Double, Double)) As Matrix

            Dim d As Matrix = Multiply(a, b)
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

        Public Shared Sub MatrixLoop(e As Action(Of Integer, Integer), r%, c%)

            If r > rowMax Then rowMax = r : Debug.WriteLine("rowMax=" & rowMax)
            If c > columnMax Then columnMax = c : Debug.WriteLine("columnMax=" & columnMax)

            ' Parallel loop is unstable there?
            'If parallelLoop AndAlso r >= parallelMinSize AndAlso c >= parallelMinSize Then

            '    Parallel.For(0, r - 1 + 1,
            '        Sub(i)
            '            Parallel.For(0, c - 1 + 1,
            '                Sub(j)
            '                    e(i, j)
            '                End Sub)
            '        End Sub)

            'ElseIf parallelLoop AndAlso r >= parallelMinSize Then

            '    Parallel.For(0, r - 1 + 1,
            '        Sub(i)
            '            For j = 0 To c - 1
            '                e(i, j)
            '            Next
            '        End Sub)

            'ElseIf parallelLoop AndAlso c >= parallelMinSize Then

            '    For i = 0 To r - 1
            '        Dim i0 = i
            '        Parallel.For(0, c - 1 + 1,
            '            Sub(j)
            '                e(i0, j)
            '            End Sub)
            '    Next

            'Else

                For i = 0 To r - 1
                    For j = 0 To c - 1
                        e(i, j)
                    Next
                Next

            'End If

        End Sub

#End Region

#Region "Public operations"

        Public Function AddColumn(m2 As Matrix) As Matrix

            If m2.c <> 1 OrElse m2.r <> Me.r Then Throw New ArgumentException("Invalid dimensions")

            Dim newMatrix = New Double(Me.r - 1, Me.c + 1 - 1) {}
            'Dim m = Me.data
            For i = 0 To Me.r - 1
                newMatrix(i, 0) = m2.data(i, 0)
            Next
            'MatrixLoop((Sub(i, j) newMatrix(i, j + 1) = m(i, j)), r, c)
            MatrixLoop((Sub(i, j) newMatrix(i, j + 1) = Me.data(i, j)), r, c)

            Dim result As Matrix = newMatrix
            Return result

        End Function

        ''' <summary>
        ''' Apply a function to every element of the array
        ''' </summary>
        Public Sub Map(lambdaFct As Func(Of Double, Double))

            For i = 0 To Me.r - 1
                For j = 0 To Me.c - 1
                    Me.data(i, j) = lambdaFct.Invoke(Me.data(i, j))
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
                    slice0(i - r1, j - c1) = Me.data(i, j)
                Next
            Next

            Dim m As Matrix = slice0
            Return m

        End Function

#End Region

#Region "Private operations"

        ''' <summary>
        ''' Add each element of the matrices
        ''' </summary>
        Private Sub Add(m As Matrix)

            For i = 0 To Me.r - 1
                For j = 0 To Me.c - 1
                    Me.data(i, j) += m.data(i, j)
                Next
            Next

        End Sub

        '''' <summary>
        '''' Subtract a value to each element of the array
        '''' </summary>
        Private Overloads Sub Subtract(n#)

            For i = 0 To Me.r - 1
                For j = 0 To Me.c - 1
                    Me.data(i, j) -= n
                Next
            Next

        End Sub

        ''' <summary>
        ''' Subtract each element of the matrices
        ''' </summary>
        Private Overloads Sub Subtract(m As Matrix)

            For i = 0 To Me.r - 1
                For j = 0 To Me.c - 1
                    Me.data(i, j) -= m.data(i, j)
                Next
            Next

        End Sub

        ''' <summary>
        ''' Subtract 2 matrices and return a new matrix
        ''' </summary>
        Private Overloads Shared Function Subtract(a As Matrix, b As Matrix) As Matrix

            Dim c As New Matrix(a.r, a.c)

            For i = 0 To c.r - 1
                For j = 0 To c.c - 1
                    c.data(i, j) = a.data(i, j) - b.data(i, j)
                Next
            Next

            Return c

        End Function

        ''' <summary>
        ''' Scalar product: Multiply each element of the array with the given number
        ''' </summary>
        Private Overloads Sub Multiply(n#)

            For i = 0 To Me.r - 1
                For j = 0 To Me.c - 1
                    Me.data(i, j) *= n
                Next
            Next

        End Sub

        ''' <summary>
        ''' Hadamard product (element-wise multiplication):
        ''' Multiply each element of the array with each element of the given array
        ''' </summary>
        Private Overloads Sub Multiply(m As Matrix)

            For i = 0 To Me.r - 1
                For j = 0 To Me.c - 1
                    Me.data(i, j) *= m.data(i, j)
                Next
            Next

        End Sub

        ''' <summary>
        ''' Matrix product
        ''' </summary>
        Private Overloads Shared Function Multiply(a As Matrix, b As Matrix) As Matrix

            If a.c <> b.r Then
                Throw New Exception("Columns of A must match columns of B")
            End If

            Dim ab As New Matrix(a.r, b.c)

            If ab.r > rowMax Then rowMax = ab.r : Debug.WriteLine("rowMax=" & rowMax)
            If ab.c > columnMax Then columnMax = ab.c : Debug.WriteLine("columnMax=" & columnMax)

            If parallelLoop AndAlso ab.r >= parallelMinSize AndAlso
                                    ab.c >= parallelMinSize Then

                Parallel.For(0, ab.r - 1 + 1,
                    Sub(i)
                        Parallel.For(0, ab.c - 1 + 1,
                            Sub(j)
                                Dim sum# = 0
                                For k = 0 To a.c - 1
                                    sum += a.data(i, k) * b.data(k, j)
                                Next
                                ab.data(i, j) = sum
                            End Sub)
                    End Sub)

            ElseIf parallelLoop AndAlso ab.r >= parallelMinSize Then

                Parallel.For(0, ab.r - 1 + 1,
                    Sub(i)
                        For j = 0 To ab.c - 1
                            Dim sum# = 0
                            For k = 0 To a.c - 1
                                sum += a.data(i, k) * b.data(k, j)
                            Next
                            ab.data(i, j) = sum
                        Next
                    End Sub)

            ElseIf parallelLoop AndAlso ab.c >= parallelMinSize Then

                For i = 0 To ab.r - 1
                    Dim i0 = i
                    Parallel.For(0, ab.c - 1 + 1,
                        Sub(j)
                            Dim sum# = 0
                            For k = 0 To a.c - 1
                                sum += a.data(i0, k) * b.data(k, j)
                            Next
                            ab.data(i0, j) = sum
                        End Sub)
                Next

            Else

                For i = 0 To ab.r - 1
                    For j = 0 To ab.c - 1
                        Dim sum# = 0
                        For k = 0 To a.c - 1
                            sum += a.data(i, k) * b.data(k, j)
                        Next
                        ab.data(i, j) = sum
                    Next
                Next

            End If

            Return ab

        End Function

        ''' <summary>
        ''' Compute absolute values of a matrix
        ''' </summary>
        Private Function Abs_() As Matrix

            Dim c As New Matrix(Me.r, Me.c)
            For i = 0 To Me.r - 1
                For j = 0 To Me.c - 1
                    c.data(i, j) = Math.Abs(Me.data(i, j))
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
                    sum += Me.data(i, j)
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

        Public Function GetValue#(r%, c%)
            Return Me.data(r, c)
        End Function

        Public Function GetRow(r%) As Matrix

            Dim row = New Double(0, Me.c - 1) {}
            For j = 0 To Me.c - 1
                row(0, j) = Me.data(r, j)
            Next
            Return row

        End Function

        Public Function GetColumn(c%) As Matrix

            Dim column = New Double(Me.r - 1, 0) {}
            For i = 0 To Me.r - 1
                column(i, 0) = Me.data(i, c)
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

            MatrixLoop((Sub(i, j) Me.data(i, j) =
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
                    c.data(i, j) = CDbl(IIf(Math.Abs(Me.data(i, j)) <= minThreshold, 1.0#, 0.0#))
                Next
            Next

            Return c

        End Function

#End Region

    End Class

#End If

End Namespace