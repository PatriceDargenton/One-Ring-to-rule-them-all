
Imports System.ComponentModel ' DescriptionAttribute

Public Class clsMLPHelper

    Public Shared Function GetVector(singleArray!(,), index%) As Single()

        Dim length = singleArray.GetLength(1)
        Dim vect!(0 To length - 1)
        For k = 0 To length - 1
            vect(k) = singleArray(index, k)
        Next
        Return vect

    End Function

    Public Shared Function FillArray(singleArray2D!(,), singleArray1D!(), index%) As Single(,)
        Dim nbItems = singleArray1D.GetLength(0)
        For j = 0 To nbItems - 1
            singleArray2D(index, j) = singleArray1D(j)
        Next
        Return singleArray2D
    End Function

    Public Shared Function FillArray(doubleArray2D#(,), singleArray1D!(), index%) As Double(,)
        Dim nbItems = singleArray1D.GetLength(0)
        For j = 0 To nbItems - 1
            doubleArray2D(j, index) = singleArray1D(j)
        Next
        Return doubleArray2D
    End Function

    Public Shared Function FillArray2(doubleArray2D#(,), singleArray1D!(), index%) As Double(,)
        Dim nbItems = singleArray1D.GetLength(0)
        For j = 0 To nbItems - 1
            doubleArray2D(index, j) = singleArray1D(j)
        Next
        Return doubleArray2D
    End Function

    Public Shared Function ConvertSingleToDouble(inputs!(,)) As Double(,)
        Dim length0 = inputs.GetLength(0)
        Dim length1 = inputs.GetLength(1)
        Dim arr#(0 To length0 - 1, 0 To length1 - 1)
        For i = 0 To length0 - 1
            For j = 0 To length1 - 1
                arr(i, j) = inputs(i, j)
            Next
        Next
        Return arr
    End Function

    Public Shared Function ConvertSingleToDouble1D(inputs!()) As Double()
        Dim length0 = inputs.GetLength(0)
        Dim arr#(0 To length0 - 1)
        For i = 0 To length0 - 1
            arr(i) = inputs(i)
        Next
        Return arr
    End Function

    Public Shared Function ConvertDoubleToSingle(inputs#()) As Single()
        Dim length0 = inputs.GetLength(0)
        Dim arr!(0 To length0 - 1)
        For i = 0 To length0 - 1
            arr(i) = CSng(inputs(i))
        Next
        Return arr
    End Function

    Public Shared Function ConvertDoubleToSingle2D(inputs#(,)) As Single(,)
        Dim length0 = inputs.GetLength(0)
        Dim length1 = inputs.GetLength(1)
        Dim arr!(0 To length0 - 1, 0 To length1 - 1)
        For i = 0 To length0 - 1
            For j = 0 To length1 - 1
                arr(i, j) = CSng(inputs(i, j))
            Next
        Next
        Return arr
    End Function

    Public Shared Function TransformDoubleArrayToJaggedArray(inputs#(,)) As Double()()

        ' Transform a 2D array into a jagged array

        Dim length0 = inputs.GetLength(0)
        Dim length1 = inputs.GetLength(1)
        Dim arr As Double()() = New Double(length0 - 1)() {}
        For i = 0 To length0 - 1
            arr(i) = New Double() {}
            ReDim arr(i)(length1 - 1)
            For j = 0 To length1 - 1
                arr(i)(j) = inputs(i, j)
            Next j
        Next i

        Return arr

    End Function

    Public Shared Function TransformDoubleArrayToJaggedArray1D(inputs#()) As Double()()

        ' Transform a 1D array into a jagged array

        Dim length0 = 1
        Dim length1 = inputs.GetLength(0)
        Dim arr As Double()() = New Double(length0 - 1)() {}
        For i = 0 To length0 - 1
            arr(i) = New Double() {}
            ReDim arr(i)(length1 - 1)
            For j = 0 To length1 - 1
                arr(i)(j) = inputs(j)
            Next j
        Next i

        Return arr

    End Function

    Public Shared Function Compare(val1#, val2#, dec%) As Boolean

        If Double.IsNaN(val1) AndAlso Double.IsNaN(val2) Then Return True
        Dim delta# = Math.Abs(Math.Round(val2 - val1, dec))
        If delta = 0 Then Return True
        Return False

    End Function

    Public Shared Function CompareArray(inputa!(,), inputb!(,)) As Boolean
        Dim length0a = inputa.GetLength(0)
        Dim length1a = inputa.GetLength(1)
        Dim length0b = inputb.GetLength(0)
        Dim length1b = inputb.GetLength(1)
        If length0a <> length0b Then Return False
        If length1a <> length1b Then Return False
        For i = 0 To length0a - 1
            For j = 0 To length1a - 1
                If inputa(i, j) <> inputb(i, j) Then Return False
            Next
        Next
        Return True
    End Function

    Public Shared Function ReadEnumDescription$(myEnum As [Enum])

        Dim fi As Reflection.FieldInfo = myEnum.GetType().GetField(myEnum.ToString())
        Dim attr() As DescriptionAttribute = DirectCast(
           fi.GetCustomAttributes(GetType(DescriptionAttribute), False),
           DescriptionAttribute())
        If attr.Length > 0 Then
            Return attr(0).Description
        Else
            Return myEnum.ToString()
        End If

    End Function

End Class
