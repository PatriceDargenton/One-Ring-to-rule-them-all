
' From: https://github.com/nlabiris/perceptrons : C# -> VB .NET conversion

Imports System.Runtime.CompilerServices ' Extension

''' <summary>
''' Add functions to standard Random generator
''' </summary>
Public Module modExt

    <Extension>
    Public Function NextDoubleGreaterThanZero#(rand As Random, minValue#, maxValue#, minAbsValue#)

Retry:
        Dim r# = rand.NextDouble * Math.Abs(maxValue - minValue) + minValue
        If Math.Abs(r) < minAbsValue Then GoTo Retry
        Return r

    End Function

    ''' <summary>
    ''' Add NextDouble function to standard Random generator
    ''' </summary>
    <Extension>
    Public Function NextDouble#(rand As Random, minValue#, maxValue#)
        Return rand.NextDouble * Math.Abs(maxValue - minValue) + minValue
    End Function

    ''' <summary>
    ''' Add NextFloat function to standard Random generator
    ''' </summary>
    <Extension>
    Public Function NextFloat!(rand As Random)
        Return CSng(rand.NextDouble)
    End Function

    ''' <summary>
    ''' Add NextFloat function to standard Random generator
    ''' </summary>
    <Extension>
    Public Function NextFloat!(rand As Random, maxValue!)
        Return CSng(rand.NextDouble * maxValue)
    End Function

    ''' <summary>
    ''' Add NextFloat function to standard Random generator
    ''' </summary>
    <Extension>
    Public Function NextFloat!(rand As Random, minValue!, maxValue!)
        Return CSng(rand.NextDouble * Math.Abs(CDbl(maxValue - minValue))) + minValue
    End Function

End Module