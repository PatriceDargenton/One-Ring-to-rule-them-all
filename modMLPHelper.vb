
Imports System.Runtime.CompilerServices ' Extension

Public Module modMLPHelper

    Public Const format2Dec$ = "0.00"
    Public Const format6Dec$ = "0.000000"

    <Extension()>
    Public Function ReplaceCommaByDot$(text$)
        Return text.Replace(",", ".")
    End Function

    Public Function isConsoleApp() As Boolean

        'Dim isReallyAConsoleWindow = Console.Read() <> -1
        'Return isReallyAConsoleWindow

        Try
            Return Console.WindowHeight > 0
        Catch 'ex As Exception
            Return False
        End Try

    End Function

End Module
