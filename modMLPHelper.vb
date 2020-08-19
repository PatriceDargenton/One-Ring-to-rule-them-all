
Imports System.Runtime.CompilerServices ' Extension

Public Module modMLPHelper

    Public Const format2Dec$ = "0.00"
    Public Const format6Dec$ = "0.000000"

    <Extension()>
    Public Function ReplaceCommaByDot$(text$)
        Return text.Replace(",", ".")
    End Function

    Public Function isConsoleApp() As Boolean

        ' https://www.codeproject.com/Questions/865642/Csharp-equivalent-for-sharpIf-TARGET-equals-winexe
#If TARGET = "winexe" Then
        ' Insert code to be compiled for a Windows application.
        Return False
#ElseIf TARGET = "exe" Then
        ' Insert code to be compiled for a console application.
        return True
#End If

        ' Ok, but show "'System.IO.IOException' in mscorlib.dll" message exception 
        '  in WinForm app in the Debug windows of Visual Studio
        'Try
        '    Return Console.WindowHeight > 0
        'Catch 'ex As Exception
        '    Return False
        'End Try

        ' Does not work:

        'Dim isReallyAConsoleWindow = Console.Read() <> -1
        'Return isReallyAConsoleWindow

        ' .NET Core:
        'System.Reflection.PortableExecutable.IsConsoleApplication

        'Dim is_console_app = Not Console.OpenStandardInput(1) = System.IO.Stream.Null
        'Dim is_console_app = Not Console.In = System.IO.Stream.Null

    End Function

End Module
