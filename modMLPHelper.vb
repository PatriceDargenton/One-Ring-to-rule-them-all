
Imports System.Runtime.CompilerServices ' Extension

Public Module modMLPHelper

    Public Const format2Dec$ = "0.00"
    Public Const format6Dec$ = "0.000000"

    <Extension()>
    Public Function ReplaceCommaByDot$(text$)
        Return text.Replace(",", ".")
    End Function

End Module
