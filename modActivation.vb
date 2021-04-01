
' Inspired from
' https://www.codeproject.com/Articles/1205732/Build-Simple-AI-NET-Library-Part-Perceptron

' https://en.wikipedia.org/wiki/Activation_function
' https://fr.wikipedia.org/wiki/Fonction_d'activation

' General formulas for derivate:
' (fg)'  = f'g+fg'
' (f/g)' = f'g-fg'/g^2
' (f°g)' = (f'°g)g' = f(g(x))' = f'(g(x))g'(x)

Imports Perceptron.MLP.ActivationFunction

Public Module modFctAct

    ''' <summary>
    ''' For non-derivatable activation functions, use an alternate derivative function
    ''' </summary>
    Public Const useAlternateDerivativeFunction As Boolean = True

    Public Const debugActivationFunction As Boolean = False

    Public Enum enumActivationFunction
        Undefined = 0
        Identity = 1
        Sigmoid = 2
        HyperbolicTangent = 3
        Gaussian = 4

        ''' <summary>
        ''' Arc tangent (Atan or tan^-1: inverse of tangent function)
        ''' </summary>
        ArcTangent = 5

        Sinus = 6

        ''' <summary>
        ''' Exponential Linear Units (ELU)
        ''' </summary>
        ELU = 7

        ''' <summary>
        ''' Rectified Linear Units (ReLU)
        ''' f(x) = Max(0, x) : return x if x > 0 or return 0
        ''' </summary>
        ReLu = 8

        ''' <summary>
        ''' Rectified Linear Units (ReLU) with sigmoid for derivate
        ''' </summary>
        ReLuSigmoid = 9

        DoubleThreshold = 10

    End Enum

    ''' <summary>
    ''' An activation function expressed from its direct function, e.g. f'(x)=g(f(x)),
    '''  can be optimized
    ''' </summary>
    Public Enum enumActivationFunctionOptimized

        Sigmoid = 1

        HyperbolicTangent = 2

        ''' <summary>
        ''' Exponential Linear Units (ELU)
        ''' </summary>
        ELU = 3

    End Enum

    ''' <summary>
    ''' Activation function type
    ''' </summary>
    Public Enum enumActivationFunctionType

        ''' <summary>
        ''' Normal (can use general activation function)
        ''' (ex.: MLPClassic, MLPMatrix, MLPMatrixVec)
        ''' </summary>
        Normal = 1

        ''' <summary>
        ''' Specific code (use only specific code for general activation function) (ex.: ?)
        ''' </summary>
        SpecificCode = 2

        ''' <summary>
        ''' Specific code and optimized (use only specific code with only optimized activation function)
        ''' (ex.: MLPTensor, MLPRProp)
        ''' </summary>
        SpecificCodeOptimized = 3

        ''' <summary>
        ''' Both normal and specific code (can use general or specific code for activation function, if defined)
        ''' (ex.: MLPOOP)
        ''' </summary>
        BothNormalAndSpecificCode = 4

        ''' <summary>
        ''' Optimized: can use only activation function expressed from its direct function,
        '''  e.g. f'(x)=g(f(x)) (ex.: ?)
        ''' </summary>
        Optimized = 5

        ''' <summary>
        ''' Library (can use general activation function proposed in the library)
        ''' (ex.: MLPEncog)
        ''' </summary>
        Library = 6

        ''' <summary>
        ''' Library (can use only optimized activation function proposed in the library)
        ''' (ex.: MLPAccord, MLPKeras, MLPTensorFlow, MLPNeuralNet)
        ''' </summary>
        LibraryOptimized = 7

    End Enum

End Module

Namespace MLP.ActivationFunction

    Public Interface IActivationFunction

        ''' <summary>
        ''' Is non linear function?
        ''' </summary>
        Function IsNonLinear() As Boolean

        ''' <summary>
        ''' Activation function
        ''' </summary>
        Function Activation#(x#, Optional gain# = 1, Optional center# = 0)

        ''' <summary>
        ''' Derivative function
        ''' </summary>
        Function Derivative#(x#, Optional gain# = 1, Optional center# = 0)

        ''' <summary>
        ''' Does the derivative f'(x) depend on the original function f(x)?
        ''' i.e. f'(x)=g(f(x))
        ''' </summary>
        Function DoesDerivativeDependOnOriginalFunction() As Boolean

        ''' <summary>
        ''' Derivative computed from the direct function, when possible: f'(x)=g(f(x))
        ''' </summary>
        Function DerivativeFromOriginalFunction#(x#, gain#)

    End Interface

    ''' <summary>
    ''' Identity Function : Always returns the same value that was used as its argument
    ''' f(x) = alpha.x
    ''' </summary>
    Public Class IdentityFunction : Implements IActivationFunction

        Public Function IsNonLinear() As Boolean Implements IActivationFunction.IsNonLinear
            Return True
        End Function

        Function DoesDerivativeDependOnOriginalFunction() As Boolean Implements IActivationFunction.DoesDerivativeDependOnOriginalFunction
            Return False
        End Function

        Public Function Activation#(x#, Optional gain# = 1, Optional center# = 0) Implements IActivationFunction.Activation
            Dim xc# = x - center
            Return xc * gain
        End Function

        Public Function Derivative#(x#, Optional gain# = 1, Optional center# = 0) Implements IActivationFunction.Derivative
            Return gain
        End Function

        Public Function DerivativeFromOriginalFunction#(x#, gain#) Implements IActivationFunction.DerivativeFromOriginalFunction
            Return gain
        End Function

    End Class

    ''' <summary>
    ''' Implements f(x) = Sigmoid
    ''' </summary>
    Public Class SigmoidFunction : Implements IActivationFunction

        Public Function IsNonLinear() As Boolean Implements IActivationFunction.IsNonLinear
            Return False
        End Function

        Function DoesDerivativeDependOnOriginalFunction() As Boolean Implements IActivationFunction.DoesDerivativeDependOnOriginalFunction
            Return True
        End Function

        Public Function Activation#(x#, Optional gain# = 1, Optional center# = 0) Implements IActivationFunction.Activation
            Return CommonActivation(x, gain, center)
        End Function

        Private Shared Function CommonActivation#(x#, gain#, center#)

            'Const expMax As Boolean = False
            Dim xc# = x - center
            Dim xg# = -gain * xc
            Dim y#
            ' To avoid arithmetic overflow
            If xg > clsMLPGeneric.expMax Then
                y = 0 '1
                'If expMax Then y = clsMLPGeneric.expMax
            ElseIf xg < -clsMLPGeneric.expMax Then
                y = 1 '0
                'If expMax Then y = -clsMLPGeneric.expMax
            Else
                y = 1 / (1 + Math.Exp(xg))
            End If

            If debugActivationFunction Then
                Dim sig# = 1 / (1 + Math.Exp(xg))
                If Not clsMLPHelper.Compare(y, sig, dec:=5) Then Stop
            End If

            Return y

        End Function

        Public Function Derivative#(x#, Optional gain# = 1, Optional center# = 0) Implements IActivationFunction.Derivative
            Return CommonDerivative(x, gain, center)
        End Function

        Public Shared Function CommonDerivative#(x#, gain#, center#)

            Dim xc# = x - center
            Dim fx# = CommonActivation(x, gain, center)
            Dim y# = gain * fx * (1 - fx) ' 31/07/2020

            'If gain = 1 Then
            '    Dim fx# = CommonActivation(x, gain, center)
            '    y = fx * (1 - fx)
            'Else
            '    Dim c# = -gain
            '    Dim exp# = Math.Exp(c * xc)
            '    Dim expP1# = 1 + exp
            '    y = -c * exp / (expP1 * expP1)
            'End If

            ' https://www.wolframalpha.com/input/?i=sigmoid+(alpha+*+x)+derivate
            If debugActivationFunction Then

                Dim cosH# = Math.Cosh(gain * xc)
                Dim y2# = gain / ((2 * cosH) + 2)
                If Not clsMLPHelper.Compare(y, y2, dec:=5) Then Stop

                Dim c# = -gain
                Dim exp# = Math.Exp(c * xc)
                Dim expP1# = 1 + exp
                Dim y3 = -c * exp / (expP1 * expP1)
                If Not clsMLPHelper.Compare(y, y3, dec:=5) Then Stop

            End If

            Return y

        End Function

        Public Function DerivativeFromOriginalFunction#(fx#, gain#) Implements IActivationFunction.DerivativeFromOriginalFunction
            'If gain <> 1 Then Return 0
            'Dim y# = CommonDerivativeFromOriginalFunction(fx)
            Dim y# = gain * CommonDerivativeFromOriginalFunction(fx) ' 31/07/2020
            Return y
        End Function

        Public Shared Function CommonDerivativeFromOriginalFunction#(fx#)
            Dim y# = fx * (1 - fx)
            Return y
        End Function

    End Class

    ''' <summary>
    ''' Implements f(x) = Hyperbolic Tangent (Bipolar Sigmoid)
    ''' </summary>
    Public Class HyperbolicTangentFunction : Implements IActivationFunction

        Public Function IsNonLinear() As Boolean Implements IActivationFunction.IsNonLinear
            Return False
        End Function

        Function DoesDerivativeDependOnOriginalFunction() As Boolean Implements IActivationFunction.DoesDerivativeDependOnOriginalFunction
            Return True
        End Function

        Public Function Activation#(x#, Optional gain# = 1, Optional center# = 0) Implements IActivationFunction.Activation

            'Const expMax As Boolean = True
            Dim xc# = x - center
            'Dim xg# = -2 * gain * xc
            Dim xg# = -gain * xc ' 31/07/2020
            Dim y#
            If xg > clsMLPGeneric.expMax20 Then
                y = -1 ' 0
                'If expMax Then y = -1 'clsMLPGeneric.expMax
            ElseIf xg < -clsMLPGeneric.expMax20 Then
                y = 1
                'If expMax Then y = 1 '-clsMLPGeneric.expMax
            Else
                y = 2 / (1 + Math.Exp(xg)) - 1 ' = Math.Tanh(-xg / 2)
                'y = Math.Tanh(-xg / 2)
            End If

            ' https://www.wolframalpha.com/input/?i=HyperbolicTangent
            If debugActivationFunction Then
                Dim th# = Math.Tanh(-xg / 2)
                If Not clsMLPHelper.Compare(y, th, dec:=5) Then Stop

                Dim e2x# = Math.Exp(-xg)
                Dim th2# = (e2x - 1) / (e2x + 1)
                If Not clsMLPHelper.Compare(y, th2, dec:=5) Then Stop
            End If

            Return y

        End Function

        Public Function Derivative#(x#, Optional gain# = 1, Optional center# = 0) Implements IActivationFunction.Derivative

            Dim xc# = x - center
            Dim fx# = Activation(x, gain, center)
            Dim y# = gain * (1 - fx * fx) / 2 ' 31/07/2020

            'Dim y#
            'If gain = 1 Then
            '    Dim fx# = Activation(x, gain, center)
            '    y = 1 - fx * fx
            'Else
            '    Dim xg# = -2 * gain
            '    Dim exp# = Math.Exp(xg * xc)
            '    Dim expP1# = 1 + exp
            '    y = -2 * xg * exp / (expP1 * expP1)
            'End If

            ' https://www.wolframalpha.com/input/?i=2+%2F+%281+%2B+Exp%28alpha+x%29%29+-+1+derivative
            If debugActivationFunction Then
                'Dim xg# = -2 * gain
                'Dim xg# = -gain ' 31/07/2020
                Dim xg# = gain ' 02/10/2020
                Dim exp# = Math.Exp(xg * xc)
                Dim expP1# = 1 + exp
                'Dim y2 = -2 * xg * exp / (expP1 * expP1)
                Dim y2 = 2 * xg * exp / (expP1 * expP1) ' 19/09/2020
                If Not clsMLPHelper.Compare(y, y2, dec:=5) Then Stop
            End If

            Return y

        End Function

        Public Function DerivativeFromOriginalFunction#(x#, gain#) Implements IActivationFunction.DerivativeFromOriginalFunction

            'If gain <> 1 Then Return 0
            'Dim y# = 1 - x * x
            Dim y# = gain * (1 - x * x) / 2 ' 02/10/2020

            'If debugActivationFunction Then
            '    Dim y2# = (1 - x) * (1 + x)
            '    If Not clsMLPHelper.Compare(y, y2, dec:=5) Then Stop
            'End If

            Return y

        End Function

    End Class

    ''' <summary>
    ''' Implements f(x) = Gaussian
    ''' </summary>
    Public Class GaussianFunction : Implements IActivationFunction

        Public Function IsNonLinear() As Boolean Implements IActivationFunction.IsNonLinear
            Return False
        End Function

        Function DoesDerivativeDependOnOriginalFunction() As Boolean Implements IActivationFunction.DoesDerivativeDependOnOriginalFunction
            Return False
        End Function

        Public Function Activation#(x#, Optional gain# = 1, Optional center# = 0) Implements IActivationFunction.Activation

            Const expMax As Boolean = False
            Dim xc# = x - center
            Dim xg# = -gain * xc * xc
            Dim y#
            If xg > clsMLPGeneric.expMax Then
                y = 1
                If expMax Then y = clsMLPGeneric.expMax
            ElseIf xg < -clsMLPGeneric.expMax Then
                y = 0
                If expMax Then y = -clsMLPGeneric.expMax
            Else
                ' https://www.wolframalpha.com/input/?i=exp+%28-alpha+x2%29
                y = Math.Exp(xg)
            End If
            Return y

        End Function

        Public Function Derivative#(x#, Optional gain# = 1, Optional center# = 0) Implements IActivationFunction.Derivative

            Dim xc# = x - center
            'Dim c# = -gain * gain
            'Dim exp# = Math.Exp(c * xc * xc)
            'Dim y# = 2 * c * xc * exp
            Dim y# = -2 * gain * xc * Math.Exp(-gain * xc * xc) ' 16/10/2020

            Return y

        End Function

        Public Function DerivativeFromOriginalFunction#(fx#, gain#) Implements IActivationFunction.DerivativeFromOriginalFunction
            Return 0
        End Function

    End Class

    ''' <summary>
    ''' Implements f(x) = Arc tangent (Atan or tan^-1: inverse of tangent function)
    ''' </summary>
    Public Class ArcTangentFunction : Implements IActivationFunction

        Public Function IsNonLinear() As Boolean Implements IActivationFunction.IsNonLinear
            Return False
        End Function

        Function DoesDerivativeDependOnOriginalFunction() As Boolean Implements IActivationFunction.DoesDerivativeDependOnOriginalFunction
            Return False
        End Function

        Public Function Activation#(x#, Optional gain# = 1, Optional center# = 0) Implements IActivationFunction.Activation
            Dim xc# = x - center
            Dim xg# = gain * xc
            Dim y# = gain * Math.Atan(xg)
            Return y
        End Function

        Public Function Derivative#(x#, Optional gain# = 1, Optional center# = 0) Implements IActivationFunction.Derivative
            Dim xc# = x - center
            ' https://www.wolframalpha.com/input/?i=arctan(alpha+*+x)+derivative
            Dim y# = gain / (1 + gain * gain * xc * xc)
            Return y
        End Function

        Public Function DerivativeFromOriginalFunction#(fx#, gain#) Implements IActivationFunction.DerivativeFromOriginalFunction
            Return 0
        End Function

    End Class

    ''' <summary>
    ''' Implements f(x) = sin(alpha.x)
    ''' </summary>
    Public Class SinusFunction : Implements IActivationFunction

        Public Function IsNonLinear() As Boolean Implements IActivationFunction.IsNonLinear
            Return False
        End Function

        Function DoesDerivativeDependOnOriginalFunction() As Boolean Implements IActivationFunction.DoesDerivativeDependOnOriginalFunction
            Return False
        End Function

        Public Function Activation#(x#, Optional gain# = 1, Optional center# = 0) Implements IActivationFunction.Activation
            Dim xc# = x - center
            ' https://www.wolframalpha.com/input/?i=sin+%28alpha+x%29
            Dim y# = gain * Math.Sin(xc)
            Return y
        End Function

        Public Function Derivative#(x#, Optional gain# = 1, Optional center# = 0) Implements IActivationFunction.Derivative
            Dim xc# = x - center
            'Dim y# = gain * Math.Cos(xc)
            Dim y# = gain * Math.Cos(gain * xc) ' 17/10/2020
            Return y
        End Function

        Public Function DerivativeFromOriginalFunction#(fx#, gain#) Implements IActivationFunction.DerivativeFromOriginalFunction
            Return 0
        End Function

    End Class

    ''' <summary>
    ''' Implements f(x) = ELU(x) : Exponential Linear Units
    ''' </summary>
    Public Class ELUFunction : Implements IActivationFunction

        Public Function IsNonLinear() As Boolean Implements IActivationFunction.IsNonLinear
            Return False
        End Function

        Function DoesDerivativeDependOnOriginalFunction() As Boolean Implements IActivationFunction.DoesDerivativeDependOnOriginalFunction
            Return True
        End Function

        Public Function Activation#(x#, Optional gain# = 1, Optional center# = 0) Implements IActivationFunction.Activation
            Dim xc# = x - center
            Dim y#
            If xc >= 0 Then
                y = xc
            Else
                y = gain * (Math.Exp(xc) - 1)
            End If
            Return y
        End Function

        Public Function Derivative#(x#, Optional gain# = 1, Optional center# = 0) Implements IActivationFunction.Derivative

            ' If gain < 0 the derivate is undefined
            If gain < 0 Then Return 0

            Dim xc# = x - center
            Dim y#
            If xc >= 0 Then
                y = 1
            Else
                ' Function: alpha(exp(x)-1)
                ' https://www.wolframalpha.com/input/?i=alpha(exp(x)-1)
                ' Derivate: alpha . exp(x) = f(x) + alpha = alpha(exp(x)-1) + alpha
                ' https://www.wolframalpha.com/input/?i=alpha(exp(x)-1)+derivate
                Dim fx# = Activation(x, gain, center)
                y = fx + gain
            End If
            Return y

        End Function

        Public Function DerivativeFromOriginalFunction#(fx#, gain#) Implements IActivationFunction.DerivativeFromOriginalFunction

            ' If gain < 0 the derivate is undefined
            If gain < 0 Then Return 0

            Dim y#
            If fx >= 0 Then
                y = 1
            Else
                y = fx + gain
            End If
            Return y

        End Function

    End Class

    ''' <summary>
    ''' Implements f(x) = ReLU(x) : Rectified Linear Unit
    ''' </summary>
    Public Class ReLuFunction : Implements IActivationFunction

        Public Function IsNonLinear() As Boolean Implements IActivationFunction.IsNonLinear
            Return True
        End Function

        Function DoesDerivativeDependOnOriginalFunction() As Boolean Implements IActivationFunction.DoesDerivativeDependOnOriginalFunction
            'If useAlternateDerivativeFunction Then Return True
            Return False
        End Function

        Public Function Activation#(x#, Optional gain# = 1, Optional center# = 0) Implements IActivationFunction.Activation
            Dim xc# = x - center
            Dim y# = Math.Max(xc * gain, 0)
            Return y
        End Function

        Public Function Derivative#(x#, Optional gain# = 1, Optional center# = 0) Implements IActivationFunction.Derivative
            Dim xc# = x - center
            If xc >= 0 Then Return gain
            Return 0
        End Function

        Public Function DerivativeFromOriginalFunction#(fx#, gain#) Implements IActivationFunction.DerivativeFromOriginalFunction
            ' ReLUFunction
            'If useAlternateDerivativeFunction Then _
            '    Return SigmoidFunction.CommonDerivativeFromOriginalFunction(fx)
            Return 0
        End Function

    End Class

    ''' <summary>
    ''' Implements f(x) = ReLU(x) : Rectified Linear Unit (ReLU) with sigmoid for derivate
    ''' </summary>
    Public Class ReLuSigmoidFunction : Implements IActivationFunction

        Public Function IsNonLinear() As Boolean Implements IActivationFunction.IsNonLinear
            Return True
            'Return False ' Linear using sigmoid?
        End Function

        Function DoesDerivativeDependOnOriginalFunction() As Boolean Implements IActivationFunction.DoesDerivativeDependOnOriginalFunction
            'If useAlternateDerivativeFunction Then Return True
            Return False
        End Function

        Public Function Activation#(x#, Optional gain# = 1, Optional center# = 0) Implements IActivationFunction.Activation
            Dim xc# = x - center
            Dim y# = Math.Max(xc * gain, 0)
            Return y
        End Function

        Public Function Derivative#(x#, Optional gain# = 1, Optional center# = 0) Implements IActivationFunction.Derivative
            Return SigmoidFunction.CommonDerivative(x, gain, center)
        End Function

        Public Function DerivativeFromOriginalFunction#(fx#, gain#) Implements IActivationFunction.DerivativeFromOriginalFunction
            Return SigmoidFunction.CommonDerivativeFromOriginalFunction(fx)
        End Function

    End Class

    ''' <summary>
    ''' f(x) = Double-threshold(x)
    ''' </summary>
    Public Class DoubleThresholdFunction : Implements IActivationFunction

        Public Function IsNonLinear() As Boolean Implements IActivationFunction.IsNonLinear
            Return True
        End Function

        Function DoesDerivativeDependOnOriginalFunction() As Boolean Implements IActivationFunction.DoesDerivativeDependOnOriginalFunction
            'If useAlternateDerivativeFunction Then Return True
            Return False
        End Function

        Public Function Activation#(x#, Optional gain# = 1, Optional center# = 0) Implements IActivationFunction.Activation

            Dim xc# = x - center
            Dim reducedGain# = gain / 8
            Dim x2#
            If reducedGain = 0 Then
                x2 = 0.5
            Else
                x2 = (xc + 0.5 / reducedGain) * reducedGain
            End If
            Dim y#
            If x2 < 0.33 Then
                y = 0
            ElseIf x2 > 0.66 Then
                y = 1
            Else
                y = x2 / 0.33 - 1
            End If
            Return y

        End Function

        Public Function Derivative#(x#, Optional gain# = 1, Optional center# = 0) Implements IActivationFunction.Derivative
            If useAlternateDerivativeFunction Then _
                Return SigmoidFunction.CommonDerivative(x, gain, center)
            Return 0
        End Function

        Public Function DerivativeFromOriginalFunction#(fx#, gain#) Implements IActivationFunction.DerivativeFromOriginalFunction
            If useAlternateDerivativeFunction Then _
                Return SigmoidFunction.CommonDerivativeFromOriginalFunction(fx)
            Return 0
        End Function

    End Class

End Namespace