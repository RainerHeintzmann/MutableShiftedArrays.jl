# MutableShiftedArrays.jl
A lightweight toolbox representing a ShiftedArray which is mutable. The code was based on `ShiftedArrays.jl`.
Via the extension mechanism, `CUDA.jl` support is provided both for mutating and non-mutating operations.
Mutations to elements outside the boundary of the original array are silently ignored and the `default` value is returned upon subsequent read operations.

This code and most of the documentation is based on https://github.com/JuliaArrays/ShiftedArrays.jl with contributers
@piever, @Nosferican, @roflmaostc, @nalimilan, @Felix-Gauthier, @RainerHeintzmann, @LilithHafner

The other toolbox also supports `CircShiftedArray` and `fftshift`.
