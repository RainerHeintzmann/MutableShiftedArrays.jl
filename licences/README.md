# Licences
This toolbox builds on [`ShiftedArrays.jl`](https://github.com/JuliaArrays/ShiftedArrays.jl) and is largely compatible with it. However, the `MutableShiftedArray` type supports mutating the lazily shifted array
The `CircShiftedArray` type is currentyl identical to the one present in `ShiftedArrays.jl`, but extended (see `ext` folder) by supporting also the `CuArray` type as defined in the `CUDA.jl` toolbox.

For completeness this `licence` folder contains the original licence of `ShiftedArrays.jl`, which applies to the code in `circshift.jl`.

