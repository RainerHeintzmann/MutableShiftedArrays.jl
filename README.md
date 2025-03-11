# MutableShiftedArrays.jl
A lightweight toolbox representing a ShiftedArray which is mutable. The code was based on `ShiftedArrays.jl`.
Via the extension mechanism, `CUDA.jl` support is provided both for mutating and non-mutating operations.

The type `MutableShiftedArray` also supports having a modifies size of the view. This is useful for region of interest views, which can even
surpass the limit of the original array.
Mutations to elements outside the boundary of the original array are silently ignored and the `default` value is returned upon subsequent read operations.

This code and most of the documentation is based on https://github.com/JuliaArrays/ShiftedArrays.jl with contributers listed below.

The other toolbox also supports `CircShiftedArray` and `fftshift`.

## Contributors
- **Pietro Vertechi** - [piever](https://github.com/piever)
  - Contributed to the initial implementation of the `ShiftedArrays.jl` project.
- **José Bayoán Santiago Calderón** - [Nosferican](https://github.com/Nosferican)
  - Contributed to `ShiftedArrays.jl` upon which `MutableShiftedArrays.jl` is based on.
- **Felix Wechsler** - [roflmaostc](https://github.com/roflmaostc)
  - Contributed to `ShiftedArrays.jl` upon which `MutableShiftedArrays.jl` is based on.
- **Milan Bouchet-Valat** - [nalimilan](https://github.com/nalimilan)
  - Contributed to `ShiftedArrays.jl` upon which `MutableShiftedArrays.jl` is based on.
- **Felix Gauthier** - [Felix-Gauthier](https://github.com/Felix-Gauthier)
  - Contributed to `ShiftedArrays.jl` upon which `MutableShiftedArrays.jl` is based on.
- **Lilith Orion Hafner** - [LilithHafner](https://github.com/LilithHafner)
  - Contributed to `ShiftedArrays.jl` upon which `MutableShiftedArrays.jl` is based on.
- **Rainer Heintzmann** - [RainerHeintzmann](https://github.com/RainerHeintzmann)

