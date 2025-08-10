
| **Documentation**                       | **Build Status**                          | **Code Coverage**               |
|:---------------------------------------:|:-----------------------------------------:|:-------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][CI-img]][CI-url] | [![][codecov-img]][codecov-url] |

# MutableShiftedArrays.jl
A lightweight toolbox representing a ShiftedArray which is mutable. The code was based on [`ShiftedArrays.jl`](https://github.com/JuliaArrays/ShiftedArrays.jl) by Pietro Vertechi et al..
Via the extension mechanism, `CUDA.jl` support is provided both for mutating and non-mutating operations for the `MutableShiftedArray` type as well as `CircShiftedArray`.

The type `MutableShiftedArray` also supports having a modifies size of the view. This is useful for region of interest views, which can even
surpass the limit of the original array.
Mutations to elements outside the boundary of the original array are silently ignored and the `default` value is returned upon subsequent read operations.

This code and most of the documentation is based on https://github.com/JuliaArrays/ShiftedArrays.jl with contributers listed below.

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

## Development
Feel free to file an issue regarding problems, suggestions or improvement ideas for this package!

[docs-dev-img]: https://img.shields.io/badge/docs-dev-pink.svg
[docs-dev-url]: https://rainerheintzmann.github.io/MutableShiftedArrays.jl/dev/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-darkgreen.svg
[docs-stable-url]: https://rainerheintzmann.github.io/MutableShiftedArrays.jl/stable/

[CI-img]: https://github.com/rainerheintzmann/MutableShiftedArrays.jl/actions/workflows/ci.yml/badge.svg
[CI-url]: https://github.com/rainerheintzmann/MutableShiftedArrays.jl/actions/workflows/ci.yml

[codecov-img]: https://codecov.io/gh/rainerheintzmann/MutableShiftedArrays.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/rainerheintzmann/MutableShiftedArrays.jl