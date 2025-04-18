# MutableShiftedArrays

Implementation of shifted arrays just like `ShiftedArrays.jl` but supporting mutation.

## MutableShifted Arrays

A `MutableShiftedArray` is a lazy view of an Array, shifted on some or all of its indexing dimensions by some constant values.

```julia
julia> v = reshape(1:16, 4, 4)
4×4 reshape(::UnitRange{Int64}, 4, 4) with eltype Int64:
 1  5   9  13
 2  6  10  14
 3  7  11  15
 4  8  12  16

julia> s = MutableShiftedArray(v, (2, 0))
4×4 MutableShiftedArray{Int64, Missing, 2, Base.ReshapedArray{Int64, 2, UnitRange{Int64}, Tuple{}}}:
  missing   missing    missing    missing
  missing   missing    missing    missing
 1         5          9         13
 2         6         10         14 
```

The parent Array as well as the amount of shifting can be recovered with `parent` and `shifts` respectively.

```julia
julia> parent(s)
4×4 reshape(::UnitRange{Int64}, 4, 4) with eltype Int64:
 1  5   9  13
 2  6  10  14
 3  7  11  15
 4  8  12  16

julia> shifts(s)
(2, 0)
```

`shifts` returns a `Tuple`, where the n-th element corresponds to the shift on the n-th dimension of the parent `Array`.

Use `copy` to collect the shifted data into an `Array`:

```julia
julia> copy(s)
4×4 Matrix{Union{Missing, Int64}}:
  missing   missing    missing    missing
  missing   missing    missing    missing
 1         5          9         13
 2         6         10         14   
```

If you pass an integer, it will shift in the first dimension:

```julia
julia> MutableShiftedArray(v, 1)
4×4 MutableShiftedArray{Int64, Missing, 2, Base.ReshapedArray{Int64, 2, UnitRange{Int64}, Tuple{}}}:
  missing   missing    missing    missing
 1         5          9         13
 2         6         10         14
 3         7         11         15
```

A custom default value (other than `missing`) can be provided with the `default` keyword:

```julia
julia> MutableShiftedArray([1.2, 3.1, 4.5], 1, default = NaN)
3-element ShiftedVector{Float64, Float64, Vector{Float64}}:
 NaN
   1.2
   3.1
```

### Out of bound indexes

Accessing indexes outside the `MutableShiftedArray` give a `BoundsError`, even if the shifted index would have been valid in the parent array.

```julia
julia> MutableShiftedArray([1, 2, 3], 1)[4]
ERROR: BoundsError: attempt to access 3-element ShiftedVector{Int64, Missing, Vector{Int64}} at index [4]
```

## Shifting the data

Using the `MutableShiftedArray` type, this package provides two operations for lazily shifting vectors: `lag` and `lead`.

```julia
julia> v = [1, 3, 5, 4];

julia> MutableShiftedArrays.lag(v)
4-element ShiftedVector{Int64, Missing, Vector{Int64}}:
  missing
 1
 3
 5       

julia> v .- MutableShiftedArrays.lag(v) # compute difference from previous element without unnecessary allocations
4-element Vector{Union{Missing, Int64}}:
   missing
  2
  2
 -1       

julia> s = MutableShiftedArrays.lag(v, 2) # shift by more than one element
4-element MutableShiftedVector{Int64, Missing, Vector{Int64}}:
  missing
  missing
 1
 3
```

`lead` is the analogous of `lag` but shifts in the opposite direction:

```julia
julia> v = [1, 3, 5, 4];

julia> MutableShiftedArrays.lead(v)
4-element ShiftedVector{Int64, Missing, Vector{Int64}}:
 3
 5
 4
  missing
```
