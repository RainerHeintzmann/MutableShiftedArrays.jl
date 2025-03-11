"""
    lag(v::AbstractArray, n = 1; default = missing)

Return a `MutableShiftedArray` object which lazily represents the array `v` shifted
by `n` (an `Integer` or a `Tuple` of `Integer`s).
If the number of dimensions of `v` exceeds the length of `n`, the shift in the
remaining dimensions is assumed to be `0`.
`default` specifies a default value to return when out of bounds in the original array.

# Examples

```jldoctest lag
julia> v = [1, 3, 5, 4];

julia> MutableShiftedArray.lag(v)
4-element ShiftedVector{Int64, Missing, Vector{Int64}}:
  missing
 1
 3
 5

julia> w = 1:2:9
1:2:9

julia> s = MutableShiftedArray.lag(w, 2)
5-element ShiftedVector{Int64, Missing, StepRange{Int64, Int64}}:
  missing
  missing
 1
 3
 5

julia> copy(s)
5-element Vector{Union{Missing, Int64}}:
  missing
  missing
 1
 3
 5

julia> v = reshape(1:16, 4, 4);

julia> s = MutableShiftedArray.lag(v, (0, 2))
4×4 MutableShiftedArray{Int64, Missing, 2, Base.ReshapedArray{Int64, 2, UnitRange{Int64}, Tuple{}}}:
 missing  missing  1  5
 missing  missing  2  6
 missing  missing  3  7
 missing  missing  4  8
```
"""
function lag(v::AbstractArray, n = 1; default = MutableShiftedArrays.default(v))
    return MutableShiftedArray(v, n; default = default)
end

"""
    lead(v::AbstractArray, n = 1; default = missing)

Return a `MutableShiftedArray` object which lazily represents the array `v` shifted
negatively by `n` (an `Integer` or a `Tuple` of `Integer`s).
If the number of dimensions of `v` exceeds the length of `n`, the shift in the
remaining dimensions is assumed to be `0`.
`default` specifies a default value to return when out of bounds in the original array.

# Examples

```jldoctest lead
julia> v = [1, 3, 5, 4];

julia> MutableShiftedArray.lead(v)
4-element ShiftedVector{Int64, Missing, Vector{Int64}}:
 3
 5
 4
  missing

julia> w = 1:2:9
1:2:9

julia> s = MutableShiftedArray.lead(w, 2)
5-element ShiftedVector{Int64, Missing, StepRange{Int64, Int64}}:
 5
 7
 9
  missing
  missing

julia> copy(s)
5-element Vector{Union{Missing, Int64}}:
 5
 7
 9
  missing
  missing

julia> v = reshape(1:16, 4, 4);

julia> s = MutableShiftedArray.lead(v, (0, 2))
4×4 MutableShiftedArray{Int64, Missing, 2, Base.ReshapedArray{Int64, 2, UnitRange{Int64}, Tuple{}}}:
  9  13  missing  missing
 10  14  missing  missing
 11  15  missing  missing
 12  16  missing  missing
```
"""
function lead(v::AbstractArray, n = 1; default = MutableShiftedArrays.default(v))
    return MutableShiftedArray(v, map(-, n); default = default)
end