"""
    ft_center_diff(s [, dims])

Return the shifts required to center dimensions `dims` at the respective
Fourier centers.
This function is internally used by [`MutableShiftedArrays.fftshift`](@ref) and
[`MutableShiftedArrays.ifftshift`](@ref).

# Examples

```jldoctest
julia> MutableShiftedArrays.ft_center_diff((4, 5, 6), (1, 2)) # Fourier center is at (2, 3, 0)
(2, 2, 0)

julia> MutableShiftedArrays.ft_center_diff((4, 5, 6), (1, 2, 3)) # Fourier center is at (2, 3, 4)
(2, 2, 3)
```
"""
function ft_center_diff(s::NTuple{N, T}, dims=ntuple(identity, Val(N))) where {N, T}
    return ntuple(i -> i ∈ dims ?  s[i] ÷ 2 : 0, N)
end

"""
    fftshift(x [, dims])

Lazy version of `AbstractFFTs.fftshift(x, dims)`. Return a `CircShiftedArray`
where each given dimension is shifted by `N÷2`, where `N` is the size of
that dimension.

# Examples

```jldoctest
julia> MutableShiftedArrays.fftshift([1 0 0 0])
1×4 CircShiftedArray{Int64, 2, Matrix{Int64}}:
 0  0  1  0

julia> MutableShiftedArrays.fftshift([1 0 0; 0 0 0; 0 0 0])
3×3 CircShiftedArray{Int64, 2, Matrix{Int64}}:
 0  0  0
 0  1  0
 0  0  0

julia> MutableShiftedArrays.fftshift([1 0 0; 0 0 0; 0 0 0], (1,))
3×3 CircShiftedArray{Int64, 2, Matrix{Int64}}:
 0  0  0
 1  0  0
 0  0  0
```
"""
function fftshift(x::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    return MutableShiftedArrays.circshift(x, ft_center_diff(size(x), dims))
end

"""
    ifftshift(x [, dims])

Lazy version of `AbstractFFTs.ifftshift(x, dims)`. Return a `CircShiftedArray`
where each given dimension is shifted by `-N÷2`, where `N` is the size of
that dimension.

# Examples

```jldoctest
julia> MutableShiftedArrays.ifftshift([0 0 1 0])
1×4 CircShiftedArray{Int64, 2, Matrix{Int64}}:
 1  0  0  0

julia> MutableShiftedArrays.ifftshift([0 0 0; 0 1 0; 0 0 0])
3×3 CircShiftedArray{Int64, 2, Matrix{Int64}}:
 1  0  0
 0  0  0
 0  0  0

julia> MutableShiftedArrays.ifftshift([0 1 0; 0 0 0; 0 0 0], (2,))
3×3 CircShiftedArray{Int64, 2, Matrix{Int64}}:
 1  0  0
 0  0  0
 0  0  0
```
"""
function ifftshift(x::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    return MutableShiftedArrays.circshift(x, map(-, ft_center_diff(size(x), dims)))
end
