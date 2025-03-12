
"""
    MutableShiftedArray(parent::AbstractArray, shifts = (), viewsize=size(v); default= MutableShiftedArrays.default(v))

Custom `AbstractArray` object to store an `AbstractArray` `parent` shifted by `shifts` steps
(where `shifts` is a `Tuple` with one `shift` value per dimension of `parent`).
As opposed to `ShiftedArray` of the `ShiftedArrays.jl` toolbox, this object is mutable and mutation operations in the padded ranges are ignored.
Furthermore it also supports size changes in the view.

For `s::MutableShiftedArray`, `s[i...] == s.parent[map(-, i, s.shifts)...]` if `map(-, i, s.shifts)`
is a valid index for `s.parent`, and `s.v[i, ...] == default` otherwise.
Use `copy` to collect the values of a `MutableShiftedArray` into a normal `Array`.
The recommended constructor is `MutableShiftedArray(parent, shifts; default = missing)`.

!!! note
    If `parent` is itself a `MutableShiftedArray` with a compatible default value,
    the constructor does not nest `MutableShiftedArray` objects but rather combines
    the shifts additively.

# Arguments
- `parent::AbstractArray`: the array to be shifted
- `shifts::Tuple{Int}`: the amount by which `parent` is shifted in each dimension. The default, an empty Tuple will result in no shifts.
- `viewsize::Tuple{Int}`: the size of the view. By default the size of the parent array is used.
- `default::M`: the default value to return when out of bounds in the original array. By default zero of the corresponding `eltype` is used.
                Note that using `missing` as default value will cause single index accesses in CUDA due to the Union type.

# Examples

```jldoctest shiftedarray
julia> v = [1, 3, 5, 4];

julia> s = MutableShiftedArray(v, (1,))
4-element MutableShiftedVector{Int64, Missing, Vector{Int64}}:
  missing
 1
 3
 5

julia> copy(s)
4-element Vector{Union{Missing, Int64}}:
  missing
 1
 3
 5

julia> v = reshape(1:16, 4, 4);

julia> s = MutableShiftedArray(v, (0, 2))
4×4 MutableShiftedArray{Int64, Missing, 2, Base.ReshapedArray{Int64, 2, UnitRange{Int64}, Tuple{}}}:
 missing  missing  1  5
 missing  missing  2  6
 missing  missing  3  7
 missing  missing  4  8

julia> shifts(s)
(0, 2)
```
"""
struct MutableShiftedArray{T, M, N, S<:AbstractArray} <: AbstractArray{Union{T, M}, N}
    parent::S
    shifts::NTuple{N, Int}
    viewsize::NTuple{N, Int}
    default::M
end

# low-level private constructor to handle type parameters
function mutableshiftedarray(v::AbstractArray{T, N}, shifts, viewsize, default::M) where {T, N, M}
    return MutableShiftedArray{T, M, N, typeof(v)}(v, padded_tuple(v, shifts), padded_tuple(v, viewsize), default)
end

function MutableShiftedArray(v::AbstractArray, n = (), viewsize=size(v); default = MutableShiftedArrays.default(v))
    return if (v isa MutableShiftedArray) && default === MutableShiftedArrays.default(v)
        shifts = map(+, MutableShiftedArrays.shifts(v), padded_tuple(v, n))
        mutableshiftedarray(parent(v), shifts, viewsize, default)
    else
        mutableshiftedarray(v, n, viewsize, default)
    end
end

"""
    MutableShiftedVector{T, S<:AbstractArray}

Shorthand for `MutableShiftedArray{T, 1, S}`.
"""
const MutableShiftedVector{T, M, S<:AbstractArray} = MutableShiftedArray{T, M, 1, S}

function MutableShiftedVector(v::AbstractVector, n = (), viewsize=size(v); default = MutableShiftedArrays.default(v))
    return MutableShiftedArray(v, n, viewsize; default = default)
end

size(s::MutableShiftedArray) = s.viewsize
axes(s::MutableShiftedArray) = ntuple((d) -> Base.OneTo(s.viewsize[d]), ndims(s))

# Computing a shifted index (subtracting the offset)
offset(offsets::NTuple{N,Int}, inds::NTuple{N,Int}) where {N} = map(-, inds, offsets)
# offset(offsets::NTuple{N,Int}, inds::Tuple) where {N} = map(.-, inds, offsets)
# replace_colon(s::MutableShiftedArray, t::Tuple) = ntuple((d)-> (t[d] isa Colon) ? (1:size(s)) : t[d], length(t))

# This getindex function handles the actual indexing. x can be several indices or ranges.
@inline function getindex(s::MutableShiftedArray{<:Any, <:Any, N}, x::Vararg{Int, N}) where {N}
    @boundscheck checkbounds(s, x...)
    # v, i = parent(s), offset(shifts(s), replace_colon(s, x))
    v, i = parent(s), offset(shifts(s), x)
    return if checkbounds(Bool, v, i...)
        @inbounds v[i...]
    else
        default(s)
    end
end

@inline function setindex!(s::MutableShiftedArray{<:Any, <:Any, N}, el, x::Vararg{Int, N}) where {N}
    @boundscheck checkbounds(s, x...)
    v, i = parent(s), offset(shifts(s), x)
    if checkbounds(Bool, v, i...)
        @inbounds v[i...] = el
    end
    return s
end

# function get_src_dst_ranges(src_shifts, src_size, dst_shifts, dst_size)
#     src_ranges = ntuple((d) -> max(1, -src_shifts[d]):min(dst_size[d], src_size[d] - src_shifts[d]), length(src_size))
#     dst_ranges = ntuple((d) -> max(1, src_shifts[d] + 1):min(src_size[d], src_size[d]), length(src_size))
#     return src_ranges, dst_ranges
# end

# function fill!(s::MutableShiftedArray{<:Any, <:Any, N}, x) where {N}
#     src_ranges, _ = get_src_dst_ranges(s.shifts, size(s), ntuple((d)->0, ndims(s.parent)), size(s.parent))
#     fill!((@view s.parent[src_ranges...]), x)
#     return s    
# end

# function fill!(sa::SubArray{<:Any, <:Any, <:MutableShiftedArray, <:Any, <:Any}, x) 
#     src_ranges, _ = get_src_dst_ranges(sa.parent, sa)
#     @show src_ranges
#     # fill!((@view A[src_ranges...]), x.default)
#     return sa
# end

parent(s::MutableShiftedArray) = s.parent

"""
    shifts(s::ShiftedArray)

Return amount by which `s` is shifted compared to `parent(s)`.
"""
shifts(s::MutableShiftedArray) = s.shifts

"""
    default(s::MutableShiftedArray)

Return default value.
"""
default(s::MutableShiftedArray) = s.default

default(a::AbstractArray) = zero(eltype(a))