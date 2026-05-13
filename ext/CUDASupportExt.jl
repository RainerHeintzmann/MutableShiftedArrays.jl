module CUDASupportExt
using CUDA
using Adapt
using MutableShiftedArrays
using Base # to allow displaying such arrays without causing the single indexing CUDA error

# define general types governing both cuda versions of, MutableShiftedArray and CircShiftedArray
const MCShiftedArrayCu{N, CD} = Union{MutableShiftedArray{<:Any,<:Any,<:Any,<:CuArray{<:Any,N,CD}}, MutableShiftedArrays.CircShiftedArray{<:Any,<:Any,<:CuArray{<:Any,N,CD}}}
const MCShiftedArrayOrWrapped = Union{MCShiftedArrayCu,
                                    Base.ReshapedArray{<:Any, <:Any, <:MCShiftedArrayCu},
                                    SubArray{<:Any, <:Any, <:MCShiftedArrayCu, <:Any, <:Any}}

get_base_arr(arr::CuArray) = arr
get_base_arr(arr::Array) = arr
function get_base_arr(arr::AbstractArray) 
    p = parent(arr)
    return (p === arr) ? arr : get_base_arr(parent(arr))
end

# lets do this for the MutableShiftedArray type
Adapt.adapt_structure(to, x::MutableShiftedArray) = MutableShiftedArray(adapt(to, parent(x)), shifts(x), size(x); default=MutableShiftedArrays.default(x));
Adapt.adapt_structure(to, x::MutableShiftedArrays.CircShiftedArray{T, N, S}) where {T, N, S} = MutableShiftedArrays.CircShiftedArray(adapt(to, parent(x)), MutableShiftedArrays.shifts(x));

# suggestions by vchuravy (https://github.com/JuliaGPU/CUDA.jl/issues/2735):
Adapt.parent_type(::Type{MutableShiftedArray{_A,_B,_C,AA}}) where {_A,_B,_C,AA} = AA
Adapt.parent_type(::Type{MutableShiftedArrays.CircShiftedArray{_T,_N, AA}}) where {_T,_N,AA} = AA
Adapt.unwrap_type(W::Type{<:MutableShiftedArray}) = unwrap_type(parent_type(W))
Adapt.unwrap_type(W::Type{<:MutableShiftedArrays.CircShiftedArray}) = unwrap_type(parent_type(W))

function Base.Broadcast.BroadcastStyle(W::Type{<:MCShiftedArrayOrWrapped}) 
    return Base.Broadcast.BroadcastStyle(unwrap_type(W))
end

function Base.show(io::IO, mm::MIME"text/plain", cs::MCShiftedArrayCu) 
    # @show "showing:"
    CUDA.@allowscalar invoke(Base.show, Tuple{IO, typeof(mm), AbstractArray}, io, mm, cs) 
end


# Unified copy/collect/Array for both MutableShiftedArray and CircShiftedArray on GPU
function Base.copy(s::MCShiftedArrayCu)
    res = similar(get_base_arr(s), eltype(s), size(s))
    res .= s
    return res
end

function Base.collect(x::MCShiftedArrayCu)
    return copy(x) # stay on the GPU
end

function Base.Array(x::MCShiftedArrayCu)
    return Array(copy(x)) # move to CPU
end

function Base.:(==)(x::T, y::AbstractArray)  where {N, CD, T<:MCShiftedArrayCu{N,CD}}    
    return all(x .== y)
end

function Base.:(==)(y::AbstractArray, x::T)  where {N, CD, T<:MCShiftedArrayCu{N,CD}}    
    return all(x .== y)
end

function Base.:(==)(x::T, y::T)  where {N, CD, T<:MCShiftedArrayCu{N,CD}}    
    return all(x .== y)
end

function Base.isapprox(x::T, y::AbstractArray; kwargs...) where {N, CD, T<:MCShiftedArrayCu{N,CD}}
    return isapprox(collect(x), y; kwargs...)
end

function Base.isapprox(x::AbstractArray, y::T; kwargs...) where {N, CD, T<:MCShiftedArrayCu{N,CD}}
    return isapprox(x, collect(y); kwargs...)
end

function Base.isapprox(x::T, y::T; kwargs...) where {N, CD, T<:MCShiftedArrayCu{N,CD}}
    return isapprox(collect(x), collect(y); kwargs...)
end

_all_dims(arr) = ntuple(identity, ndims(arr))
_to_scalar(x) = only(Array(x))

# This is necessary, since sum(mutableshiftedarray) did throw an error
Base.reduce(op, arr::MCShiftedArrayOrWrapped) = _to_scalar(reduce(op, arr; dims=_all_dims(arr)))
Base.mapreduce(f, op, arr::MCShiftedArrayOrWrapped) = _to_scalar(mapreduce(f, op, arr; dims=_all_dims(arr)))

end