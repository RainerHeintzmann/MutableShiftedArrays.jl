module CUDASupportExt
using CUDA 
using Adapt
using MutableShiftedArrays
using Base # to allow displaying such arrays without causing the single indexing CUDA error
MutableShiftedArrayCu{N, CD} = MutableShiftedArray{<:Any,<:Any,<:Any,<:CuArray{<:Any,N,CD}}
MutableShiftedArrayOrWrapped = Union{MutableShiftedArray,
                                    Base.ReshapedArray{<:Any, <:Any, <:MutableShiftedArray},
                                    SubArray{<:Any, <:Any, <:MutableShiftedArray, <:Any, <:Any}}

# lets do this for the MutableShiftedArray type
Adapt.adapt_structure(to, x::MutableShiftedArray) = MutableShiftedArray(adapt(to, parent(x)), shifts(x), size(x); default=MutableShiftedArrays.default(x));
# suggestions by vchuravy (https://github.com/JuliaGPU/CUDA.jl/issues/2735):
Adapt.parent_type(::Type{MutableShiftedArray{_A,_B,_C,AA}}) where {_A,_B,_C,AA} = AA
Adapt.unwrap_type(W::Type{<:MutableShiftedArray}) = unwrap_type(parent_type(W))

# function Base.Broadcast.BroadcastStyle(::Type{T})  where {N, CD, T<:MutableShiftedArraySubCu{N, CD}}
function Base.Broadcast.BroadcastStyle(W::Type{<:MutableShiftedArrayOrWrapped})
    return Base.Broadcast.BroadcastStyle(unwrap_type(W))
end

function Base.show(io::IO, mm::MIME"text/plain", cs::MutableShiftedArrayCu) 
    # @show "showing:"
    CUDA.@allowscalar invoke(Base.show, Tuple{IO, typeof(mm), AbstractArray}, io, mm, cs) 
end

function Base.collect(x::T)  where {N, CD, T<:MutableShiftedArrayCu{N,CD}}
    return copy(x) # stay on the GPU        
end

function Base.Array(x::T)  where {N, CD, T<:MutableShiftedArrayCu{N,CD}}
    return Array(copy(x)) # stay on the GPU        
end

function Base.:(==)(x::T, y::AbstractArray)  where {N, CD, T<:MutableShiftedArrayCu{N,CD}}    
    return all(x .== y)
end

function Base.:(==)(y::AbstractArray, x::T)  where {N, CD, T<:MutableShiftedArrayCu{N,CD}}    
    return all(x .== y)
end

function Base.:(==)(x::T, y::T)  where {N, CD, T<:MutableShiftedArrayCu{N,CD}}    
    return all(x .== y)
end

end