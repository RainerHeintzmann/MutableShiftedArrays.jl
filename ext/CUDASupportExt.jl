module CUDASupportExt
using CUDA 
using Adapt
using MutableShiftedArrays
using Base # to allow displaying such arrays without causing the single indexing CUDA error
MutableShiftedArrayCu{N, CD} = MutableShiftedArray{<:Any,<:Any,<:Any,<:CuArray{<:Any,N,CD}}
MutableShiftedArraySubCu{N, CD} = Union{SubArray{<:Any, <:Any, <:MutableShiftedArrayCu{N, CD}},
                                    Base.ReshapedArray{<:Any, <:Any, <:MutableShiftedArrayCu{N, CD}},
                                    SubArray{<:Any, <:Any, <:Base.ReshapedArray{<:Any, <:Any, <:MutableShiftedArrayCu{N, CD}}, <:Any, <:Any}}

# lets do this for the MutableShiftedArray type
Adapt.adapt_structure(to, x::MutableShiftedArray) = MutableShiftedArray(adapt(to, parent(x)), shifts(x), size(x); default=MutableShiftedArrays.default(x));

function Base.Broadcast.BroadcastStyle(::Type{T})  where {N, CD, T<:MutableShiftedArrayCu{N, CD}}
    CUDA.CuArrayStyle{N,CD}()
end

# Define the BroadcastStyle for SubArray of MutableShiftedArray with CuArray
function Base.Broadcast.BroadcastStyle(::Type{T})  where {N, CD, T<:MutableShiftedArraySubCu{N, CD}}
    CUDA.CuArrayStyle{N,CD}()
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