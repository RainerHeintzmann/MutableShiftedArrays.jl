module CUDASupportExt
using CUDA 
using Adapt
using MutableShiftedArrays
using Base # to allow displaying such arrays without causing the single indexing CUDA error

# Adapt.adapt_structure(to, x::CircShiftedArray{T, D}) where {T, D} = CircShiftedArray(adapt(to, parent(x)), shifts(x));
# parent_type(::Type{CircShiftedArray{T, N, S}})  where {T, N, S} = S
# Base.Broadcast.BroadcastStyle(::Type{T})  where {T<:CircShiftedArray} = Base.Broadcast.BroadcastStyle(parent_type(T))

# cu_storage_type(::Type{T}) where {CT,CN,CD,T<:MutableShiftedArray{<:Any,<:Any,<:Any,<:CuArray{CT,CN,CD}}} = CD

# lets do this for the MutableShiftedArray type
# Adapt.adapt_structure(to, x::MutableShiftedArray{T, M, N, S}) where {T, M, N, S} = MutableShiftedArray(adapt(to, parent(x)), shifts(x), size(x); default=MutableShiftedArrays.default(x));
Adapt.adapt_structure(to, x::MutableShiftedArray) = MutableShiftedArray(adapt(to, parent(x)), shifts(x), size(x); default=MutableShiftedArrays.default(x));

# function Base.Broadcast.BroadcastStyle(::Type{MutableShiftedArray{<:Any,<:Any,<:Any, <:CuArray{T,N,CD}}}) where {T,N,CD}
# function Base.Broadcast.BroadcastStyle(::Type{MutableShiftedArray{<:Any,<:Any,<:Any, <:CuArray{T,N,CD}}}) where {T,N,CD}
# function Base.Broadcast.BroadcastStyle(::Type{MutableShiftedArray{<:Any,<:Any,<:Any, <:CuArray{T,N}}}) where {T,N}
function Base.Broadcast.BroadcastStyle(::Type{T})  where {CT, N, CD, T<:MutableShiftedArray{<:Any,<:Any,<:Any,<:CuArray{CT,N,CD}}}
    CUDA.CuArrayStyle{N,CD}()
end

# Define the BroadcastStyle for SubArray of MutableShiftedArray with CuArray
function Base.Broadcast.BroadcastStyle(::Type{T})  where {CT, N, CD, T<:SubArray{<:Any, <:Any, <:MutableShiftedArray{<:Any,<:Any,<:Any,<:CuArray{CT,N,CD}}}}
    CUDA.CuArrayStyle{N,CD}()
end

# Define the BroadcastStyle for ReshapedArray of MutableShiftedArray with CuArray
function Base.Broadcast.BroadcastStyle(::Type{T})  where {CT, N, CD, T<:Base.ReshapedArray{<:Any, <:Any, <:MutableShiftedArray{<:Any,<:Any,<:Any,<:CuArray{CT,N,CD}}, <:Any}}
    CUDA.CuArrayStyle{N,CD}()
end

# Define the BroadcastStyle for SubArray of a ReshapedArray of MutableShiftedArray with CuArray
# function Base.Broadcast.BroadcastStyle(::Type{T})  where {MT, T<:MutableShiftedArray{<:Any,<:Any,<:Any,<:MutableShiftedArray{<:Any,<:Any,<:Any,<:MT}}}
#     Base.Broadcast.BroadcastStyle(MT)
# end


function Base.show(io::IO, mm::MIME"text/plain", cs::MutableShiftedArray) 
    # @show "showing:"
    CUDA.@allowscalar invoke(Base.show, Tuple{IO, typeof(mm), AbstractArray}, io, mm, cs) 
end

function Base.collect(x::T)  where {CT, N, CD, T<:MutableShiftedArray{<:Any,<:Any,<:Any,<:CuArray{CT,N,CD}}}
    return copy(x) # stay on the GPU        
end

function Base.Array(x::T)  where {CT, N, CD, T<:MutableShiftedArray{<:Any,<:Any,<:Any,<:CuArray{CT,N,CD}}}
    return Array(copy(x)) # stay on the GPU        
end

function Base.:(==)(x::T, y)  where {CT, N, CD, T<:MutableShiftedArray{<:Any,<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    return all(x .== y)
end

function Base.:(==)(x::T, y::AbstractArray)  where {CT, N, CD, T<:MutableShiftedArray{<:Any,<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    return all(x .== y)
end

function Base.:(==)(y, x::T)  where {CT, N, CD, T<:MutableShiftedArray{<:Any,<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    return all(x .== y)
end

function Base.:(==)(y::AbstractArray, x::T)  where {CT, N, CD, T<:MutableShiftedArray{<:Any,<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    return all(x .== y)
end

function Base.:(==)(x::T, y::T)  where {CT, N, CD, T<:MutableShiftedArray{<:Any,<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    return all(x .== y)
end

end