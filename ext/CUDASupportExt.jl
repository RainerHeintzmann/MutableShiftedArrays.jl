module CUDASupportExt
using CUDA 
using Adapt
using MutableShiftedArrays
using Base # to allow displaying such arrays without causing the single indexing CUDA error

# cu_storage_type(::Type{T}) where {CT,CN,CD,T<:CuArray{CT,CN,CD}} = CD
# lets do this for the ShiftedArray type
Adapt.adapt_structure(to, x::MutableShiftedArray{T, M, N}) where {T, M, N} = MutableShiftedArray(adapt(to, parent(x)), shifts(x); default=MutableShiftedArrays.default(x));

# function Base.Broadcast.BroadcastStyle(::Type{T})  where (CT,CN,CD,T<: ShiftedArray{<:Any,<:Any,<:Any,<:CuArray})
function Base.Broadcast.BroadcastStyle(::Type{MutableShiftedArray{<:Any,<:Any,<:Any, <:CuArray{T,N,CD}}}) where {T,N,CD}
    CUDA.CuArrayStyle{N,CD}()
end

# function Base.show(io::IO, mm::MIME"text/plain", cs::CircShiftedArray) 
#     CUDA.@allowscalar invoke(Base.show, Tuple{IO, typeof(mm), AbstractArray}, io, mm, cs) 
# end

# function Base.show(io::IO, mm::MIME"text/plain", cs::ShiftedArray) 
#     CUDA.@allowscalar invoke(Base.show, Tuple{IO, typeof(mm), AbstractArray}, io, mm, cs) 
# end
end