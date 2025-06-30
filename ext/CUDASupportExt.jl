module CUDASupportExt
using CUDA 
using Adapt
using MutableShiftedArrays
using Base # to allow displaying such arrays without causing the single indexing CUDA error
const MutableShiftedArrayCu{N, CD} = MutableShiftedArray{<:Any,<:Any,<:Any,<:CuArray{<:Any,N,CD}}
const MutableShiftedArrayOrWrapped = Union{MutableShiftedArray,
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

####### code for CircShiftedArray

get_base_arr(arr::CuArray) = arr
get_base_arr(arr::Array) = arr
function get_base_arr(arr::AbstractArray) 
    p = parent(arr)
    return (p === arr) ? arr : get_base_arr(parent(arr))
end

# define a number of Union types to not repeat all definitions for each type
const AllShiftedType = MutableShiftedArrays.CircShiftedArray{<:Any,<:Any,<:Any}

# these are special only if a CuArray is wrapped

const AllSubArrayType = Union{SubArray{<:Any, <:Any, <:AllShiftedType, <:Any, <:Any},
                        Base.ReshapedArray{<:Any, <:Any, <:AllShiftedType, <:Any},
                        SubArray{<:Any, <:Any, <:Base.ReshapedArray{<:Any, <:Any, <:AllShiftedType, <:Any}, <:Any, <:Any}}
const AllShiftedAndViews = Union{AllShiftedType, AllSubArrayType}

const AllShiftedTypeCu{N, CD} = MutableShiftedArrays.CircShiftedArray{<:Any,<:Any,<:CuArray{<:Any,N,CD}}
const AllSubArrayTypeCu{N, CD} = Union{SubArray{<:Any, <:Any, <:AllShiftedTypeCu{N,CD}, <:Any, <:Any},
                                 Base.ReshapedArray{<:Any, <:Any, <:AllShiftedTypeCu{N,CD}, <:Any},
                                 SubArray{<:Any, <:Any, <:Base.ReshapedArray{<:Any, <:Any, <:AllShiftedTypeCu{N,CD}, <:Any}, <:Any, <:Any}}
const AllShiftedAndViewsCu{N, CD} = Union{AllShiftedTypeCu{N, CD}, AllSubArrayTypeCu{N, CD}}

Adapt.adapt_structure(to, x::MutableShiftedArrays.CircShiftedArray{T, N, S}) where {T, N, S} = MutableShiftedArrays.CircShiftedArray(adapt(to, parent(x)), MutableShiftedArrays.shifts(x));

function Base.Broadcast.BroadcastStyle(::Type{T})  where {N, CD, T<:AllShiftedTypeCu{N, CD}}
    CUDA.CuArrayStyle{N,CD}()
end

# Define the BroadcastStyle for SubArray of MutableShiftedArray with CuArray

function Base.Broadcast.BroadcastStyle(::Type{T})  where {N, CD, T<:AllSubArrayTypeCu{N, CD}}
    CUDA.CuArrayStyle{N,CD}()
end

function Base.copy(s::AllShiftedAndViews)
    res = similar(get_base_arr(s), eltype(s), size(s));
    res .= s
    return res
end

function Base.collect(x::AllShiftedAndViews) 
    return copy(x) # stay on the GPU        
end

function Base.Array(x::AllShiftedAndViews) 
    return Array(copy(x)) # remove from GPU
end

function Base.:(==)(x::AllShiftedAndViewsCu, y::AbstractArray) 
    return all(x .== y)
end

function Base.:(==)(y::AbstractArray, x::AllShiftedAndViewsCu) 
    return all(x .== y)
end

function Base.:(==)(x::AllShiftedAndViewsCu, y::AllShiftedAndViewsCu) 
    return all(x .== y)
end

function Base.isapprox(x::AllShiftedAndViewsCu, y::AbstractArray; atol=0, rtol=atol>0 ? 0 : sqrt(eps(real(eltype(x)))), va...) 
    atol = (atol != 0) ? atol : rtol * maximum(abs.(x))
    return all(abs.(x .- y) .<= atol)
end

function Base.isapprox(y::AbstractArray, x::AllShiftedAndViewsCu; atol=0, rtol=atol>0 ? 0 : sqrt(eps(real(eltype(x)))),  va...)     
    atol = (atol != 0) ? atol : rtol * maximum(abs.(x))
    return all(abs.(x .- y) .<= atol)
end

function Base.isapprox(x::AllShiftedAndViewsCu, y::AllShiftedAndViewsCu; atol=0, rtol=atol>0 ? 0 : sqrt(eps(real(eltype(x)))),  va...) # where {CT, N, CD, T<:ShiftedArrays.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    atol = (atol != 0) ? atol : rtol * maximum(abs.(x))
    return all(abs.(x .- y) .<= atol)
end

function Base.show(io::IO, mm::MIME"text/plain", cs::AllShiftedAndViews) 
    CUDA.@allowscalar invoke(Base.show, Tuple{IO, typeof(mm), AbstractArray}, io, mm, cs) 
end

end