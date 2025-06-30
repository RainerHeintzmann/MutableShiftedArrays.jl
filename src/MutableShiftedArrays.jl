module MutableShiftedArrays
# Note that the CircShiftedArray is code copied directly from ShiftedArrays.jl
# This is since the CUDASupportExt was not supported in ShiftedArrays.jl

import Base: fill!, checkbounds, getindex, setindex!, parent, size, axes, similar, copy, collect
export shifts, default, get_src_dst_ranges
export MutableShiftedArray, MutableShiftedVector
export CircShiftedArray, CircShiftedVector

include("utils.jl")
include("mutableshiftedarray.jl")
include("circshiftedarray.jl")
include("lag.jl")
include("circshift.jl")
include("fftshift.jl")

end # module MutableShiftedArrays
