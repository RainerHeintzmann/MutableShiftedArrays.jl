module MutableShiftedArrays

import Base: fill!, checkbounds, getindex, setindex!, parent, size, axes, similar, copy, collect
export shifts, default, get_src_dst_ranges
export MutableShiftedArray, MutableShiftedVector

include("utils.jl")
include("mutableshiftedarray.jl")
include("lag.jl")

end # module MutableShiftedArrays
