module MutableShiftedArrays

import Base: checkbounds, getindex, setindex!, parent, size, axes
export shifts, default
export MutableShiftedArray, MutableShiftedVector

include("utils.jl")
include("mutableshiftedarray.jl")
include("lag.jl")

end # module MutableShiftedArrays
