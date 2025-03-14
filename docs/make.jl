using Documenter, MutableShiftedArrays

DocMeta.setdocmeta!(MutableShiftedArrays, :DocTestSetup, :(using MutableShiftedArrays); recursive=true)

makedocs(
    # options
    modules = [MutableShiftedArrays],
    sitename = "MutableShiftedArrays.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
    ),
    pages = Any[
        "Introduction" => "index.md",
        "API" => "api.md",
    ],
    # strict = true,
)

deploydocs(repo = "github.com/RainerHeintzmann/MutableShiftedArrays.jl.git", devbranch = "main")

# deploydocs(
#     # options
#     repo = "github.com/JuliaArrays/MutableShiftedArrays.jl.git",
#     push_preview = true,
# )
