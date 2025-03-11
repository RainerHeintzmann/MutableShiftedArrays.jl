using MutableShiftedArrays, Test
# using AbstractFFTs 
use_cuda = false;  # set this to true to test ShiftedArrays for the CuArray datatype
if (use_cuda)
    using CUDA
    CUDA.allowscalar(true); # needed for some of the comparisons
end

function opt_convert(v)
    if (use_cuda)
        CuArray(v)
    else
        v
    end
end

@testset "MutableShiftedVector" begin
    v = [1, 3, 5, 4]
    v = opt_convert(v);
    @test all(v .== MutableShiftedVector(v))
    sv = MutableShiftedVector(v, -1)
    @test isequal(sv, MutableShiftedVector(v, (-1,)))
    @test length(sv) == 4
    @test all(sv[1:3] .== [3, 5, 4])
    @test ismissing(sv[4])
    diff = v .- sv
    @test isequal(diff, [-2, -2, 1, missing])
    @test shifts(sv) == (-1,)
    svneg = MutableShiftedVector(v, -1, default = -100)
    @test default(svneg) == -100
    @test copy(svneg) == coalesce.(sv, -100)
    @test isequal(sv[1:3], Union{Int64, Missing}[3, 5, 4])
    svnest = MutableShiftedVector(MutableShiftedVector(v, 1), 2)
    sv = MutableShiftedVector(v, 3)
    @test sv === svnest
    sv = MutableShiftedVector(v, 2, default = nothing)
    sv1 = MutableShiftedVector(sv, 1)
    sv2 = MutableShiftedVector(sv, 1, default = 0)
    @test isequal(collect(sv1), [nothing, nothing, nothing, 1])
    @test isequal(collect(sv2), [0, nothing, nothing, 1])
end

@testset "MutableShiftedArray" begin
    v = collect(reshape(1:16, 4, 4))  # index assignment to reshaped ranges is not supported
    v = opt_convert(v);
    @test all(v .== MutableShiftedArray(v))
    sv = MutableShiftedArray(v, (-2, 0))
    @test length(sv) == 16
    @test sv[1, 3] == 11
    @test ismissing(sv[3, 3])
    @test shifts(sv) == (-2,0)
    @test isequal(sv, MutableShiftedArray(v, -2))
    @test isequal(@inferred(MutableShiftedArray(v, (2,))), @inferred(MutableShiftedArray(v, 2)))
    @test isequal(@inferred(MutableShiftedArray(v)), @inferred(MutableShiftedArray(v, (0, 0))))
    s = MutableShiftedArray(v, (0, -2))
    @test isequal(collect(s), [ 9 13 missing missing;
                               10 14 missing missing;
                               11 15 missing missing;
                               12 16 missing missing])
    sneg = MutableShiftedArray(v, (0, -2), default = -100)
    @test all(sneg .== coalesce.(s, default(sneg)))
    @test checkbounds(Bool, sv, 2, 2)
    @test !checkbounds(Bool, sv, 123, 123)
    svnest = MutableShiftedArray(MutableShiftedArray(v, (1, 1)), 2)
    sv = MutableShiftedArray(v, (3, 1))
    @test sv === svnest
    sv = MutableShiftedArray(v, 2, default = nothing)
    sv1 = MutableShiftedArray(sv, (1, 1))
    sv2 = MutableShiftedArray(sv, (1, 1), default = 0)
    @test isequal(collect(sv1), [nothing   nothing   nothing   nothing
                                 nothing   nothing   nothing   nothing
                                 nothing   nothing   nothing   nothing
                                 nothing  1         5         9      ])
    @test isequal(collect(sv2), [0  0         0         0
                                 0   nothing   nothing   nothing
                                 0   nothing   nothing   nothing
                                 0  1         5         9      ])
end

@testset "mutations" begin
    v = collect(reshape(1:16, 4, 4))  # index assignment to reshaped ranges is not supported
    v = opt_convert(v);

    sv = MutableShiftedArray(v, 2, default = nothing)
    sv1 = MutableShiftedArray(sv, (1, 1))
    sv2 = MutableShiftedArray(sv, (1, 1), default = 0)
    # test some mutation operations
    sv[1,1] = 0
    @test sv[1,1] == nothing
    sv[3,3] = 0
    @test sv[3,3] == 0
    @test v[1,3] == 0
    @test sv1[4,4] == 0
    sv1[4,4] = 55
    @test v[1,3] == 55
    sv1[:,:] .= -1
    @test v[1,1] == -1
    @test v[1,4] == 13
    sv2[:] .= -2
    @test sv2[4,2] == -2
end

@testset "mutations & size changes" begin
    v = collect(reshape(1:16, 4, 4))  # index assignment to reshaped ranges is not supported
    v = opt_convert(v);

    ns = (6, 3) # one bigger one smaller
    sv = MutableShiftedArray(v, 2, ns, default = nothing)
    sv1 = MutableShiftedArray(sv, (1, 1), ns)
    sv2 = MutableShiftedArray(sv, (1, 1), ns, default = 0)
    @test size(sv) == ns
    @test size(sv1) == ns
    @test size(sv2) == ns
    # test some mutation operations
    sv[1,1] = 0
    @test sv[1,1] == nothing
    sv[3,3] = 0
    @test sv[3,3] == 0
    @test v[1,3] == 0
    
    @test_throws BoundsError sv1[4,4] == 0
    @test_throws BoundsError sv1[4,4] = 55
    @test v[1,3] == 0
    sv1[:,:] .= -1
    @test v[1,1] == -1
    @test v[1,4] == 13
    sv2[:] .= -2
    @test sv2[4,2] == -2
end

@testset "padded_tuple" begin
    v = rand(2, 2)
    v = opt_convert(v);
    @test (1, 0) == @inferred MutableShiftedArrays.padded_tuple(v, 1)
    @test (0, 0) == @inferred MutableShiftedArrays.padded_tuple(v, ())
    @test (3, 0) == @inferred MutableShiftedArrays.padded_tuple(v, (3,))
    @test (1, 5) == @inferred MutableShiftedArrays.padded_tuple(v, (1, 5))
end

@testset "bringwithin" begin
    @test MutableShiftedArrays.bringwithin(1, 1:10) == 1   
    @test MutableShiftedArrays.bringwithin(0, 1:10) == 10   
    @test MutableShiftedArrays.bringwithin(-1, 1:10) == 9 
    
    # test to check for offset axes
    @test MutableShiftedArrays.bringwithin(5, 5:10) == 5
    @test MutableShiftedArrays.bringwithin(4, 5:10) == 10
end

@testset "laglead" begin
    v = [1, 3, 8, 12]
    v = opt_convert(v);
    diff = v .- MutableShiftedArrays.lag(v)
    @test isequal(diff, [missing, 2, 5, 4])

    diff2 = v .- MutableShiftedArrays.lag(v, 2)
    @test isequal(diff2, [missing, missing, 7, 9])

    @test all(MutableShiftedArrays.lag(v, 2, default = -100) .== coalesce.(MutableShiftedArrays.lag(v, 2), -100))

    diff = v .- MutableShiftedArrays.lead(v)
    @test isequal(diff, [-2, -5, -4, missing])

    diff2 = v .- MutableShiftedArrays.lead(v, 2)
    @test isequal(diff2, [-7, -9, missing, missing])

    @test all(MutableShiftedArrays.lead(v, 2, default = -100) .== coalesce.(MutableShiftedArrays.lead(v, 2), -100))

    @test MutableShiftedArrays.lag(MutableShiftedArrays.lag(v, 1), 2) === MutableShiftedArrays.lag(v, 3)
    @test MutableShiftedArrays.lead(MutableShiftedArrays.lead(v, 1), 2) === MutableShiftedArrays.lead(v, 3)
end
