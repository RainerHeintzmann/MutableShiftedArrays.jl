using MutableShiftedArrays, Test
using AbstractFFTs 
# use_cuda = true;  # set this to true to test ShiftedArrays for the CuArray datatype
# if (use_cuda)
using CUDA
    # CUDA.allowscalar(true); # needed for some of the comparisons
# end

function opt_convert(v, use_cuda=false)
    if (use_cuda)
        CuArray(v)
    else
        v
    end
end

function run_all_tests(use_cuda)
@testset "MutableShiftedVector" begin
    v = [1, 3, 5, 4]
    v = opt_convert(v, use_cuda);
    @test v == MutableShiftedVector(v, default=0)
    sv = MutableShiftedVector(v, -1, default=0)
    @test isequal(sv, MutableShiftedVector(v, (-1,), default=0))
    @test length(sv) == 4
    @test sv[1:3] == opt_convert([3, 5, 4], use_cuda)
    @test sv[4] == 0
    diff = v .- sv
    @test isequal(Array(diff), [-2, -2, 1, 4])
    @test shifts(sv) == (-1,)
    svneg = MutableShiftedVector(v, -1, default = -100)
    @test default(svneg) == -100
    # @test copy(svneg) == coalesce.(sv, -100)
    @test isequal(sv[1:3], opt_convert([3, 5, 4], use_cuda))
    svnest = MutableShiftedVector(MutableShiftedVector(v, 1), 2)
    sv = MutableShiftedVector(v, 3)
    @test sv === svnest
    sv = MutableShiftedVector(v, 2, default = nothing)
    sv1 = MutableShiftedVector(sv, 1)
    sv2 = MutableShiftedVector(sv, 1, default = 0)
    @test isequal(collect(sv1), opt_convert([nothing, nothing, nothing, 1], use_cuda))
    @test isequal(collect(sv2), opt_convert([0, nothing, nothing, 1], use_cuda))
end

@testset "MutableShiftedArray" begin
    v = collect(reshape(1:16, 4, 4))  # index assignment to reshaped ranges is not supported
    v = opt_convert(v, use_cuda);
    @test v == MutableShiftedArray(v)
    sv = MutableShiftedArray(v, (-2, 0))
    @test length(sv) == 16
    @test sv[1:1, 3] == opt_convert([11], use_cuda)
    @test (sv[3, 3] == 0)
    @test shifts(sv) == (-2,0)
    @test isequal(sv, MutableShiftedArray(v, -2))
    @test isequal(@inferred(MutableShiftedArray(v, (2,))), @inferred(MutableShiftedArray(v, 2)))
    @test isequal(@inferred(MutableShiftedArray(v)), @inferred(MutableShiftedArray(v, (0, 0))))
    s = MutableShiftedArray(v, (0, -2), default=-100)
    @test isequal(collect(s), opt_convert([ 9 13 -100 -100;
                               10 14  -100 -100;
                               11 15  -100 -100;
                               12 16  -100 -100], use_cuda))
    sneg = MutableShiftedArray(v, (0, -2), default = -100)
    @test collect(sneg) == coalesce.(collect(s), default(sneg))
    @test checkbounds(Bool, sv, 2, 2)
    @test !checkbounds(Bool, sv, 123, 123)
    svnest = MutableShiftedArray(MutableShiftedArray(v, (1, 1)), 2)
    sv = MutableShiftedArray(v, (3, 1))
    @test sv === svnest
    sv = MutableShiftedArray(v, 2, default = nothing)
    sv1 = MutableShiftedArray(sv, (1, 1))
    sv2 = MutableShiftedArray(sv, (1, 1), default = 0)
    @test isequal(collect(sv1), opt_convert([nothing   nothing   nothing   nothing
                                 nothing   nothing   nothing   nothing
                                 nothing   nothing   nothing   nothing
                                 nothing  1         5         9      ], use_cuda))
    @test isequal(collect(sv2), opt_convert([0  0         0         0
                                 0   nothing   nothing   nothing
                                 0   nothing   nothing   nothing
                                 0  1         5         9      ], use_cuda))
end

@testset "mutations" begin
    v = collect(reshape(1:16, 4, 4));  # index assignment to reshaped ranges is not supported
    v = opt_convert(v, use_cuda);

    sv = MutableShiftedArray(v, 2, default = nothing)
    sv1 = MutableShiftedArray(sv, (1, 1))
    sv2 = MutableShiftedArray(sv, (1, 1))
    # test some mutation operations
    sv[1,1] = 0
    @test sv[1,1] == nothing
    CUDA.@allowscalar sv[3,3] = 0
    CUDA.@allowscalar @test sv[3,3] == 0
    CUDA.@allowscalar @test v[1,3] == 0
    CUDA.@allowscalar @test sv1[4,4] == 0
    CUDA.@allowscalar sv1[4,4] = 55
    CUDA.@allowscalar @test v[1,3] == 55
    sv1[:,:] .= -1;
    CUDA.@allowscalar @test v[1,1] == -1
    CUDA.@allowscalar @test v[1,4] == 13
    CUDA.@allowscalar sv2[:] .= -2;  # here the broadcasting still does not work, with multiple nested MutableShiftedArrays and CUDA
    CUDA.@allowscalar @test sv2[4,2] == -2
end

@testset "mutations & size changes" begin
    v = collect(reshape(1:16, 4, 4))  # index assignment to reshaped ranges is not supported
    v = opt_convert(v, use_cuda);

    ns = (6, 3) # one bigger one smaller
    sv = MutableShiftedArray(v, 2, ns, default = nothing)
    sv1 = MutableShiftedArray(sv, (1, 1), ns)
    sv2 = MutableShiftedArray(sv, (1, 1), ns, default = 0)
    @test size(sv) == ns
    @test size(sv1) == ns
    @test size(sv2) == ns
    # test some mutation operations
    CUDA.@allowscalar sv[1,1] = 0
    @test sv[1,1] == nothing
    CUDA.@allowscalar sv[3,3] = 0
    CUDA.@allowscalar @test sv[3,3] == 0
    CUDA.@allowscalar @test v[1,3] == 0
    
    @test_throws BoundsError sv1[4,4] == 0
    @test_throws BoundsError sv1[4,4] = 55
    CUDA.@allowscalar @test v[1,3] == 0
    sv1[:,:] .= -1;
    CUDA.@allowscalar @test v[1,1] == -1
    CUDA.@allowscalar @test v[1,4] == 13
    CUDA.@allowscalar sv2[:] .= -2;  # here the broadcasting still does not work, with multiple nested MutableShiftedArrays and CUDA
    CUDA.@allowscalar @test sv2[4,2] == -2
end

@testset "padded_tuple" begin
    v = rand(2, 2)
    v = opt_convert(v, use_cuda);
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
    v = opt_convert(v, use_cuda);
    diff = v .- MutableShiftedArrays.lag(v)
    @test isequal(diff, opt_convert([1, 2, 5, 4], use_cuda))

    diff2 = v .- MutableShiftedArrays.lag(v, 2, default=missing)
    @test isequal(diff2, opt_convert([missing, missing, 7, 9], use_cuda))

    @test MutableShiftedArrays.lag(v, 2, default = -100) == coalesce.(MutableShiftedArrays.lag(v, 2, default=missing), -100)

    diff = v .- MutableShiftedArrays.lead(v, default=missing)
    @test isequal(diff, opt_convert([-2, -5, -4, missing], use_cuda))

    diff2 = v .- MutableShiftedArrays.lead(v, 2, default=missing)
    @test isequal(diff2, opt_convert([-7, -9, missing, missing], use_cuda))

    @test MutableShiftedArrays.lead(v, 2, default = -100) == coalesce.(MutableShiftedArrays.lead(v, 2, default=missing), -100)

    @test MutableShiftedArrays.lag(MutableShiftedArrays.lag(v, 1), 2) === MutableShiftedArrays.lag(v, 3)
    @test MutableShiftedArrays.lead(MutableShiftedArrays.lead(v, 1), 2) === MutableShiftedArrays.lead(v, 3)
end

    @testset "CircShiftedVector" begin
        v = [1, 3, 5, 4]
        v = opt_convert(v, use_cuda);
        @test all(v .== CircShiftedVector(v))
        sv = CircShiftedVector(v, -1)
        @test isequal(sv, CircShiftedVector(v, (-1,)))
        @test length(sv) == 4
        @test all(sv .== opt_convert([3, 5, 4, 1], use_cuda))
        diff = v .- sv
        @test diff == opt_convert([-2, -2, 1, 3], use_cuda)
        @test shifts(sv) == (3,)
        sv2 = CircShiftedVector(v, 1)
        diff = v .- sv2
        @test copy(sv2) == opt_convert([4, 1, 3, 5], use_cuda)
        @test all(CircShiftedVector(v, 1) .== circshift(v, 1))
        CUDA.@allowscalar sv[2] = 0
        @test collect(sv) == opt_convert([3, 0, 4, 1], use_cuda)
        @test v == opt_convert([1, 3, 0, 4], use_cuda)
        CUDA.@allowscalar sv[3] = 12 
        @test collect(sv) == opt_convert([3, 0, 12, 1], use_cuda)
        @test v == opt_convert([1, 3, 0, 12], use_cuda)
        CUDA.@allowscalar @test sv === setindex!(sv, 12, 3) 
        @test checkbounds(Bool, sv, 2)
        @test !checkbounds(Bool, sv, 123)
        sv = CircShiftedArray(v, 3)
        svnest = CircShiftedArray(CircShiftedArray(v, 2), 1)
        @test sv === svnest
    end
    
    @testset "CircShiftedArray" begin
        v = reshape(1:16, 4, 4)
        v = opt_convert(v, use_cuda);
        @test all(v .== CircShiftedArray(v))
        sv = CircShiftedArray(v, (-2, 0))
        @test length(sv) == 16
        CUDA.@allowscalar @test sv[1, 3] == 11
        @test shifts(sv) == (2, 0)
        @test isequal(sv, CircShiftedArray(v, -2))
        @test isequal(@inferred(CircShiftedArray(v, 2)), @inferred(CircShiftedArray(v, (2,))))
        @test isequal(@inferred(CircShiftedArray(v)), @inferred(CircShiftedArray(v, (0, 0))))
        s = CircShiftedArray(v, (0, 2))
        @test isequal(collect(s), opt_convert([ 9 13 1 5;
                                   10 14 2 6;
                                   11 15 3 7;
                                   12 16 4 8], use_cuda))
        sv = CircShiftedArray(v, 3)
        svnest = CircShiftedArray(CircShiftedArray(v, 2), 1)
        @test sv === svnest
    end
    
    @testset "circshift" begin
        v = reshape(1:16, 4, 4)
        v = opt_convert(v, use_cuda);
        @test all(circshift(v, (1, -1)) .== MutableShiftedArrays.circshift(v, (1, -1)))
        @test all(circshift(v, (1,)) .== MutableShiftedArrays.circshift(v, (1,)))
        @test all(circshift(v, 3) .== MutableShiftedArrays.circshift(v, 3))
        sv = MutableShiftedArrays.circshift(v, 3)
        svnest = MutableShiftedArrays.circshift(MutableShiftedArrays.circshift(v, 2), 1)
        @test sv === svnest
    end
    
    @testset "fftshift and ifftshift" begin
        function test_fftshift(x, dims=1:ndims(x))
            x = opt_convert(x, use_cuda)
            @test fftshift(x, dims) == MutableShiftedArrays.fftshift(x, dims)
            @test ifftshift(x, dims) == MutableShiftedArrays.ifftshift(x, dims)
        end
    
        test_fftshift(randn((10,)))
        test_fftshift(randn((11,)))
        test_fftshift(randn((10,)), (1,))
        test_fftshift(randn(ComplexF32, (11,)), (1,))
        test_fftshift(randn((10, 11)), (1,))
        test_fftshift(randn((10, 11)), (2,))
        test_fftshift(randn(ComplexF32,(10, 11)), (1, 2))
        test_fftshift(randn((10, 11)))
    
        test_fftshift(randn((10, 11, 12, 13)), (2, 4))
        test_fftshift(randn((10, 11, 12, 13)), (5))
        test_fftshift(randn((10, 11, 12, 13)))
    
        @test (2, 2, 0) == MutableShiftedArrays.ft_center_diff((4, 5, 6), (1, 2)) # Fourier center is at (2, 3, 0)
        @test (2, 2, 3) == MutableShiftedArrays.ft_center_diff((4, 5, 6), (1, 2, 3)) # Fourier center is at (2, 3, 4)
    end
    @testset "similar" begin
        x = opt_convert(rand(10,11), use_cuda)
        @test typeof(x) == typeof(similar(CircShiftedArray(x, (3,4))))
        @test typeof(x) == typeof(similar(MutableShiftedArray(x, (3,4))))
    end
end

run_all_tests(false)
if CUDA.functional()
    @testset "all in CUDA" begin
    run_all_tests(true)

        # some extra tests to check for indexing with integers
        v = rand(10,11)
        sv = MutableShiftedArray(cu(rand(10,11)), (3,4))
        @test_throws ErrorException sv[5,6] 
        @test (CUDA.@allowscalar sv[5,6]) == Array(sv)[5,6]
        @test_throws ErrorException sv[47] 
        @test_throws BoundsError sv[1,2,3] 
        @test (CUDA.@allowscalar sv[47]) == Array(sv)[47]
    end
else
    @testset "no CUDA available!" begin
        @test true == true
    end
end
