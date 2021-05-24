import .CUDAKernels: CUDACtx, CUDACTX
import Cassette
import Enzyme

@inline function Cassette.overdub(::CUDACtx, ::typeof(Enzyme.autodiff_no_cassette), f, args...)
    f′ = (args...) -> (Base.@_inline_meta; Cassette.overdub(CUDACTX, f, args...))
    Enzyme.autodiff_no_cassette(f′, args...)
end