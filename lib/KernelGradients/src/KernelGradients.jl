module KernelGradients

import KernelAbstractions: Kernel, CPUCTX, CPUCtx
import Cassette
import Enzyme
using Requires

@inline function Cassette.overdub(::CPUCtx, ::typeof(Enzyme.autodiff_no_cassette), f, args...)
    f′ = (args...) -> (Base.@_inline_meta; Cassette.overdub(CPUCTX, f, args...))
    Enzyme.autodiff_no_cassette(f′, args...)
end

function Enzyme.autodiff(kernel::Kernel{<:Any, <:Any, <:Any, Fun}) where Fun
    function df(ctx, args...)
        Enzyme.autodiff_no_cassette(kernel.f, ctx, args...)
    end
    similar(kernel, df)
end

function __init__()
    @require CUDAKernels="72cfdca4-0801-4ab0-bf6a-d52aa10adc57" include("cuda.jl")
    @require ROCKernels="7eb9e9f0-4bd3-4c4c-8bef-26bd9629d9b9" include("roc.jl")
end

end # module
