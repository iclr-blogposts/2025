using DifferentiationInterface
using SparseConnectivityTracer, SparseMatrixColorings
using ForwardDiff: ForwardDiff

function iter_diff(x, k)
    if k == 0
        return x
    else
        y = iter_diff(x, k - 1)
        return diff(y)
    end
end

iter_diff([1, 4, 9, 16], 1)
iter_diff([1, 4, 9, 16], 2)
iter_diff([1, 4, 9, 16], 3)
iter_diff([1, 4, 9, 16], 4)

ad = AutoForwardDiff()

sparsity_detector = TracerSparsityDetector()
coloring_algorithm = GreedyColoringAlgorithm()
asd = AutoSparse(ad; sparsity_detector, coloring_algorithm)

x, k = rand(10), 3;
jacobian(iter_diff, ad, x, Constant(k))
jacobian(iter_diff, asd, x, Constant(k))

prep = prepare_jacobian(iter_diff, asd, x, Constant(k));

jacobian(iter_diff, prep, asd, x, Constant(k))

ncolors(prep)
sparsity_pattern(prep)

column_colors(prep)

using DifferentiationInterfaceTest

scen = Scenario{:jacobian,:out}(iter_diff, rand(1000); contexts=(Constant(10),))
data = benchmark_differentiation([dense_backend, sparse_backend], [scen]; benchmark=:full)
