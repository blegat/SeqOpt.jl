module TestMOI

import SeqOpt
import HiGHS
using Test

import MathOptInterface as MOI

# See the docstring of MOI.Test.Config for other arguments.
const CONFIG = MOI.Test.Config(
    atol = 1e-6,
    rtol = 1e-6,
    optimal_status = MOI.LOCALLY_SOLVED,
    exclude = Any[MOI.VariableName, MOI.delete],
)

"""
    runtests()

This function runs all functions in the this Module starting with `test_`.
"""
function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
end

# Test from https://github.com/jump-dev/MathOptInterface.jl/pull/2059
function test_PR2059()
    lp = MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)
    model = SeqOpt.Optimizer(lp)
    x = MOI.add_variable(model)
    con_f = MOI.ScalarNonlinearFunction(:^, Any[x, 2])
    MOI.add_constraint(model, con_f, MOI.LessThan(1.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    my_f(x) = (x - 0.5)^2
    MOI.set(model, MOI.UserDefinedFunction(:my_f, 1), (my_f,))
    obj_f = MOI.ScalarNonlinearFunction(:my_f, Any[x])
    MOI.set(model, MOI.ObjectiveFunction{typeof(obj_f)}(), obj_f)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.VariablePrimal(), x) isa Float64
end

"""
    test_SolverName()

You can also write new tests for solver-specific functionality. Write each new
test as a function with a name beginning with `test_`.
"""
function test_SolverName()
    @test MOI.get(SeqOpt.Optimizer(HiGHS.Optimizer), MOI.SolverName()) == "SeqOpt with HiGHS for linearized programs"
    return
end

end # module TestMOI

# This line at tne end of the file runs all the tests!
TestMOI.runtests()
