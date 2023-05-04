module SeqOpt

import LinearAlgebra, SparseArrays
import MathOptInterface as MOI

struct IterationLimit <: MOI.AbstractOptimizerAttribute end
struct MinStepSize <: MOI.AbstractOptimizerAttribute end
struct MaxStepSize <: MOI.AbstractOptimizerAttribute end

const DEFAULT_OPTIONS = Dict{String,Any}(
    "max_iters" => 100,
    "ϵ_primal" => 1e-4,
    "min_step_size" => 1.0,
    "max_step_size" => 1.0,
)

const RAW_OPTIMIZE_NOT_CALLED = "Optimize not called"

mutable struct Solution{T}
    primal::Vector{T}
    raw_status::String
    termination_status::MOI.TerminationStatusCode
    primal_status::MOI.ResultStatusCode
    dual_status::MOI.ResultStatusCode
    solve_time::Float64
    iter::Int
    function Solution{T}(k) where T
        return new{T}(
            zeros(T, k),
            RAW_OPTIMIZE_NOT_CALLED,
            MOI.OPTIMIZE_NOT_CALLED,
            MOI.UNKNOWN_RESULT_STATUS,
            MOI.UNKNOWN_RESULT_STATUS,
            0.0,
            0,
        )
    end
end

struct Optimizer{O} <: MOI.AbstractOptimizer
    nonlinear::MOI.Nonlinear.Model
    map_linearized::MOI.Utilities.IndexMap
    linearized::O
    solution::Union{Nothing,Solution{Float64}}
    silent::Bool
    options::Dict{String,Any}
    function Optimizer(opt_constuctor)
        nonlinear = MOI.Nonlinear.Model()
        linearized = MOI.instantiate(opt_constuctor, with_bridge_type = Float64)
        return new{typeof(linearized)}(nonlinear, MOI.Utilities.IndexMap(), linearized, Solution{Float64}(0), false, copy(DEFAULT_OPTIONS))
    end
end

function MOI.is_empty(optimizer::Optimizer)
    return MOI.is_empty(optimizer.nonlinear) && MOI.is_empty(optimizer.linearized)
end

function MOI.empty!(optimizer::Optimizer)
    MOI.empty!(optimizer.nonlinear)
    MOI.empty!(optimizer.map_linearized)
    MOI.empty!(optimizer.linearized)
    optimizer.solution = nothing
    return
end

function MOI.get(model::Optimizer, attr::MOI.SolverName)
    lin = MOI.get(model.linearized, attr)
    return "SeqOpt with $lin for linearized programs"
end

# MOI.RawOptimizerAttribute

function MOI.supports(optimizer::Optimizer, param::MOI.RawOptimizerAttribute)
    return haskey(optimizer.options, param.name)
end

function MOI.set(optimizer::Optimizer, param::MOI.RawOptimizerAttribute, value)
    if !MOI.supports(optimizer, param)
        throw(MOI.UnsupportedAttribute(param))
    end
    optimizer.options[param.name] = value
    return
end

function MOI.get(optimizer::Optimizer, param::MOI.AbstractOptimizerAttribute)
    if !MOI.supports(optimizer, param)
        throw(MOI.UnsupportedAttribute(param))
    end
    return optimizer.options[param.name]
end

# MOI.Silent

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(optimizer::Optimizer, ::MOI.Silent, value::Bool)
    optimizer.silent = value
    return
end

MOI.get(optimizer::Optimizer, ::MOI.Silent) = optimizer.silent

# Variables

function MOI.add_variable(model::Optimizer)
    push!(model.solution.primal, 0.0)
    vi = MOI.add_variable(model.linearized)
    model.map_linearized[vi] = MOI.VariableIndex(length(model.solution.primal))
    return vi
end

function MOI.set(model::Optimizer, ::MOI.VariablePrimalStart, vi::MOI.VariableIndex, value)
    model.solution.primal[model.map_linearized[vi].value] = value
    return
end

# Constraints

function MOI.supports_constraint(
    model::Optimizer,
    ::Type{F},
    ::Type{S},
) where {F<:MOI.AbstractFunction,S<:MOI.AbstractSet}
    return MOI.supports_constraint(model.linearized, F, S) ||
           MOI.supports_constraint(model.nonlinear, F, S)
end

function MOI.is_valid(model::Optimizer, ci::MOI.ConstraintIndex)
    return MOI.is_valid(model.linearized, ci)
end

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.ScalarNonlinearFunction},
)
    index = MOI.Nonlinear.ConstraintIndex(ci.value)
    return MOI.is_valid(model.nonlinear, index)
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.AbstractFunction,
    s::MOI.AbstractSet,
)
    return MOI.add_constraint(model.linearized, f, s)
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.ScalarNonlinearFunction,
    s::MOI.AbstractScalarSet,
)
    index = MOI.Nonlinear.add_constraint(model.nonlinear, f, s)
    return MOI.ConstraintIndex{typeof(f),typeof(s)}(index.value)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveSense,
    value::MOI.OptimizationSense,
)
    MOI.set(model.linearized, attr, value)
    return
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveFunction{F},
    func::F,
) where {F}
    MOI.Nonlinear.set_objective(model.nonlinear, nothing)
    MOI.set(model.linearized, attr, func)
    return
end

function MOI.set(
    model::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction},
    func::MOI.ScalarNonlinearFunction,
)
    MOI.Nonlinear.set_objective(model.nonlinear, func)
    return
end

### UserDefinedFunction

MOI.supports(model::Optimizer, ::MOI.UserDefinedFunction) = true

function MOI.set(model::Optimizer, attr::MOI.UserDefinedFunction, args)
    MOI.Nonlinear.register_operator(
        model.nonlinear,
        attr.name,
        attr.arity,
        args...,
    )
    return
end

### ListOfSupportedNonlinearOperators

function MOI.get(model::Optimizer, attr::MOI.ListOfSupportedNonlinearOperators)
    return MOI.get(model.nonlinear, attr)
end

function _linearize(
    linearized,
    nonlinear,
    constraint_map,
    evaluator,
    I,
    J,
    vars,
    solution,
)
    if !isnothing(nonlinear.objective)
        obj_val = MOI.eval_objective(evaluator, solution)
        grad = similar(solution)
        MOI.eval_objective_gradient(evaluator, grad, solution)
        aff =
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(grad, vars), obj_val)
        MOI.set(linearized, MOI.ObjectiveFunction{typeof(aff)}(), aff)
    end
    g = zeros(length(nonlinear.constraints))
    MOI.eval_constraint(evaluator, g, solution)
    V = zeros(length(I))
    MOI.eval_constraint_jacobian(evaluator, V, solution)
    G = SparseArrays.sparse(I, J, V)
    for (nl_ci, con) in nonlinear.constraints
        row = nl_ci.value
        G_row = G[row, :]
        aff = MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm{Float64}[
                MOI.ScalarAffineTerm(val, vars[ind]) for (ind, val) in zip(
                    SparseArrays.nonzeroinds(G_row),
                    SparseArrays.nonzeros(G_row),
                )
            ],
            g[row],
        )
        aff, set = MOI.Utilities.normalize_constant(
            aff,
            con.set,
            allow_modify_function = true,
        )
        if haskey(constraint_map, nl_ci)
            aff_ci = constraint_map[nl_ci]
            MOI.set(linearized, MOI.ConstraintFunction(), aff_ci, aff)
            MOI.set(linearized, MOI.ConstraintSet(), aff_ci, set)
        else
            constraint_map[nl_ci] = MOI.add_constraint(linearized, aff, set)
        end
    end
end

function MOI.optimize!(model::Optimizer)
    sol = model.solution
    options = model.options
    max_iters = options["max_iters"]
    min_step_size = options["min_step_size"]
    max_step_size = options["max_step_size"]
    vars = MOI.get(model.linearized, MOI.ListOfVariableIndices())
    backend = MOI.Nonlinear.SparseReverseMode()
    evaluator = MOI.Nonlinear.Evaluator(model.nonlinear, backend, vars)
    MOI.initialize(evaluator, [:Grad, :Jac])
    IJ = MOI.jacobian_structure(evaluator)
    I = getindex.(IJ, 1)
    J = getindex.(IJ, 2)
    constraint_map = Dict{MOI.Nonlinear.ConstraintIndex,MOI.ConstraintIndex}()
    Δprimal = Inf
    sol.solve_time = @elapsed for _ in 1:max_iters
        _linearize(
            model.linearized,
            model.nonlinear,
            constraint_map,
            evaluator,
            I,
            J,
            vars,
            model.solution.primal,
        )
        MOI.optimize!(model.linearized)
        new_primal = MOI.get(model.linearized, MOI.VariablePrimal(), vars)
        Δprimal = new_primal - model.solution.primal
        if min_step_size != max_step_size
            error("Line search not supported yet")
        else
            @. model.solution.primal += min_step_size * Δprimal
        end
        if LinearAlgebra.norm(Δprimal) <= options["ϵ_primal"]
            sol.raw_status = "Solved to stationarity"
            sol.termination_status = MOI.LOCALLY_SOLVED
            sol.primal_status = MOI.UNKNOWN_RESULT_STATUS
            sol.dual_status = MOI.NO_SOLUTION
            return
        end
    end
    sol.raw_status = "Maximum number of iterations ($max_iters) reached"
    sol.termination_status = MOI.ITERATION_LIMIT
    sol.primal_status = MOI.UNKNOWN_RESULT_STATUS
    sol.dual_status = MOI.NO_SOLUTION
    return
end

MOI.get(optimizer::Optimizer, ::MOI.SolveTimeSec) = optimizer.solution.solve_time

function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    return optimizer.solution.raw_status
end

# Implements getter for result value and statuses

function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    return optimizer.solution.termination_status
end

function MOI.get(optimizer::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    return optimizer.solution.primal_status
end

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(model, attr)
    return model.solution.primal[model.map_linearized[vi].value]
end

function MOI.get(optimizer::Optimizer, attr::MOI.DualStatus)
    if attr.result_index > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    return optimizer.solution.dual_status
end

function MOI.get(optimizer::Optimizer, ::MOI.ResultCount)
    if isnothing(optimizer.solution)
        return 0
    else
        return 1
    end
end


end # module SeqOpt
