module SeqOpt

import SparseArrays
import MathOptInterface as MOI

struct Optimizer{O} <: MOI.AbstractOptimizer
    nonlinear::MOI.Nonlinear.Model
    linearized::O
    solution::Vector{Float64}
    function Optimizer(opt_constuctor)
        nonlinear = MOI.Nonlinear.Model()
        linearized = MOI.instantiate(opt_constuctor, with_bridge_type = Float64)
        return new{typeof(linearized)}(nonlinear, linearized, Float64[])
    end
end

function MOI.get(model::Optimizer, attr::MOI.SolverName)
    lin = MOI.get(model.linearized, attr)
    return "SeqOpt with $lin for linearized programs"
end

MOI.add_variable(model::Optimizer) = MOI.add_variable(model.linearized)

function MOI.supports_constraint(
    model::Optimizer,
    ::Type{F},
    ::Type{S},
) where {F<:MOI.AbstractFunction,S<:MOI.AbstractSet}
    return MOI.supports_constraint(model.linearized) ||
           MOI.supports_constraint(model.nonlinear)
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
    vars = MOI.get(model.linearized, MOI.ListOfVariableIndices())
    backend = MOI.Nonlinear.SparseReverseMode()
    evaluator = MOI.Nonlinear.Evaluator(model.nonlinear, backend, vars)
    MOI.initialize(evaluator, [:Grad, :Jac])
    IJ = MOI.jacobian_structure(evaluator)
    I = getindex.(IJ, 1)
    J = getindex.(IJ, 2)
    resize!(model.solution, length(vars))
    for i in eachindex(model.solution)
        model.solution[i] = 0.0
    end
    constraint_map = Dict{MOI.Nonlinear.ConstraintIndex,MOI.ConstraintIndex}()
    _linearize(
        model.linearized,
        model.nonlinear,
        constraint_map,
        evaluator,
        I,
        J,
        vars,
        model.solution,
    )
    for i in 1:2 # FIXME stopping crit
        MOI.optimize!(model.linearized)
        MOI.get!(model.solution, model.linearized, MOI.VariablePrimal(), vars)
        _linearize(
            model.linearized,
            model.nonlinear,
            constraint_map,
            evaluator,
            I,
            J,
            vars,
            model.solution,
        )
    end
    return
end

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    return MOI.get(model.linearized, attr, vi)
end

end # module SeqOpt
