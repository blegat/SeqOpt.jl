using Test

for file in readdir(@__DIR__)
    if file in ["runtests.jl"]
        continue
    elseif !endswith(file, ".jl")
        continue
    end
    @testset "$(file)" begin
        include(file)
    end
end
