using CSV, DataFrames, Statistics
using Random: seed!
seed!(90125)

# Load the train set
path = joinpath(ENV["VOLTAIRE"], "data", "datasets", "SILFIAC")
data = CSV.read(joinpath(path, "train.csv"), DataFrame)

# Separate data from labels
labels = 0:23 .|> x -> "power_$(x)"
X = data[:, Not(vcat(Symbol.(labels), :date_col))]
y = data[:, labels] |> eachrow .|> mean

using Distributed
n_cpus = 4
workers = addprocs(n_cpus, exeflags="--project=.")

@everywhere using DataFrames, Dates, MLJ, MLJBase
@everywhere smape(ytrue, ypred)= 200 * mean(@. abs(ytrue - ypred) / (abs(ytrue) + abs(ypred)))
@everywhere PCA = @load PCA
@everywhere model = (@load EpsilonSVR)()
@everywhere pipe = @pipeline(
    FeatureSelector(features = name -> occursin("48.2_-3.1", string(name)),
                        ignore=true),
    Standardizer(),                 
    PCA(pratio=0.95),
    model,
    target=x->log.(x.+1),
    inverse=x->exp.(x).-1,
    name = "my_pipeline")

@everywhere using Random
@everywhere import Base.rand
@everywhere struct LogUniform
    a
    b    
end

@everywhere Base.rand(rng::Random.AbstractRNG, d::LogUniform) = exp(log(d.a) + (log(d.b) - log(d.a)) * rand(rng))


r1 = (:(epsilon_svr.epsilon), LogUniform(10e-9, 10e9))
r2 = (:(epsilon_svr.gamma), LogUniform(10e-9, 10e9))
r3 = (:(pca.pratio), [0.75, 0.9, 0.95, 0.99])
r4 = (:(feature_selector.features), [
    name -> false,
    name -> occursin("48.3_-3.0", string(name)),
    name -> occursin("48.2_-3.0", string(name)),
    name -> occursin("48.3_-3.1", string(name)),
    name -> occursin("48.1_-3.0", string(name)),
    name -> occursin("48.2_-3.1", string(name)),
    name -> occursin("48.1_-3.1", string(name)),
    name -> occursin("48.1_-3.2", string(name)),
    name -> occursin("48.2_-3.2", string(name)),
    name -> occursin("48.2_-3.3", string(name))    
])

tuned_model = TunedModel(model = pipe,
                         resampling = Holdout(; fraction_train=0.75),
                         tuning = RandomSearch(),
                         range = [r1, r2, r3, r4],
                         measure = smape,
                         n = 100,
                         acceleration = CPUProcesses())
mach_tuned_model = machine(tuned_model, X, y)

tuned_model.n = 4
fit!(mach_tuned_model)

tuned_model.n = 254
start = now()
fit!(mach_tuned_model)
stop = now()
mach_tuned_model.report.best_model

println("BEST SMAPE IS : " * string(mach_tuned_model.report.best_history_entry.measurement[1]))
println("TOTAL TIME IS : " * string((stop - start).value/1000) * "s")
# 31.5%, 435 seconds

MLJ.save("my_machine.jlso", mach_tuned_model)
