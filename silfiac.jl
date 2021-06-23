using CSV, DataFrames, Statistics, Dates
using MLJ, MLJBase, PyPlot

# Load the train set
path = joinpath(ENV["VOLTAIRE"], "data", "datasets", "SILFIAC")
data = CSV.read(joinpath(path, "train.csv"), DataFrame)

# Separate data from labels
labels = 0:23 .|> x -> "power_$(x)"
X = data[:, Not(vcat(Symbol.(labels), :date_col))]
y = data[:, labels] |> eachrow .|> mean

# Pick and load a model
Model = @load EpsilonSVR pkg=LIBSVM
model = Model(tolerance=1.0)

# Create a first machine
mach = machine(model, X, y)
fit!(mach)
predict(mach, X)

MLJ.save("machine_example.jlso", mach)
machine("machine_example.jlso", X, y)

# Inspect the machine
fitted_params(mach)

# list of all models
models(matching(X, y))

# Evaluate the forecasting model
# Pick a metric
using PyPlot
plot(y)
hist(y)

100 * mean(y .== 0)
100 * mean(y .<= 100)

smape(ytrue, ypred)= 200 * mean(@. abs(ytrue - ypred) / (abs(ytrue) + abs(ypred)))
smape(y, predict(mach, X))
MLJ.mae(y, predict(mach, X))

# evaluation loop
fraction_train = 1 - mean(year.(data.date_col) .== 2019)
evaluate!(mach; measure=[smape, MLJ.mae],
          resampling=Holdout(; fraction_train=fraction_train))

# Let's increase performances!
# Wrap the model in a pipeline
scaler = Standardizer()
mach_std = machine(scaler, X)
fit!(mach_std)
MLJ.transform(mach_std, X)
           
pipe = @pipeline Standardizer() model
mach_pipe = machine(pipe, X, y)
fit!(mach_pipe)
evaluate!(mach_pipe; measure=[smape, MLJ.mae],
          resampling=Holdout(; fraction_train=fraction_train))

pipe = @pipeline Standardizer() model target=Standardizer()
mach_pipe = machine(pipe, X, y)
evaluate!(mach_pipe; measure=[smape, MLJ.mae],
          resampling=Holdout(; fraction_train=fraction_train))

# Build a custom transformer
plot(log.(y))

pipe = @pipeline Standardizer() model target=x->log.(x) inverse=x->exp.(x)
mach_pipe = machine(pipe, X, y)
evaluate!(mach_pipe; measure=[smape, MLJ.mae],
          resampling=Holdout(; fraction_train=fraction_train))

# Final Pipeline
PCA = @load PCA
fs = FeatureSelector(features = name -> occursin("48.2_-3.1", string(name)),
                     ignore=true)
mach_fs = machine(fs, X)
fit!(mach_fs)
MLJ.transform(mach_fs, X)

pipe = @pipeline(FeatureSelector(features = name -> occursin("48.2_-3.1", string(name)),
                                 ignore=true),
                 Standardizer(),                 
                 PCA(pratio=0.95),
                 model,
                 target=x->log.(x),
                 inverse=x->exp.(x))
mach_pipe = machine(pipe, X, y)
evaluate!(mach_pipe; measure=measure=[smape, MLJ.mae],
          resampling=Holdout(; fraction_train=fraction_train))

# A little plotting
xindices = 1:length(y)
plot(xindices, y)
scatter(xindices, predict(mach_pipe, X), c="r")

# Optimize hyper-parameters
# Create a Log Uniform distributions
using Random
import Base.rand

struct LogUniform
    a
    b    
end

Base.rand(rng::Random.AbstractRNG, d::LogUniform) = exp(log(d.a) + (log(d.b) - log(d.a)) * rand(rng))

r1 = (:(epsilon_svr.cost), LogUniform(10e-9, 10e9))
r2 = (:(epsilon_svr.gamma), LogUniform(10e-9, 10e9))

tuned_model = TunedModel(model = pipe,
                         resampling = Holdout(; fraction_train=fraction_train),
                         measure = smape,                         
                         tuning = RandomSearch(),
                         range = [r1, r2],
                         n = 10)

mach_tuned_model = machine(tuned_model, X, y)
fit!(mach_tuned_model)

pms = mach_tuned_model.report.plotting
scatter(pms.parameter_values[:, 1], pms.parameter_values[:, 2],
        c=pms.measurements, ec="k", cmap=get_cmap("Reds"))
xlabel(pms.parameter_names[1])
ylabel(pms.parameter_names[2])
colorbar()
xscale("log")
yscale("log")

# Optimize other parameters
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
                         resampling = Holdout(; fraction_train=fraction_train),
                         tuning = RandomSearch(),
                         range = [r1, r2, r3, r4],
                         measure = smape,
                         n = 10)

mach_tuned_model = machine(tuned_model, X, y)
fit!(mach_tuned_model)
predict(mach_tuned_model, X)

# Use acceleration
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
    target=x->log.(x),
    inverse=x->exp.(x),
    name = "my_pipeline")

@everywhere using Random
@everywhere import Base.rand
@everywhere  struct LogUniform
    a
    b    
end

@everywhere Base.rand(rng::Random.AbstractRNG, d::LogUniform) = exp(log(d.a) + (log(d.b) - log(d.a)) * rand(rng))

tuned_model = TunedModel(model = pipe,
                         resampling = Holdout(; fraction_train=fraction_train),
                         tuning = RandomSearch(),
                         range = [r1, r2, r3, r4],
                         measure = smape,
                         n = 10,
                         acceleration = CPUProcesses())
mach_tuned_model = machine(tuned_model, X, y)
fit!(mach_tuned_model)
mach_tuned_model.report.best_model

points = ["", "48.3_-3.0", "48.2_-3.0", "48.3_-3.1", "48.1_-3.0", "48.2_-3.1",
          "48.1_-3.1", "48.1_-3.2", "48.2_-3.2", "48.2_-3.3"]
points[r2[2] .==  mach_tuned_model.report.best_model.feature_selector.features]
