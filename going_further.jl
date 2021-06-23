# Going further

# A custom target transformer
script_path = joinpath(pwd(), "work", "SILFIAC")
include(joinpath(script_path, "FStandardizer.jl"))

fs = FStandardizer(Standardizer(), y -> asinh.(y), y -> sinh.(y))
mach_fs = machine(fs, y)
fit!(mach_fs)
plot(MLJ.transform(mach_fs, y))

pipe = @pipeline(
    FeatureSelector(features = name -> occursin("48.2_-3.2", string(name)),
                    ignore=true),
    Standardizer(),                 
    PCA(pratio=0.99),
    model,
    target=fs)
mach_pipe = machine(pipe, X, y)
evaluate!(mach_pipe; measure=smape, resampling=Holdout(; fraction_train=0.75))

# A NON-linear pipeline : Learning networks

# Multivariate case
y = data[:, labels]
model = (@load MultitargetRidgeRegressor pkg=MultivariateStats)()
function smape(ytrue, ypred)
    ytrue, ypred = MLJ.matrix(ytrue), MLJ.matrix(ypred)
    200 * mean(@. abs(ytrue - ypred) / (abs(ytrue) + abs(ypred)))
end

Xs = source(X)
Ys = source(y)

fs = machine(
    FeatureSelector(features = name -> occursin("48.2_-3.2", string(name)), ignore=true),
    Xs)
X1 = transform(fs, Xs)

st_in = machine(Standardizer(), X1)
X2 = transform(st_in, X1)

pca_in = machine(PCA(pratio=0.99), X2)
X3 = transform(pca_in, X2)

pca_out = machine(PCA(pratio=0.99), Ys)
Y1 = transform(pca_out, Ys)

st_out = machine(Standardizer(), Y1)
Y2 = transform(st_out, Y1)

Y3 = @node broadcast(x -> asinh(x), Y2)

mach = machine(model, X3, Y3)
ypred = predict(mach, X3)
ypred1 = @node broadcast(x -> sinh(x), ypred)
ypred2 = inverse_transform(st_out, ypred1)
ypred3 = inverse_transform(pca_out, ypred2)

fit!(ypred3)
ypred3(X)

pca_in.model.pratio=0.95
fit!(ypred3)

mach = machine(Deterministic(), Xs, Ys; predict=ypred3);
evaluate!(mach; measure=smape, resampling=Holdout(; fraction_train=0.75))


# Model updating
y = data[:, labels] |> eachrow .|> mean
smape(ytrue, ypred)= 200 * mean(@. abs(ytrue - ypred) / (abs(ytrue) + abs(ypred)))

model = Model()
r1 = (:cost, LogUniform(10e-9, 10e9))
r2 = (:gamma, LogUniform(10e-9, 10e9))
tuned_model = TunedModel(model = model,
                         resampling = Holdout(; fraction_train=0.75),
                         tuning = RandomSearch(),
                         range = [r1, r2],
                         measure = smape,
                         n = 10)
mach_tuned_model = machine(tuned_model, X, y)
fit!(mach_tuned_model)

mach_tuned_model.model.n += 6
fit!(mach_tuned_model)
mach_tuned_model.report.plotting

# Adding estimators to an ensemble
ensemble = EnsembleModel(atom=model, bagging_fraction=0.75, n=5)
mach_ensemble = machine(ensemble, X, y)
fit!(mach_ensemble)
mach_ensemble.fitresult.ensemble 

ensemble.n += 5
fit!(mach_ensemble)
mach_ensemble.fitresult.ensemble

