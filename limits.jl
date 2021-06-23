y = data[:, labels]
function smape(ytrue, ypred)
    ytrue, ypred = MLJ.matrix(ytrue), MLJ.matrix(ypred)
    200 * mean(@. abs(ytrue - ypred) / (abs(ytrue) + abs(ypred)))
end

model = (@load MultitargetNeuralNetworkRegressor)()
pipe = @pipeline(
    FeatureSelector(features = name -> occursin("48.2_-3.1", string(name)),
                        ignore=true),
    Standardizer(),                 
    PCA(pratio=0.95),
    model,
    target=Standardizer())
mach_pipe = machine(pipe, X, y)
evaluate!(mach_pipe; measure=smape,
          resampling=Holdout(; fraction_train=0.75))

model.loss = smape
pipe.multitarget_neural_network_regressor = (@load MultitargetNeuralNetworkRegressor)(loss=smape)
evaluate!(mach_pipe; measure=smape,
          resampling=Holdout(; fraction_train=0.75))

# Few Nativly implemented models
models()
models(x -> x.is_pure_julia)
models(matching(X, y))[models(matching(X, y)) .|> x -> getproperty(x, :is_pure_julia)]
