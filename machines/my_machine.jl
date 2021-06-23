using Random
import Base.rand

struct LogUniform
    a
    b    
end

Base.rand(rng::Random.AbstractRNG, d::LogUniform) = exp(log(d.a) + (log(d.b) - log(d.a)) * rand(rng))


model = (@load EpsilonSVR pkg=LIBSVM)()
PCA = @load PCA
pipe = @pipeline(
    FeatureSelector(features = name -> occursin("48.2_-3.1", string(name)),
                        ignore=true),
    Standardizer(),                 
    PCA(pratio=0.95),
    model,
    target=x->log.(x),
    inverse=x->exp.(x),
    name = "my_pipeline")
