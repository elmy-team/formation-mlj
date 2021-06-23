using MLJModels, MLJModelInterface
import MLJModelInterface: clean!, fit, fitted_params, transform, inverse_transform
const MMI = MLJModelInterface

target(y) = asinh.(y)
inverse(y) = sinh.(y)

mutable struct FStandardizer<:MLJModels.Unsupervised
    standardizer::MLJModels.Unsupervised
    target
    inverse
end
FStandardizer() = FStandardizer(Standardizer(), target, inverse)

MMI.clean!(fs::FStandardizer) = MMI.clean!(fs.standardizer)
MMI.fit(fs::FStandardizer, verbosity::Int, X) = MMI.fit(fs.standardizer, verbosity, X)
MMI.fitted_params(fs::FStandardizer, fitresult) = MMI.fitted_params(fs.standardizer, fitresult)
MMI.transform(fs::FStandardizer, fitresult, X) = MMI.transform(fs.standardizer, fitresult, X) |> fs.target
MMI.inverse_transform(fs::FStandardizer, fitresult, X) = MMI.inverse_transform(fs.standardizer, fitresult, fs.inverse(X))
