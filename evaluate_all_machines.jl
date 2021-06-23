using MLJ, CSV, DataFrames

# Load the train set
path = joinpath(ENV["VOLTAIRE"], "data", "datasets", "SILFIAC")
script_path = joinpath(pwd(), "work", "SILFIAC")
data = CSV.read(joinpath(path, "test.csv"), DataFrame)

# Separate data from labels
labels = 0:23 .|> x -> "power_$(x)"
X = data[:, Not(vcat(Symbol.(labels), :date_col))]
y = data[:, labels] |> eachrow .|> mean

smape(ytrue, ypred)= 200 * mean(@. abs(ytrue - ypred) / (abs(ytrue) + abs(ypred)))
################################################################
machine_dir = joinpath(script_path, "machines")
machines = filter(x -> x[end-4:end] == ".jlso", readdir(machine_dir))
scripts = filter(x -> x[end-2:end] == ".jl", readdir(machine_dir))
results = 10000 * ones(Float64, length(machines))

for i in 1:length(results)

    machine_ = machines[i]
    script_ = scripts[i]

    try
        include(joinpath(machine_dir, script_))
        mach = machine(joinpath(machine_dir, machine_))
        results[i] = smape(y, mach(X))
    catch
        println("Error loading machine $(machine_)!!!")
    end
end

res = DataFrame(machines = machines, smapes = results)
res = sort!(res, :smapes)
    
