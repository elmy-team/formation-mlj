using CSV, DataFrames, Statistics, Dates

path = joinpath(ENV["VOLTAIRE"], "data", "datasets", "SILFIAC")
data = CSV.read(
    joinpath(path, "exports_Silfiac_2015-01-01-2021-01-01_20210617081340300_file_0.csv"),
    DataFrame, header=0)
header = CSV.read(
    joinpath(path, "exports_Silfiac_2015-01-01-2021-01-01_20210617081340300_header.csv"),
    DataFrame, header=1)

rename!(data, zip(names(data), names(header)) .|> x -> Pair(x[1], x[2]))
dfmt = DateFormat("yyyy-mm-ddTHH:MM:SSZ")
data.forecastTime = DateTime.(data.forecastTime, Ref(dfmt))
sort!(data, :forecastTime)

variables =  ["u_component_of_wind_10m",
              "v_component_of_wind_10m",
              "u_component_of_wind_gust_10m",
              "v_component_of_wind_gust_10m",
              "temperature_2m",
              "relative_humidity_2m",
              "pressure_mean_sea_level",
              "u_component_of_wind_50m",
              "v_component_of_wind_50m",
              "u_component_of_wind_75m",
              "v_component_of_wind_75m",
              "u_component_of_wind_100m",
              "v_component_of_wind_100m"]
variables = variables[variables .|> x -> data[!, x] .|> ismissing |> mean .|> x -> x < 0.05]
points = zip(data.lat, data.lon) |> unique

points_suffixes = points .|> x -> "_$(x[1])_$(x[2])"
data[!, :hour] = hour.(data.forecastTime)

# Get forecasts for the next day
data = data[Date.(data.forecastTime) .== data.publicationDate + Day(1), :]
dropmissing!(data, variables)
dates =  data.forecastTime .|> Date |> unique
n = dates |> length
m = length(variables) * length(points) * 24

X = missings(Float64, n, m)
for (i, date) in dates |> enumerate
    x = data[Date.(data.forecastTime) .== date, :]
    (size(x, 1) != m/length(variables)) && continue

    for h in 0:23
        xtemp = x[x.hour .== h, variables]
        xtemp = xtemp |> Matrix |> x -> reshape(x, *(size(x)...)) |> Vector
        nf = Int(m/24)
        X[i, 1 + (h * nf):(h * nf) + nf] = xtemp
    end
end

cols = vcat((variables .|> x -> [x * ps * "_$(h)" for ps in points_suffixes])...)
cols = vcat((0:23 .|> x -> [col * "_$(x)" for col in cols])...)
df = DataFrame(X, cols) 
df[!, :date_col] = dates
dropmissing!(df)
CSV.write(joinpath(path, "input_data.csv"), df)

# Load labels
label_path = joinpath(path, "silfi_2021_02_08_production_parc_eolien_pas_10_min_2015_a_2020")
dfmt = DateFormat(" dd/mm/yyyy HH:MM")
label = " P Ø [kW]"
labels = DataFrame()
for y in 2015:2020
    labels_ = CSV.read(
        joinpath(label_path, "Eolienne_10 minutes_2616_$(y)-01-01_$(y)-12-31.csv"),
        DataFrame, header=1)
    labels = vcat(labels, labels_)
end
labels.Heure = DateTime.(labels.Heure, Ref(dfmt))

# Sum across all four units
labels = combine(groupby(labels, :Heure), label => sum)
labels[!, :unique_hour] = labels.Heure .|> x -> (Date(x), hour(x))
labels = combine(groupby(labels, :unique_hour), label * "_sum" => mean)

u_dates = unique(labels.unique_hour .|> x -> x[1])
Y = missings(Float64, length(u_dates), 24)
for (i, date) in u_dates |> enumerate
    x = labels[getindex.(labels.unique_hour, Ref(1)) .== date, label * "_sum" * "_mean"]
    length(x) != 24 && continue
    Y[i, :] = x
end
cols = 0:23 .|> x -> "power_$(x)"
dfY = DataFrame(Y, cols) 
dfY[!, :date_col] = u_dates
dropmissing!(dfY)

CSV.write(joinpath(path, "output_data.csv"), dfY)

# Merge and constitute train/test pair
dataset = innerjoin(df, dfY; on=:date_col)
CSV.write(joinpath(path, "all_data.csv"), dataset)

# Split data
testyear = 2020
indices = year.(dataset.date_col) .< testyear

train_dataset = dataset[indices, :]
test_dataset = dataset[.! indices, :]

CSV.write(joinpath(path, "train.csv"), train_dataset)
CSV.write(joinpath(path, "test.csv"), test_dataset)

