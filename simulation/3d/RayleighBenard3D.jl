# run: julia -> ] -> activate . -> backspace -> include("RayleighBenard3D.jl")

using Printf
using Oceananigans
using Statistics
using HDF5
using CUDA # when using GPU: (1) julia -> ] -> add CUDA (2) uncomment line 44


# script directory
dirpath = string(@__DIR__)

# domain size
Lx = 2 * pi
Ly = 2 * pi
Lz = 2

# number of discrete points
Nx = 48
Ny = 48
Nz = 32

# time
Δt = 0.01 # simulation delta
Δt_snap = 0.3 # save delta
duration = 300 # duration of simulation

Ra = 1500
Pr = 0.71

Re = sqrt(Ra / Pr)

ν = 1 / Re
κ = 1 / Re


# Temperature difference between bottom and top plate
Δb = 1

# Set the amplitude of the random perturbation (kick)
kick = 0.2


# without GPU:
# grid = RectilinearGrid(size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(0, Lz), topology=(Periodic, Periodic, Bounded))
# with GPU:
grid = RectilinearGrid(GPU(), size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(0, Lz), topology=(Periodic, Periodic, Bounded))


u_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(0),
    bottom=ValueBoundaryCondition(0))
v_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(0),
    bottom=ValueBoundaryCondition(0))
# w_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(0),
#                                 bottom = ValueBoundaryCondition(0))
b_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(1),
    bottom=ValueBoundaryCondition(1 + Δb))

model = NonhydrostaticModel(; grid,
    advection=UpwindBiasedFifthOrder(),
    timestepper=:RungeKutta3,
    tracers=(:b),
    buoyancy=Buoyancy(model=BuoyancyTracer()),
    closure=(ScalarDiffusivity(ν=ν, κ=κ)),
    boundary_conditions=(u=u_bcs, v=v_bcs, b=b_bcs,),
    coriolis=nothing
)

# Set initial conditions
uᵢ(x, y, z) = kick * randn()
vᵢ(x, y, z) = kick * randn()
wᵢ(x, y, z) = kick * randn()
bᵢ(x, y, z) = 1 + (2 - z) * Δb / 2 + kick * randn()

# Send the initial conditions to the model to initialize the variables
set!(model, u=uᵢ, v=vᵢ, w=wᵢ, b=bᵢ)

# Now, we create a 'simulation' to run the model for a specified length of time
simulation = Simulation(model, Δt=Δt, stop_time=Δt_snap)

cur_time = 0.0

simulation.verbose = true


totalsteps = Int(div(duration, Δt_snap))

# Preparing HDF5 file
simulation_name = "$(Nx)_$(Ny)_$(Nz)_$(Ra)_$(Pr)_$(Δt)_$(Δt_snap)_$(duration)"

data_dir = joinpath(dirpath, "data", simulation_name)
mkpath(data_dir) # create if not existent

i = 1
while isfile(joinpath(data_dir, "sim$(i).h5"))
    global i += 1
end

h5_file_path = joinpath(data_dir, "sim$(i).h5")

if isfile(h5_file_path) 
    print("Do you want to overwrite $(simulation_name)? (y/n)")
    if readline() != "y"
        exit()
    end
    rm(h5_file_path)
end

h5_file = h5open(h5_file_path, "w")

data = create_dataset(h5_file, "data", datatype(Float64),
    dataspace(totalsteps + 1, 4, Nx, Ny, Nz), chunk=(1, 1, Nx, Ny, Nz))
# temps = create_dataset(h5_file, "temperature", datatype(Float64),
#     dataspace(totalsteps + 1, Nx, Ny, Nz), chunk=(1, Nx, Ny, Nz))
# vels = create_dataset(h5_file, "velocity", datatype(Float64),
#     dataspace(totalsteps + 1, 3, Nx, Ny, Nz), chunk=(1, 1, Nx, Ny, Nz))

# save initial state
data[1, 1, :, :, :] = model.tracers.b[1:Nx, 1:Ny, 1:Nz]
data[1, 2, :, :, :] = model.velocities.u[1:Nx, 1:Ny, 1:Nz]
data[1, 3, :, :, :] = model.velocities.v[1:Nx, 1:Ny, 1:Nz]
data[1, 4, :, :, :] = model.velocities.w[1:Nx, 1:Ny, 1:Nz]

for i in 1:totalsteps
    #update the simulation stop time for the next step
    global simulation.stop_time = Δt_snap * i

    run!(simulation)
    global cur_time += Δt_snap

    # collect results
    data[i+1, 1, :, :, :] = model.tracers.b[1:Nx, 1:Ny, 1:Nz]
    data[i+1, 2, :, :, :] = model.velocities.u[1:Nx, 1:Ny, 1:Nz]
    data[i+1, 3, :, :, :] = model.velocities.v[1:Nx, 1:Ny, 1:Nz]
    data[i+1, 4, :, :, :] = model.velocities.w[1:Nx, 1:Ny, 1:Nz]

    # check for NaNs
    if (any(isnan, model.tracers.b[1:Nx, 1:Ny, 1:Nz]) ||
        any(isnan, model.velocities.u[1:Nx, 1:Ny, 1:Nz]) ||
        any(isnan, model.velocities.v[1:Nx, 1:Ny, 1:Nz]) ||
        any(isnan, model.velocities.w[1:Nx, 1:Ny, 1:Nz]))

        printstyled("[ERROR] NaN values found!\n"; color=:red)
        exit()
    end

    println(cur_time)
end


close(h5_file)
println("Simulation data saved as: $(h5_file_path)")