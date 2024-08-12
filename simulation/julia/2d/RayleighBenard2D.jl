# julia -> ] -> activate . -> backspace -> include("RayleighBenard2D.jl")

using Printf
using Oceananigans
using Statistics
using NPZ
# using CUDA # (1) julia -> ] -> add CUDA (2) uncomment line 42


#dir variable
dirpath = string(@__DIR__)

Lx = 2 * pi
Lz = 2

Nx = 96
Nz = 64


Δt = 0.03
Δt_snap = 1.5
duration = 300

Ra = 1e4
Pr = 0.71

Re = sqrt(Ra / Pr)

ν = 1 / Re
κ = 1 / Re


# Temperature difference between bottom and top plate
Δb = 1

# Set the amplitude of the random perturbation (kick)
kick = 0.2


grid = RectilinearGrid(size=(Nx, Nz), x=(0, Lx), z=(0, Lz), topology=(Periodic, Flat, Bounded))

# GPU version would be:
# grid = RectilinearGrid(GPU(), size = (Nx, Nz), x = (0, Lx), z = (0, Lz), topology = (Periodic, Flat, Bounded))



u_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(0),
    bottom=ValueBoundaryCondition(0))
# v: vel in y direction
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
    boundary_conditions=(u=u_bcs, b=b_bcs,),
    coriolis=nothing
)

# Set initial conditions
uᵢ(x, z) = kick * randn()
wᵢ(x, z) = kick * randn()
bᵢ(x, z) = 1 + (2 - z) * Δb / 2 + kick * randn()

# Send the initial conditions to the model to initialize the variables
set!(model, u=uᵢ, w=wᵢ, b=bᵢ)

# Now, we create a 'simulation' to run the model for a specified length of time
simulation = Simulation(model, Δt=Δt, stop_time=Δt_snap)

cur_time = 0.0

simulation.verbose = true


# Now, run the simulation
totalsteps = Int(duration / Δt_snap)

temps = zeros(totalsteps + 1, Nx, Nz)
temps[1, :, :] = model.tracers.b[1:Nx, 1, 1:Nz]

vels = zeros(totalsteps + 1, 3, Nx, Nz)
vels[1, 1, :, :] = model.velocities.u[1:Nx, 1, 1:Nz]
vels[1, 2, :, :] = model.velocities.v[1:Nx, 1, 1:Nz]
vels[1, 3, :, :] = model.velocities.w[1:Nx, 1, 1:Nz]

for i in 1:totalsteps

    #update the simulation stop time for the next step
    global simulation.stop_time = Δt_snap * i

    run!(simulation)
    global cur_time += Δt_snap

    # collect results
    temps[i+1, :, :] = model.tracers.b[1:Nx, 1, 1:Nz]
    vels[i+1, 1, :, :] = model.velocities.u[1:Nx, 1, 1:Nz]
    vels[i+1, 2, :, :] = model.velocities.v[1:Nx, 1, 1:Nz]
    vels[i+1, 3, :, :] = model.velocities.w[1:Nx, 1, 1:Nz]

    if (any(isnan, temps[i+1, :, :]) ||
        any(isnan, vels[i+1, 1, :, :]) ||
        any(isnan, vels[i+1, 2, :, :]) ||
        any(isnan, vels[i+1, 3, :, :]))

        printstyled("[WARNING] NaN values found!\n"; color=:red)
    end
    
    println(cur_time)
end

# save data in npy file
# use `npzread(save_file)` (or `np.load(save_file)`` in python) to read data
println("Saving data in .npy file...")

simulation_name = "$(Nx)_$(Nz)_$(Ra)_$(Pr)_$(Δt)_$(Δt_snap)_$(duration)"
data_dir = joinpath(dirpath, "data", simulation_name)
mkpath(data_dir)
save_file = joinpath(data_dir, "sim.npy")
npzwrite(save_file, temperature=temps, velocity=vels)

println("Simulation data saved as: $(save_file)")