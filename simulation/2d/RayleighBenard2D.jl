#? run: cd simulation -> julia -> ] -> activate . -> backspace -> include("RayleighBenard2D.jl") -> simulate_2d_rb()

using Printf
import Random
using Oceananigans
using Statistics
using HDF5
using CUDA
using Plots

theme(:dark)

# supports gpu for simulation
use_gpu = true

# create animations of data
visualize = true
fps = 15

# script directory
dirpath = string(@__DIR__)

# domain size
L = (2 * pi, 2) # x,z

# number of discrete sampled points
N = (128, 64)

# time
Δt = 0.01 # simulation delta
Δt_snap = 0.3 # save delta
duration = 300 # duration of simulation

# temperature
min_b = 0 # Temperature at top plate
Δb = 1 # Temperature difference between bottom and top plate

# Rayleigh Benard Parameters
Ra = 10^5
Pr = 0.7

# Set the amplitude of the random initial perturbation (kick)
random_kick = 0.2


function simulate_2d_rb(; random_initializations=1, Ra=Ra, Pr=Pr, N=N, L=L, min_b=min_b, Δb=Δb, random_kick=random_kick,
    Δt=Δt, Δt_snap=Δt_snap, duration=duration, use_gpu=use_gpu, visualize=visualize, fps=fps)

    ν = sqrt(Pr / Ra) # c.f. line 33: https://github.com/spectralDNS/shenfun/blob/master/demo/RayleighBenard2D.py
    κ = 1 / sqrt(Pr*Ra) # c.f. line 37: https://github.com/spectralDNS/shenfun/blob/master/demo/RayleighBenard2D.py

    totalsteps = Int(div(duration, Δt_snap))

    grid = define_sample_grid(N, L, use_gpu)
    u_bcs, b_bcs = define_boundary_conditions(min_b, Δb)

    for i ∈ 1:random_initializations
        println("Simulating random initialization $(i)/$(random_initializations)...")

        simulation_name = "x$(N[1])_z$(N[2])_Ra$(Ra)_Pr$(Pr)_t$(Δt)_snap$(Δt_snap)_dur$(duration)"
        h5_file, dataset, h5_file_path, sim_num = create_hdf5_dataset(simulation_name, N, totalsteps)

        # Make sure that every random initialization is indeed independend of each other
        # (even when script is restarted)
        Random.seed!(sim_num)

        model = define_model(grid, ν, κ, u_bcs, b_bcs)
        initialize_model(model, min_b, L[2], Δb, random_kick)

        simulate_model(model, dataset, Δt, Δt_snap, totalsteps, N)

        if visualize
            animation_dir = joinpath(dirpath, "data", simulation_name, "sim$(sim_num)", "animations")
            mkpath(animation_dir)
            for (channel_num, channel_name) in enumerate(["temp", "u", "w"])
                println("Animating $(channel_name)...")
                visualize_simulation(dataset, animation_dir, channel_num, channel_name, fps, N, L, Δt_snap, min_b, Δb)
            end
        end

        close(h5_file)
        println("Simulation data saved as: $(h5_file_path)")
    end
end


function define_sample_grid(N, L, use_gpu)
    if use_gpu
        grid = RectilinearGrid(GPU(), size=N, x=(0, L[1]), z=(0, L[2]),
                               topology=(Periodic, Flat, Bounded))
    else
        grid = RectilinearGrid(size=(N), x=(0, L[1]), z=(0, L[2]), 
                               topology=(Periodic, Flat, Bounded))
    end
    return grid
end


function define_boundary_conditions(min_b, Δb)
    u_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(0),
        bottom=ValueBoundaryCondition(0))
    # no explicit boundary condition for vertical velocity (seems to be inferred automatically)
    # w_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(0),
    #                                 bottom = ValueBoundaryCondition(0))
    b_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(min_b),
        bottom=ValueBoundaryCondition(min_b + Δb))
    return u_bcs, b_bcs
end


function define_model(grid, ν, κ, u_bcs, b_bcs)
    model = NonhydrostaticModel(; grid,
        advection=UpwindBiasedFifthOrder(),
        timestepper=:RungeKutta3,
        tracers=(:b),
        buoyancy=Buoyancy(model=BuoyancyTracer()),
        closure=(ScalarDiffusivity(ν=ν, κ=κ)),
        boundary_conditions=(u=u_bcs, b=b_bcs,),
        coriolis=nothing
    )
    return model
end


function initialize_model(model, min_b, Lz, Δb, kick)
    # Set initial conditions
    uᵢ(x, z) = kick * randn()
    wᵢ(x, z) = kick * randn()
    bᵢ(x, z) = min_b + (Lz - z) * Δb / 2 + kick * randn()

    # Send the initial conditions to the model to initialize the variables
    set!(model, u=uᵢ, w=wᵢ, b=bᵢ)
end


function create_hdf5_dataset(simulation_name, N, totalsteps)
    data_dir = joinpath(dirpath, "data", simulation_name)
    mkpath(data_dir) # create directory if not existent

    # compute number of this simulation
    sim_num = 1
    while isfile(joinpath(data_dir, "sim$(sim_num)", "sim.h5"))
        sim_num += 1
    end

    mkpath(joinpath(data_dir, "sim$(sim_num)"))
    path = joinpath(data_dir, "sim$(sim_num)", "sim.h5")
    h5_file = h5open(path, "w")
    # save temperature and velocities in one dataset:
    dataset = create_dataset(h5_file, "data", datatype(Float64),
        dataspace(totalsteps + 1, 3, N...), chunk=(1, 1, N...))

    # seperate datasets for temperature and velocity:
    # temps = create_dataset(h5_file, "temperature", datatype(Float64),
    #     dataspace(totalsteps + 1, N...), chunk=(1, N...))
    # vels = create_dataset(h5_file, "velocity", datatype(Float64),
    #     dataspace(totalsteps + 1, 2, N...), chunk=(1, 1, N...))

    return h5_file, dataset, path, sim_num
end


function simulate_model(model, dataset, Δt, Δt_snap, totalsteps, N)
    simulation = Simulation(model, Δt=Δt, stop_time=Δt_snap)
    simulation.verbose = true

    cur_time = 0.0

    # save initial state
    save_simulation_step(model, dataset, 1, N)

    for i in 1:totalsteps
        #update the simulation stop time for the next step
        global simulation.stop_time = Δt_snap * i

        run!(simulation)
        cur_time += Δt_snap

        save_simulation_step(model, dataset, i + 1, N)

        if (step_contains_NaNs(model, N))
            printstyled("[ERROR] NaN values found!\n"; color=:red)
            return
        end

        println(cur_time)
    end
end


function save_simulation_step(model, dataset, step, N)
    dataset[step, 1, :, :] = model.tracers.b[1:N[1], 1, 1:N[2]]
    dataset[step, 2, :, :] = model.velocities.u[1:N[1], 1, 1:N[2]]
    dataset[step, 3, :, :] = model.velocities.w[1:N[1], 1, 1:N[2]]
end


function step_contains_NaNs(model, N)
    contains_nans = (any(isnan, model.tracers.b[1:N[1], 1, 1:N[2]]) ||
                     any(isnan, model.velocities.u[1:N[1], 1, 1:N[2]]) ||
                     any(isnan, model.velocities.w[1:N[1], 1, 1:N[2]]))
    return contains_nans
end


function visualize_simulation(data, animation_dir, channel, channel_name, fps, N, L, Δt_snap, min_b, Δb)
    if channel == 1 # temperature channel
        clims = (min_b, min_b+Δb)
    else
        clims = (minimum(data[:, channel, :, :]), maximum(data[:, channel, :, :]))
    end

    function show_snapshot(i)
        t = round((i - 1) * Δt_snap, digits=1)
        x = range(0, L[1], length=N[1])
        z = range(0, L[2], length=N[2])
        snap = transpose(data[i, channel, :, :])
        heatmap(x, z, snap,
            c=:jet, clims=clims, aspect_ratio=:equal, xlim=(0, L[1]), ylim=(0, L[2]),
            title="2D Rayleigh-Bénard $(channel_name) (t=$t)")
    end

    animation_path = joinpath(animation_dir, "$(channel_name).mp4")
    anim = @animate for i ∈ 1:size(data, 1)
        show_snapshot(i)
    end
    mp4(anim, animation_path, fps=fps)
end