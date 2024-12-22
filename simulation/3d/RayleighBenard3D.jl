#? run: cd simulation -> julia -> ] -> activate . -> backspace -> include("RayleighBenard3D.jl") -> simulate_3d_rb()

using Printf
import Random
using Oceananigans
using Statistics
using HDF5
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
L = (2*pi, 2*pi, 2) # x,y,z

# number of discrete sampled points
N = (48, 48, 32)

# time
Δt = 0.01 # simulation delta
Δt_snap = 0.3 # save delta
duration = 300 # duration of simulation

# temperature
min_b = 0 # Temperature at top plate
Δb = 1 # Temperature difference between bottom and top plate

# Rayleigh Benard Parameters
Ra = 10^4
Pr = 0.7

# Set the amplitude of the random initial perturbation (kick)
random_kick = 0.2


function simulate_3d_rb(; random_inits=1, Ra=Ra, Pr=Pr, N=N, L=L, min_b=min_b, Δb=Δb, random_kick=random_kick,
    Δt=Δt, Δt_snap=Δt_snap, duration=duration, use_gpu=use_gpu, visualize=visualize, fps=fps)

    ν = sqrt(Pr / Ra) # c.f. line 33: https://github.com/spectralDNS/shenfun/blob/master/demo/RayleighBenard.py
    κ = 1 / sqrt(Pr * Ra) # c.f. line 37: https://github.com/spectralDNS/shenfun/blob/master/demo/RayleighBenard.py

    # simulation is done in free-flow time units
    # t_ff = H/U_ff = H/sqrt(gαΔTH) = H/(1/H) = H^2
    # since computation of ν,κ assumes that gαΔTH^3=1 ⇔ sqrt(gαΔTH) = 1/H
    t_ff = L[3]^2

    totalsteps = Int(div(duration, Δt_snap * t_ff))

    grid = define_sample_grid(N, L, use_gpu)
    u_bcs, v_bcs, b_bcs = define_boundary_conditions(min_b, Δb)
    
    for i ∈ 1:random_inits
        println("Simulating random initialization $(i)/$(random_inits)...")

        simulation_name = "$(N[1])_$(N[2])_$(N[3])_$(Ra)_$(Pr)_$(Δt)_$(Δt_snap)_$(duration)"
        h5_file, dataset, h5_file_path, sim_num = create_hdf5_dataset(simulation_name, N, totalsteps)

        # Make sure that every random initialization is indeed independend of each other
        # (even when script is restarted)
        Random.seed!(sim_num)

        model = define_model(grid, ν, κ, u_bcs, v_bcs, b_bcs)
        initialize_model(model, min_b, L[3], Δb, random_kick)
        
        simulate_model(model, dataset, Δt, t_ff, Δt_snap, totalsteps, N)

        if visualize
            animation_dir = joinpath(dirpath, "data", simulation_name, "sim$(sim_num)", "animations")
            mkpath(animation_dir)
            for (channel_num, channel_name) in enumerate(["temp", "u", "v", "w"])
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
        grid = RectilinearGrid(GPU(), size=N, x=(0, L[1]), y=(0, L[2]), z=(0, L[3]),
                               topology=(Periodic, Periodic, Bounded))
    else
        grid = RectilinearGrid(size=(N), x=(0, L[1]), y=(0, L[2]), z=(0, L[3]), 
                               topology=(Periodic, Periodic, Bounded))
    end
    return grid
end


function define_boundary_conditions(min_b, Δb)
    u_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(0),
        bottom=ValueBoundaryCondition(0))
    v_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(0),
        bottom=ValueBoundaryCondition(0))
    # no explicit boundary condition for vertical velocity (seems to be inferred automatically)
    # w_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(0),
    #                                 bottom = ValueBoundaryCondition(0))
    b_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(min_b),
        bottom=ValueBoundaryCondition(min_b + Δb))
    return u_bcs, v_bcs, b_bcs
end


function define_model(grid, ν, κ, u_bcs, v_bcs, b_bcs)
    model = NonhydrostaticModel(; grid,
        advection=UpwindBiasedFifthOrder(),
        timestepper=:RungeKutta3,
        tracers=(:b),
        buoyancy=Buoyancy(model=BuoyancyTracer()),
        closure=(ScalarDiffusivity(ν=ν, κ=κ)),
        boundary_conditions=(u=u_bcs, v=v_bcs, b=b_bcs,),
        coriolis=nothing
    )
    return model
end


function initialize_model(model, min_b, Lz, Δb, kick)
    # Set initial conditions
    uᵢ(x, y, z) = kick * randn()
    vᵢ(x, y, z) = kick * randn()
    wᵢ(x, y, z) = kick * randn()
    bᵢ(x, y, z) = min_b + (Lz - z) * Δb / 2 + kick * randn()

    # Send the initial conditions to the model to initialize the variables
    set!(model, u=uᵢ, v=vᵢ, w=wᵢ, b=bᵢ)
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
        dataspace(totalsteps + 1, 4, N...), chunk=(1, 1, N...))

    # seperate datasets for temperature and velocity:
    # temps = create_dataset(h5_file, "temperature", datatype(Float64),
    #     dataspace(totalsteps + 1, N...), chunk=(1, N...))
    # vels = create_dataset(h5_file, "velocity", datatype(Float64),
    #     dataspace(totalsteps + 1, 3, N...), chunk=(1, 1, N...))

    return h5_file, dataset, path, sim_num
end


function simulate_model(model, dataset, Δt, t_ff, Δt_snap, totalsteps, N)
    simulation = Simulation(model, Δt=Δt, stop_time=Δt_snap*t_ff)
    simulation.verbose = true

    cur_time = 0.0

    # save initial state
    save_simulation_step(model, dataset, 1, N)

    for i in 1:totalsteps
        #update the simulation stop time for the next step (in free fall time units)
        global simulation.stop_time = Δt_snap*t_ff * i

        run!(simulation)
        cur_time += Δt_snap*t_ff

        save_simulation_step(model, dataset, i + 1, N)

        if (step_contains_NaNs(model, N))
            printstyled("[ERROR] NaN values found!\n"; color=:red)
            return
        end

        println(cur_time)
    end
end


function save_simulation_step(model, dataset, step, N)
    dataset[step, 1, :, :, :] = model.tracers.b[1:N[1], 1:N[2], 1:N[3]]
    dataset[step, 2, :, :, :] = model.velocities.u[1:N[1], 1:N[2], 1:N[3]]
    dataset[step, 3, :, :, :] = model.velocities.v[1:N[1], 1:N[2], 1:N[3]]
    dataset[step, 4, :, :, :] = model.velocities.w[1:N[1], 1:N[2], 1:N[3]]
end


function step_contains_NaNs(model, N)
    contains_nans = (any(isnan, model.tracers.b[1:N[1], 1:N[2], 1:N[3]]) ||
                     any(isnan, model.velocities.u[1:N[1], 1:N[2], 1:N[3]]) ||
                     any(isnan, model.velocities.v[1:N[1], 1:N[2], 1:N[3]]) ||
                     any(isnan, model.velocities.w[1:N[1], 1:N[2], 1:N[3]]))
    return contains_nans
end


function visualize_simulation(data, animation_dir, channel, channel_name, fps, N, L, Δt_snap, min_b, Δb)
    width = div(N[1], 2)
    depth = div(N[2], 2)
    height = div(N[3], 2)

    if channel == 1 # temperature channel
        clims_width = (min_b, min_b + Δb)
        clims_depth = (min_b, min_b + Δb)
        clims_height = (min_b, min_b + Δb)
    else
        total_min = min(minimum(data[:, channel, width, :, :]), 
                        minimum(data[:, channel, :, depth, :]), 
                        minimum(data[:, channel, :, :, height]))
        total_max = max(maximum(data[:, channel, width, :, :]),
                        maximum(data[:, channel, :, depth, :]),
                        maximum(data[:, channel, :, :, height]))
        extreme_value = max(abs(total_min), abs(total_max))

        # use same color map for all animations and center limits around zero
        clims_width = (-extreme_value, extreme_value)
        clims_depth = (-extreme_value, extreme_value)
        clims_height = (-extreme_value, extreme_value)
    end

    function show_snapshot_width(i)
        t = round((i - 1) * Δt_snap, digits=1)
        x = range(0, L[2], length=N[2])
        z = range(0, L[3], length=N[3])
        snap = transpose(data[i, channel, width, :, :])
        heatmap(x, z, snap,
            c=:jet, clims=clims_width, aspect_ratio=:equal, xlim=(0, L[2]), ylim=(0, L[3]),
            title="Side View (w=$(width)): 3D Rayleigh-Bénard $(channel_name) (t=$t)")
    end
    function show_snapshot_depth(i)
        t = round((i - 1) * Δt_snap, digits=1)
        x = range(0, L[1], length=N[1])
        z = range(0, L[3], length=N[3])
        snap = transpose(data[i, channel, :, depth, :])
        heatmap(x, z, snap,
            c=:jet, clims=clims_depth, aspect_ratio=:equal, xlim=(0, L[1]), ylim=(0, L[3]),
            title="Side View (d=$(depth)): 3D Rayleigh-Bénard $(channel_name) (t=$t)")
    end
    function show_snapshot_height(i)
        t = round((i - 1) * Δt_snap, digits=1)
        x = range(0, L[1], length=N[1])
        z = range(0, L[2], length=N[2])
        snap = transpose(data[i, channel, :, :, height])
        heatmap(x, z, snap,
            c=:jet, clims=clims_height, aspect_ratio=:equal, xlim=(0, L[1]), ylim=(0, L[2]),
            title="Top View (h=$(height)): 3D Rayleigh-Bénard $(channel_name) (t=$t)")
    end

    animation_path_width = joinpath(animation_dir, "$(channel_name)_width.mp4")
    anim_width = @animate for i ∈ 1:size(data, 1)
        show_snapshot_width(i)
    end
    mp4(anim_width, animation_path_width, fps=fps)

    animation_path_depth = joinpath(animation_dir, "$(channel_name)_depth.mp4")
    anim_depth = @animate for i ∈ 1:size(data, 1)
        show_snapshot_depth(i)
    end
    mp4(anim_depth, animation_path_depth, fps=fps)

    animation_path_height = joinpath(animation_dir, "$(channel_name)_height.mp4")
    anim_height = @animate for i ∈ 1:size(data, 1)
        show_snapshot_height(i)
    end
    mp4(anim_height, animation_path_height, fps=fps)
end