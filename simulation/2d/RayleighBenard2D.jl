using Printf
import Random
using Oceananigans
using Statistics
using HDF5
using CUDA
using Plots
using ArgParse

theme(:dark)

# script directory
dirpath = string(@__DIR__)


function simulate_2d_rb(dir, random_inits, Ra, Pr, N, L, min_b, Δb, random_kick, Δt, Δt_snap, 
    duration, use_gpu, visualize, fps; sim_name=nothing)

    ν = sqrt(Pr / Ra) # c.f. line 33: https://github.com/spectralDNS/shenfun/blob/master/demo/RayleighBenard2D.py
    κ = 1 / sqrt(Pr * Ra) # c.f. line 37: https://github.com/spectralDNS/shenfun/blob/master/demo/RayleighBenard2D.py

    totalsteps = Int(div(duration, Δt_snap))

    grid = define_sample_grid(N, L, use_gpu)
    u_bcs, b_bcs = define_boundary_conditions(min_b, Δb)

    for i ∈ 1:random_inits
        println("Simulating random initialization $(i)/$(random_inits)...")

        if (isnothing(sim_name))
            sim_name = "x$(N[1])_z$(N[2])_Ra$(Ra)_Pr$(Pr)_t$(Δt)_snap$(Δt_snap)_dur$(duration)"
        end
        h5_file, dataset, h5_file_path, sim_num = create_hdf5_dataset(dir, sim_name, N, totalsteps)

        # Make sure that every random initialization is indeed independend of each other
        # (even when script is restarted)
        Random.seed!(sim_num)

        model = define_model(grid, ν, κ, u_bcs, b_bcs)
        initialize_model(model, min_b, L[2], Δb, random_kick)

        success = simulate_model(model, dataset, Δt, Δt_snap, totalsteps, N)

        if (!success)
            return
        end

        if visualize
            animation_dir = joinpath(dir, sim_name, "sim$(sim_num)", "animations")
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


function create_hdf5_dataset(dir, sim_name, N, totalsteps)
    sim_dir = joinpath(dir, sim_name)
    mkpath(sim_dir) # create directory if not existent

    # compute number of this simulation
    sim_num = 1
    while isfile(joinpath(sim_dir, "sim$(sim_num)", "sim.h5"))
        sim_num += 1
    end

    mkpath(joinpath(sim_dir, "sim$(sim_num)"))
    sim_path = joinpath(sim_dir, "sim$(sim_num)", "sim.h5")
    h5_file = h5open(sim_path, "w")
    # save temperature and velocities in one dataset:
    dataset = create_dataset(h5_file, "data", datatype(Float64),
        dataspace(totalsteps + 1, 3, N...), chunk=(1, 1, N...))

    return h5_file, dataset, sim_path, sim_num
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
            return false
        end

        println(cur_time)
    end

    return true
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
        clims = (min_b, min_b + Δb)
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


function parse_arguments()
    s = ArgParseSettings(description="Simulates 2D Rayleigh-Bénard.")

    @add_arg_table s begin
        "--sim_name"
        help = "The name of the simulation. Defaults to the following scheme \
        x<N[1]>_z<N[2]>_Ra<Ra>_Pr<Pr>_t<Δt>_snap<Δt_snap>_dur<duration>"
        default = nothing
        "--dir"
        help = "The path to the directory to store the simulations in."
        default = joinpath(dirpath, "data")
        "--random_inits"
        help = "The number of random initializations to simulate"
        arg_type = Int
        default = 1
        "--Ra"
        help = "Rayleigh number"
        arg_type = Int
        default = 10^5
        "--Pr"
        help = "Prandtl number"
        arg_type = Float64
        default = 0.7
        "--N"
        help = "The size of the grid [width, height]"
        arg_type = Int
        nargs = 2
        default = [128, 64]
        "--L"
        help = "The spatial dimensions of the domain [width, height]"
        arg_type = Float64
        nargs = 2
        default = [2 * pi, 2]
        "--min_b"
        help = "The temperature of the top plate"
        arg_type = Float64
        default = 0
        "--delta_b"
        help = "The temperature difference between the bottom and top plate"
        arg_type = Float64
        default = 1
        dest_name = "Δb"
        "--random_kick"
        help = "The amplitude of the random initial perturbation (kick)"
        arg_type = Float64
        default = 0.2
        "--delta_t"
        help = "Time delta between simulated steps"
        arg_type = Float64
        default = 0.01
        dest_name = "Δt"
        "--delta_t_snap"
        help = "Time delta between saved snapshots"
        arg_type = Float64
        default = 0.3
        dest_name = "Δt_snap"
        "--duration"
        help = "The duration of a simulation"
        arg_type = Int
        default = 300
        "--use_cpu"
        help = "Runs the simulation on CPU when argument given."
        action = :store_false
        dest_name = "use_gpu"
        "--no_visualization"
        help = "No animation of the simulated data is created when argument given."
        action = :store_false
        dest_name = "visualize"
        "--fps"
        help = "The fps of the animated simulation data"
        arg_type = Int
        default = 15
    end

    return parse_args(s)
end

is_script = abspath(PROGRAM_FILE) == @__FILE__
if (is_script)
    args = parse_arguments()
    simulate_2d_rb(
        args["dir"],
        args["random_inits"],
        args["Ra"],
        args["Pr"],
        args["N"],
        args["L"],
        args["min_b"],
        args["Δb"],
        args["random_kick"],
        args["Δt"],
        args["Δt_snap"],
        args["duration"],
        args["use_gpu"],
        args["visualize"],
        args["fps"],
        sim_name=args["sim_name"],)
end