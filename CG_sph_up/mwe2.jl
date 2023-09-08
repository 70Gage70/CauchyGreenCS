using Distributed, MAT, StaticArrays, AxisArrays, Dates, Plots, NetCDF, Interpolations

include("sph.jl")

@everywhere using CoherentStructures, OrdinaryDiffEq, Tensors

# read-in velocity data
file = matopen(joinpath(@__DIR__, "..", "data", "boyas-xyt.mat"))
x, y, t = read(file, "x", "y", "t")
close(file)

file = matopen(joinpath(@__DIR__, "..", "data", "boyas-u.mat"))
vx = read(file, "u")
vx = permutedims(vx, (2,1,3)) # Nx x Ny x Nt
replace!(vx, NaN => 0)
close(file)

file = matopen(joinpath(@__DIR__, "..", "data", "boyas-v.mat"))
vy = read(file, "v")
vy = permutedims(vy, (2,1,3))
replace!(vy, NaN => 0)
close(file)

# x,y,t ranges
xmin, xmax = extrema(x)
ymin, ymax = extrema(y)
tmin, tmax = extrema(t)
xrange = range(xmin, stop = xmax, length = length(x))
yrange = range(ymin, stop = ymax, length = length(y))
trange = range(tmin, stop = tmax, length = length(t))

# velocity interpolant
v = SVector{2}.(vx, vy);
#Fv = interpolateVF(xrange, yrange, trange, vx, vy) # old
Fv = interpolateVF_new(xrange, yrange, trange, vx, vy) # new

# dealing with land...
landx = dropdims(mapslices(all∘iszero, vx; dims=[3]); dims=3)
landy = dropdims(mapslices(all∘iszero, vy; dims=[3]); dims=3)
landxy = landx .& landy
land = map(Iterators.product(1:size(landxy, 1)-1, 1:size(landxy, 2)-1)) do (i, j)
	return landxy[i,j] & landxy[i+1,j] & landxy[i,j+1] & landxy[i+1,j+1]
end
dx, dy = step.((xrange, yrange))
is_land = let land=land, dx=dx, dy=dy, xmin=xmin, ymin=ymin
	x -> land[Int((x[1] - xmin)÷dx + 1), Int((x[2] - ymin)÷dy + 1)]
end

# x0,y0
x0min, x0max, y0min, y0max = xmin+5, xmax-5, ymin+5, ymax-5
Nx = 4 #512
Ny = floor(Int, (y0max - y0min) / (x0max - x0min) * Nx)
x0 = range(x0min, stop=x0max, length=Nx)
y0 = range(y0min, stop=y0max, length=Ny)
P = AxisArray(SVector{2}.(x0, y0'), x0, y0)

# t0
t0 = (Date(2021, 10, 10) - Date(0, 1, 1)).value + 1

# vortex parameters
known_centers = [Singularity((300,-250), 1)]
p = LCSParameters(
   boxradius = 50,
	n_seeds = 500,	
   pmin = .8,
   pmax = 2.5,
	rdist = 4e-5,
   merge_heuristics = []
   )
T = 19

# extract vortices
tspan = range(t0, stop=t0+T, length=2)

function vF(du, u, p, t)
	x, y = u
	du[1:2] .= Fv(x, y, t)
end

CG = let ts = tspan
	u -> is_land(u) ? one(SymmetricTensor{2,2}) :
	     CG_tensor(
	vF,
	u,
	ts,
	1e-3;
	tolerance = 1e-6,
	solver = Tsit5(),
	)
end

@btime map(CG, P)