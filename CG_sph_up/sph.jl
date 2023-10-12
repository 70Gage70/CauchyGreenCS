using CoherentStructures, Tensors, Interpolations

function CG_tensor_sph(odefun, u, tspan, δ; kwargs...)

    lf = linearized_flow(odefun, u, [tspan[1], tspan[end]], δ; kwargs...)
    u_final, dF = lf[1][end], lf[2][end]
	 G0inv = Tensor{2,2}([cos(u[2])^2 0.0; 0.0 1.0]^(-1/2)) #u[2] is the initial y (latitude, in radians)
	 G = Tensor{2,2}([cos(u_final[2])^2 0.0; 0.0 1.0]) #u_final[2] is the final y (latitude, in radians)
	 return Tensors.unsafe_symmetric(G0inv ⋅ dF' ⋅ G ⋅ dF ⋅ G0inv)

end

function interpolateVF_new(X::AbstractRange{S1},
    Y::AbstractRange{S1},
    T::AbstractRange{S1},
    U::AbstractArray{S2,3},
    V::AbstractArray{S2,3},
    itp_type=BSpline(Cubic(Free(OnGrid())))
    ) where {S1 <: Real, S2 <: Real}

   UV = map(SVector{2,S2}, U, V)::Array{SVector{2,S2},3}
   return extrapolate(scale(interpolate(UV, itp_type), X, Y, T), SVector{2}([0.0,0.0]))

end