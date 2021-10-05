include("l2_projection.jl");

points = [Vec((x, 0.75)) for x in range(-1.0, 1.0, length=101)];

ph = PointEvalHandler(dh, points);

q_points = Ferrite.get_point_values(ph, q_nodes);

u_points = Ferrite.get_point_values(ph, u, :u);

import Plots

Plots.plot(getindex.(points,1), u_points, label="Temperature", xlabel="X-Coordinate", ylabel = "Temperature")

Plots.plot(getindex.(points,1), getindex.(q_points,1),label="Flux", legend=:topleft, xlabel = "X-Coordinate", ylabel = "Heat flux")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

