using Ferrite, SparseArrays, BlockArrays, OrdinaryDiffEq, LinearAlgebra, UnPack

x_cells = 220
y_cells = 41
grid = generate_grid(Quadrilateral, (x_cells, y_cells), Vec{2}((0.0, 0.0)), Vec{2}((2.2, 0.41)));

cell_indices = filter(ci->norm(mean(map(i->grid.nodes[i].x-[0.2,0.2], Ferrite.vertices(grid.cells[ci]))))>0.05, 1:length(grid.cells))
hole_cell_indices = filter(ci->norm(mean(map(i->grid.nodes[i].x-[0.2,0.2], Ferrite.vertices(grid.cells[ci]))))<=0.05, 1:length(grid.cells))

hole_face_ring = Set{FaceIndex}()
for hci ∈ hole_cell_indices
    push!(hole_face_ring, FaceIndex((hci+1, 4)))
    push!(hole_face_ring, FaceIndex((hci-1, 2)))
    push!(hole_face_ring, FaceIndex((hci-x_cells, 3)))
    push!(hole_face_ring, FaceIndex((hci+x_cells, 1)))
end
grid.facesets["hole"] = Set(filter(x->x.idx[1] ∉ hole_cell_indices, collect(hole_face_ring)))

cell_indices_map = map(ci->norm(mean(map(i->grid.nodes[i].x-[0.2,0.2], Ferrite.vertices(grid.cells[ci]))))>0.05 ? indexin([ci], cell_indices)[1] : 0, 1:length(grid.cells))
grid.cells = grid.cells[cell_indices]
for facesetname in keys(grid.facesets)
    grid.facesets[facesetname] = Set(map(fi -> FaceIndex( cell_indices_map[fi.idx[1]] ,fi.idx[2]), collect(grid.facesets[facesetname])))
end

dim = 2
T = 5
Δt₀ = 0.01
Δt_save = 0.05

ν = 0.001 #dynamic viscosity
vᵢₙ(t) = 1.0 #inflow velocity

ip_v = Lagrange{dim, RefCube, 2}()
ip_geom = Lagrange{dim, RefCube, 1}()
qr_v = QuadratureRule{dim, RefCube}(3)
cellvalues_v = CellVectorValues(qr_v, ip_v, ip_geom);

ip_p = Lagrange{dim, RefCube, 1}()
#Note that the pressure term comes in combination with a higher order test function...
qr_p = qr_v
cellvalues_p = CellScalarValues(qr_p, ip_p);

dh = DofHandler(grid)
push!(dh, :v, dim, ip_v)
push!(dh, :p, 1, ip_p)
close!(dh);

M = create_sparsity_pattern(dh);
K = create_sparsity_pattern(dh);

ch = ConstraintHandler(dh);

∂Ω_noslip = union(getfaceset.((grid, ), ["top", "bottom", "hole"])...);
∂Ω_inflow = getfaceset(grid, "left");
∂Ω_free = getfaceset(grid, "right");

noslip_bc = Dirichlet(:v, ∂Ω_noslip, (x, t) -> [0,0], [1,2])
add!(ch, noslip_bc);
inflow_bc = Dirichlet(:v, ∂Ω_inflow, (x, t) -> [clamp(t, 0.0, 1.0)*4*vᵢₙ(t)*x[2]*(0.41-x[2])/0.41^2,0], [1,2])
add!(ch, inflow_bc);

close!(ch)
update!(ch, 0.0);

function assemble_linear(cellvalues_v::CellVectorValues{dim}, cellvalues_p::CellScalarValues{dim}, ν, M::SparseMatrixCSC, K::SparseMatrixCSC, dh::DofHandler) where {dim}

    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p
    v▄, p▄ = 1, 2
    Me = PseudoBlockArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])
    Ke = PseudoBlockArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])

    f = zeros(ndofs(dh))
    stiffness_assembler = start_assemble(K)
    mass_assembler = start_assemble(M)

    @inbounds for cell in CellIterator(dh)

        fill!(Me, 0)
        fill!(Ke, 0)

        Ferrite.reinit!(cellvalues_v, cell)
        Ferrite.reinit!(cellvalues_p, cell)

        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)
            #Mass term
            for i in 1:n_basefuncs_v
                v = shape_value(cellvalues_v, q_point, i)
                for j in 1:n_basefuncs_v
                    φ = shape_value(cellvalues_v, q_point, j)
                    Me[BlockIndex((v▄, v▄),(i, j))] += (v ⋅ φ) * dΩ
                end
            end

            #Viscosity term
            for i in 1:n_basefuncs_v
                ∇v = shape_gradient(cellvalues_v, q_point, i)
                for j in 1:n_basefuncs_v
                    ∇φ = shape_gradient(cellvalues_v, q_point, j)
                    Ke[BlockIndex((v▄, v▄), (i, j))] -= ν * (∇v ⊡ ∇φ) * dΩ
                end
            end
            #Incompressibility term
            for i in 1:n_basefuncs_p
                ψ = shape_value(cellvalues_p, q_point, i)
                for j in 1:n_basefuncs_v
                    divv = shape_divergence(cellvalues_v, q_point, j)
                    Ke[BlockIndex((p▄, v▄), (i, j))] += (ψ * divv) * dΩ
                end
            end
            #Pressure term
            dΩ = getdetJdV(cellvalues_p, q_point)
            for i in 1:n_basefuncs_v
                divφ = shape_divergence(cellvalues_v, q_point, i)
                for j in 1:n_basefuncs_p
                    p = shape_value(cellvalues_p, q_point, j)
                    Ke[BlockIndex((v▄, p▄), (i, j))] += (p * divφ) * dΩ
                end
            end
        end

        assemble!(stiffness_assembler, celldofs(cell), Ke)
        assemble!(mass_assembler, celldofs(cell), Me)
    end
    return M, K
end

M, K = assemble_linear(cellvalues_v, cellvalues_p, ν, M, K, dh);

function OrdinaryDiffEq.initialize!(nlsolver::OrdinaryDiffEq.NLSolver{<:NLNewton,true}, integrator)
    @unpack u,uprev,t,dt,opts = integrator
    @unpack z,tmp,cache = nlsolver
    @unpack weight = cache

    cache.invγdt = inv(dt * nlsolver.γ)
    cache.tstep = integrator.t + nlsolver.c * dt
    OrdinaryDiffEq.calculate_residuals!(weight, fill!(weight, one(eltype(u))), uprev, u,
                         opts.abstol, opts.reltol, opts.internalnorm, t);

    update!(ch, t);

    apply!(uprev, ch)
    apply!(tmp, ch)
    apply_zero!(z, ch);

    nothing
end

mutable struct FerriteLinSolve{CH,F}
    ch::CH
    factorization::F
    A
end
FerriteLinSolve(ch) = FerriteLinSolve(ch,lu,nothing)
function (p::FerriteLinSolve)(::Type{Val{:init}},f,u0_prototype)
    FerriteLinSolve(ch)
end
function (p::FerriteLinSolve)(x,A,b,update_matrix=false;reltol=nothing, kwargs...)
    if update_matrix
        # Apply Dirichlet BCs
        apply_zero!(A, b, p.ch)
        # Update factorization
        p.A = p.factorization(A)
    end
    ldiv!(x, p.A, b)
    apply_zero!(x, p.ch)
    return nothing
end

u₀ = zeros(ndofs(dh))
apply!(u₀, ch)

jac_sparsity = sparse(K)
function navierstokes!(du,u,p,t)
    K,dh,cellvalues = p
    du .= K * u

    n_basefuncs = getnquadpoints(cellvalues)

    # Nonlinaer contribution
    for cell in CellIterator(dh)
        # Trilinear form evaluation
        v_celldofs = celldofs(cell)
        Ferrite.reinit!(cellvalues, cell)
        v_cell = u[v_celldofs[dof_range(dh, :v)]]
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            v_div = function_divergence(cellvalues, q_point, v_cell)
            v_val = function_value(cellvalues, q_point, v_cell)
            nl_contrib = - v_div * v_val
            for j in 1:n_basefuncs
                Nⱼ = shape_value(cellvalues, q_point, j)
                du[v_celldofs[j]] += nl_contrib ⋅ Nⱼ * dΩ
            end
        end
    end
end;
rhs = ODEFunction(navierstokes!, mass_matrix=M; jac_prototype=jac_sparsity)
p = [K, dh, cellvalues_v]
problem = ODEProblem(rhs, u₀, (0.0,T), p);

sol = solve(problem, MEBDF2(linsolve=FerriteLinSolve(ch)), progress=true, progress_steps=1, dt=Δt₀, saveat=Δt_save, initializealg=NoInit());

pvd = paraview_collection("vortex-street.pvd");

for (solution,t) in zip(sol.u, sol.t)
    #compress=false flag because otherwise each vtk file will be stored in memory
    vtk_grid("vortex-street-$t.vtu", dh; compress=false) do vtk
        vtk_point_data(vtk,dh,solution)
        vtk_save(vtk)
        pvd[t] = vtk
    end
end
vtk_save(pvd);

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

