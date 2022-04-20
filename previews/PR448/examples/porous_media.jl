using Ferrite, FerriteMeshParser, Tensors

function elastic_stiffness(E=20.e3, ν=0.3)
    G = E / 2(1 + ν)
    K = E / 3(1 - 2ν)
    I2 = one(SymmetricTensor{2,2})
    I4vol = I2⊗I2
    I4dev = minorsymmetric(otimesu(I2,I2)) - I4vol / 3
    return 2G*I4dev + K*I4vol
end

function element_routine!(Ke, _, cell, _, cv::CellVectorValues, _, _)
    reinit!(cv, cell)
    n_basefuncs = getnbasefunctions(cv)
    fill!(Ke, 0)
    dσdϵ = elastic_stiffness()
    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)
        for i in 1:n_basefuncs
            δ∇N = shape_symmetric_gradient(cv, q_point, i)
            for j in 1:n_basefuncs
                ∇N = shape_symmetric_gradient(cv, q_point, j)
                Ke[i, j] += δ∇N ⊡ dσdϵ ⊡ ∇N * dΩ
            end
        end
    end
end

function element_routine!(Ke, fext, cell, fh::FieldHandler, cvs::Tuple{CellVectorValues, CellScalarValues}, a_old, Δt)
    # Setup cellvalues and give easier names
    reinit!.(cvs, (cell,))
    cv_u, cv_p = cvs
    num_u_basefuncs, num_p_basefuncs = getnbasefunctions.(cvs)

    # Check that cellvalues are compatible with each other (should have same quadrature rule)
    @assert getnquadpoints(cv_u) == getnquadpoints(cv_p)

    # Reset element stiffness and external force
    fill!(Ke, 0.0)
    fill!(fext, 0.0)

    # Assign views to the matrix and vector parts
    udofs = dof_range(fh, :u)
    pdofs = dof_range(fh, :p)
    Kuu = @view Ke[udofs, udofs]
    Kpu = @view Ke[pdofs, udofs]
    Kup = @view Ke[udofs, pdofs]
    Kpp = @view Ke[pdofs, pdofs]
    # fu = @view fext[udofs]    # Not used, traction is zero or displacements prescribed
    fp = @view fext[pdofs]
    au_old = @view a_old[udofs]
    ap_old = @view a_old[pdofs]

    # Material parameters
    μ = 1.e-4       # [Ns/mm^2] Dynamic viscosity
    k = 5.0e-6      # [mm^2] Intrinsic permeability
    k_darcy = k/μ
    n = 0.8         # [-] Porosity
    K_liquid = 2.e3 # [MPa] Liquid bulk modulus
    dσdϵ = elastic_stiffness()

    # Assemble stiffness and force vectors
    for q_point in 1:getnquadpoints(cv_u)
        dΩ = getdetJdV(cv_u, q_point)
        # Variation of u_i
        for i in 1:num_u_basefuncs
            ∇δNu = shape_symmetric_gradient(cv_u, q_point, i)
            div_δNu = shape_divergence(cv_u, q_point, i)
            for j in 1:num_u_basefuncs
                ∇Nu = shape_symmetric_gradient(cv_u, q_point, j)
                Kuu[i, j] -= ∇δNu ⊡ dσdϵ ⊡ ∇Nu * dΩ
            end
            for j in 1:num_p_basefuncs
                Np = shape_value(cv_p, q_point, j)
                Kup[i, j] += div_δNu * Np
            end
        end
        # Variation of p_i
        for i in 1:num_p_basefuncs
            δNp = shape_value(cv_p, q_point, i)
            ∇δNp = shape_gradient(cv_p, q_point, i)
            for j in 1:num_u_basefuncs
                div_Nu = shape_divergence(cv_u, q_point, j)
                Lpu_ij = δNp*div_Nu*dΩ
                Kpu[i,j] += Lpu_ij
                fp[i] += Lpu_ij*au_old[j]
            end
            for j in 1:num_p_basefuncs
                ∇Np = shape_gradient(cv_p, q_point, j)
                Np = shape_value(cv_p, q_point, j)
                Kpp_ij = (k_darcy/n) * ∇δNp ⋅ ∇Np * dΩ
                Lpp_ij = δNp*Np/K_liquid
                Kpp[i,j] += Δt*Kpp_ij + Lpp_ij
                fp[i] += Lpp_ij*ap_old[j]
            end
        end
    end
end

function doassemble!(K, f, dh, cvs, a_old, Δt)
    assembler = start_assemble(K, f)
    for (fh, cv) in zip(dh.fieldhandlers, cvs)
        doassemble!(assembler, dh, cv, fh, a_old, Δt)
    end
end

function doassemble!(assembler, dh, cv, fh::FieldHandler, a_old, Δt)
    n = ndofs_per_cell(dh, first(fh.cellset))
    Ke = zeros(n,n)
    fe = zeros(n)
    for cell in CellIterator(dh, collect(fh.cellset))
        element_routine!(Ke, fe, cell, fh, cv, a_old[celldofs(cell)], Δt)
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
end

function get_grid()
    # Import grid from abaqus mesh
    grid = get_ferrite_grid(joinpath(@__DIR__, "porous_media_0p25.inp"))

    # Create cellsets for each fieldhandler
    addcellset!(grid, "solid3", intersect(getcellset(grid, "solid"), getcellset(grid, "CPS3")))
    addcellset!(grid, "solid4", intersect(getcellset(grid, "solid"), getcellset(grid, "CPS4R")))
    addcellset!(grid, "porous3", intersect(getcellset(grid, "porous"), getcellset(grid, "CPS3")))
    addcellset!(grid, "porous4", intersect(getcellset(grid, "porous"), getcellset(grid, "CPS4R")))
    return grid
end

function setup_problem(;t_rise=0.1, p_max=100.0)

    grid = get_grid()

    # Setup the interpolation and integration rules
    dim=Ferrite.getdim(grid)
    ip3_lin = Lagrange{dim, RefTetrahedron, 1}()
    ip4_lin = Lagrange{dim, RefCube, 1}()
    ip3_quad = Lagrange{dim, RefTetrahedron, 2}()
    ip4_quad = Lagrange{dim, RefCube, 2}()
    qr3 = QuadratureRule{dim, RefTetrahedron}(1)
    qr4 = QuadratureRule{dim, RefCube}(2)

    # Setup the MixedDofHandler
    dh = MixedDofHandler(grid)
    push!(dh, FieldHandler([Field(:u, ip3_lin, dim)], getcellset(grid,"solid3")))
    push!(dh, FieldHandler([Field(:u, ip4_lin, dim)], getcellset(grid,"solid4")))
    push!(dh, FieldHandler([Field(:u, ip3_quad, dim), Field(:p, ip3_lin, 1)], getcellset(grid,"porous3")))
    push!(dh, FieldHandler([Field(:u, ip4_quad, dim), Field(:p, ip4_lin, 1)], getcellset(grid,"porous4")))
    close!(dh)

    # Setup cellvalues with the same order as the FieldHandlers in the dh
    # - Linear displacement elements in the solid domain
    # - Taylor hood (quadratic displacement, linear pressure) and linear geometry in porous domain
    cv = ( CellVectorValues(qr3, ip3_lin),
           CellVectorValues(qr4, ip4_lin),
           (CellVectorValues(qr3, ip3_quad, ip3_lin), CellScalarValues(qr3, ip3_lin)),
           (CellVectorValues(qr4, ip4_quad, ip4_lin), CellScalarValues(qr4, ip4_lin)) )

    # Add boundary conditions, use code from PR427.jl to make more general
    ch = ConstraintHandler(dh);
    # With #PR427 (keep code for if/when it is merged)
    # add!(ch, Dirichlet(:u, getfaceset(grid, "bottom"), (x, t) -> zero(Vec{2}), [1,2]))
    # add!(ch, Dirichlet(:p, getfaceset(grid, "bottom_p"), (x, t) -> 0.0))
    # add!(ch, Dirichlet(:p, getfaceset(grid, "top_p"), (x, t) -> p_max*clamp(t/t_rise,0,1)))
    # With master (only works if no tri-elements on boundary)
    add!(ch, dh.fieldhandlers[2], Dirichlet(:u, getfaceset(grid, "bottom"), (x, t) -> zero(Vec{2}), [1,2]))
    add!(ch, dh.fieldhandlers[4], Dirichlet(:u, getfaceset(grid, "bottom_p"), (x, t) -> zero(Vec{2}), [1,2]))
    add!(ch, dh.fieldhandlers[4], Dirichlet(:p, getfaceset(grid, "bottom_p"), (x, t) -> 0.0))
    add!(ch, dh.fieldhandlers[4], Dirichlet(:p, getfaceset(grid, "top_p"), (x, t) -> p_max*clamp(t/t_rise,0,1)))
    close!(ch)
    return dh, ch, cv
end

function solve(dh, ch, cv; Δt=0.025, t_total=1.0)
    # Assemble stiffness matrix
    K = create_sparsity_pattern(dh);
    f = zeros(ndofs(dh))
    a = zeros(ndofs(dh))
    pvd = paraview_collection("porous_media.pvd");
    for (step, t) = enumerate(0:Δt:t_total)
        if t>0
            doassemble!(K, f, dh, cv, a, Δt)
            update!(ch, t)
            apply!(K, f, ch)
            a .= K\f
        end
        vtk_grid("porous_media-$step", dh) do vtk
            vtk_point_data(vtk, dh, a)
            vtk_save(vtk)
            pvd[step] = vtk
        end
    end
    vtk_save(pvd);
end

dh, ch, cv = setup_problem()
solve(dh, ch, cv)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

