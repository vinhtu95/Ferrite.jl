using Ferrite, FerriteGmsh, Tensors, LinearAlgebra

function setup_grid(h=0.05)
    # Initialize gmsh
    gmsh.initialize()

    # Add the points
    o = gmsh.model.geo.add_point(0.0, 0.0, 0.0, h)
    p1 = gmsh.model.geo.add_point(0.5, 0.0, 0.0, h)
    p2 = gmsh.model.geo.add_point(1.0, 0.0, 0.0, h)
    p3 = gmsh.model.geo.add_point(0.0, 1.0, 0.0, h)
    p4 = gmsh.model.geo.add_point(0.0, 0.5, 0.0, h)

    # Add the lines
    l1 = gmsh.model.geo.add_line(p1, p2)
    l2 = gmsh.model.geo.add_circle_arc(p2, o, p3)
    l3 = gmsh.model.geo.add_line(p3, p4)
    l4 = gmsh.model.geo.add_circle_arc(p4, o, p1)

    # Create the closed curve loop and the surface
    loop = gmsh.model.geo.add_curve_loop([l1, l2, l3, l4])
    surf = gmsh.model.geo.add_plane_surface([loop])

    # Synchronize the model
    gmsh.model.geo.synchronize()

    # Create the physical domains
    gmsh.model.add_physical_group(1, [l1], -1, "right")
    gmsh.model.add_physical_group(1, [l2], -1, "outer")
    gmsh.model.add_physical_group(1, [l3], -1, "left")
    gmsh.model.add_physical_group(1, [l4], -1, "inner")
    gmsh.model.add_physical_group(2, [surf])

    # Add the periodicity constraint using 4x4 affine transformation matrix,
    # see https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations
    transformation_matrix = zeros(4, 4)
    transformation_matrix[1, 2] = 1  # -sin(-pi/2)
    transformation_matrix[2, 1] = -1 #  cos(-pi/2)
    transformation_matrix[3, 3] = 1
    transformation_matrix[4, 4] = 1
    transformation_matrix = vec(transformation_matrix')
    gmsh.model.mesh.set_periodic(1, [l1], [l3], transformation_matrix)

    # Generate a 2D mesh
    gmsh.model.mesh.generate(2)

    # Save the mesh and read back in as a Ferrite Grid
    grid = mktempdir() do dir
        path = joinpath(dir, "mesh.msh")
        gmsh.write(path)
        saved_file_to_grid(path)
    end

    return grid
end

function setup_dofs(grid, ipu, ipp)
    dh = DofHandler(grid)
    push!(dh, :u, 2, ipu)
    push!(dh, :p, 1, ipp)
    close!(dh)
    return dh
end

function setup_fevalues(ipu, ipp, ipg)
    qr = QuadratureRule{2,RefTetrahedron}(2)
    cvu = CellVectorValues(qr, ipu, ipg)
    cvp = CellScalarValues(qr, ipp, ipg)
    return cvu, cvp
end

function setup_constraints(dh)
    ch = ConstraintHandler(dh)
    # Periodic BC
    R = rotation_tensor(-pi/2)
    periodic_faces = collect_periodic_faces(grid, "left", "right", x -> R ⋅ x)
    periodic = PeriodicDirichlet(:u, periodic_faces, nothing, [1, 2], R)
    add!(ch, periodic)
    # Dirichlet BC
    set = union(getfaceset(dh.grid, "inner"), getfaceset(dh.grid, "outer"))
    dbc = Dirichlet(:u, set, (x, t) -> 0*x, [1, 2])
    add!(ch, dbc)
    # Finalize
    close!(ch)
    update!(ch, 0)
    return ch
end


function assemble_system!(K, f, dh, cvu, cvp)
    assembler = start_assemble(K, f)
    ke = zeros(ndofs_per_cell(dh), ndofs_per_cell(dh))
    fe = zeros(ndofs_per_cell(dh))
    range_u = dof_range(dh, :u)
    ndofs_u = length(range_u)
    range_p = dof_range(dh, :p)
    ndofs_p = length(range_p)
    ϕᵤ = Vector{Vec{2,Float64}}(undef, ndofs_u)
    ∇ϕᵤ = Vector{Tensor{2,2,Float64,4}}(undef, ndofs_u)
    divϕᵤ = Vector{Float64}(undef, ndofs_u)
    ϕₚ = Vector{Float64}(undef, ndofs_p)
    for cell in CellIterator(dh)
        reinit!(cvu, cell)
        reinit!(cvp, cell)
        ke .= 0
        fe .= 0
        for qp in 1:getnquadpoints(cvu)
            dΩ = getdetJdV(cvu, qp)
            for i in 1:ndofs_u
                ϕᵤ[i] = shape_value(cvu, qp, i)
                ∇ϕᵤ[i] = shape_gradient(cvu, qp, i)
                divϕᵤ[i] = shape_divergence(cvu, qp, i)
            end
            for i in 1:ndofs_p
                ϕₚ[i] = shape_value(cvp, qp, i)
            end
            # u-u
            for (i, I) in pairs(range_u), (j, J) in pairs(range_u)
                ke[I, J] += ( ∇ϕᵤ[i] ⊡ ∇ϕᵤ[j] ) * dΩ
            end
            # u-p
            for (i, I) in pairs(range_u), (j, J) in pairs(range_p)
                ke[I, J] += ( - divϕᵤ[i] * ϕₚ[j] ) * dΩ
            end
            # p-u
            for (i, I) in pairs(range_p), (j, J) in pairs(range_u)
                ke[I, J] += ( - divϕᵤ[j] * ϕₚ[i] ) * dΩ
            end
            # rhs
            for (i, I) in pairs(range_u)
                x = spatial_coordinate(cvu, qp, getcoordinates(cell))
                b = exp(-100 * norm(x - Vec{2}((0.75, 0.1)))^2)
                bv = Vec{2}((b, 0.0))
                fe[I] += (ϕᵤ[i] ⋅ bv) * dΩ
            end
        end
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    return K, f
end

function main()
    # Grid
    h = 0.05
    grid = setup_grid(h)
    # Interpolations
    ipu = Lagrange{2,RefTetrahedron,2}()
    ipp = Lagrange{2,RefTetrahedron,1}()
    # Dofs
    dh = setup_dofs(grid, ipu, ipp)
    # Boundary conditions
    ch = setup_constraints(dh)
    # FE values
    ipg = Lagrange{2,RefTetrahedron,1}() # linear geometric interpolation
    cvu, cvp = setup_fevalues(ipu, ipp, ipg)
    # Global tangent matrix and rhs
    K = create_sparsity_pattern(dh, ch)
    f = zeros(ndofs(dh))
    # Assemble system
    assemble_system!(K, f, dh, cvu, cvp)
    # Apply boundary conditions and solve
    apply!(K, f, ch)
    u = K \ f
    apply!(u, ch)
    # Export the solution
    vtk_grid("step-45", grid) do vtk
        vtk_point_data(vtk, dh, u)
    end
end
