using Documenter, Ferrite, Pkg

Pkg.precompile()

const is_ci = haskey(ENV, "GITHUB_ACTIONS")

# Generate examples
include("generate.jl")

GENERATEDEXAMPLES = [joinpath("examples", f) for f in (
    "heat_equation.md",
    "postprocessing.md",
    "helmholtz.md",
    "incompressible_elasticity.md",
    "hyperelasticity.md",
    "threaded_assembly.md",
    "plasticity.md",
    "transient_heat_equation.md",
    "landau.md",
    "linear_shell.md",
    "quasi_incompressible_hyperelasticity.md",
    "ns_vs_diffeq.md",
    "computational_homogenization.md",
    "porous_media.md"
    )]

# Build documentation.
makedocs(
    format = Documenter.HTML(),
    sitename = "Ferrite.jl",
    doctest = false,
    # strict = VERSION.minor == 6 && sizeof(Int) == 8, # only strict mode on 0.6 and Int64
    strict = false,
    pages = Any[
        "Home" => "index.md",
        "manual/fe_intro.md",
        "Manual" => [
            "manual/degrees_of_freedom.md",
            "manual/assembly.md",
            "manual/boundary_conditions.md",
            "manual/constraints.md",
            "manual/grid.md",
            "manual/export.md"
            ],
        "Examples" => GENERATEDEXAMPLES,
        "API Reference" => [
            "reference/quadrature.md",
            "reference/interpolations.md",
            "reference/fevalues.md",
            "reference/dofhandler.md",
            "reference/assembly.md",
            "reference/boundary_conditions.md",
            "reference/grid.md",
            "reference/export.md"
            ]
        ],
)

# make sure there are no *.vtu files left around from the build
cd(joinpath(@__DIR__, "build", "examples")) do
    foreach(file -> endswith(file, ".vtu") && rm(file), readdir())
end


# Deploy built documentation
deploydocs(
    repo = "github.com/Ferrite-FEM/Ferrite.jl.git",
    push_preview=true,
)
