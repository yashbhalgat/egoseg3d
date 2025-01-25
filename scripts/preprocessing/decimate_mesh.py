import open3d as o3d

# input path of mesh
PATH_INPUT = "./scripts/reconstructions/sparse/P01_104/dense/poisson-output.ply"

# target number of triangles
NB_TRIANGLES = 20000

# output path of mesh
PATH_OUTPUT = PATH_INPUT.replace(".ply", f"-decimate{NB_TRIANGLES}.ply")

# Load the mesh
mesh = o3d.io.read_triangle_mesh(PATH_INPUT)

# Check if the mesh needs to have normals computed
if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()

# Apply quadric simplification
mesh_simplified = mesh.simplify_quadric_decimation(
    target_number_of_triangles=NB_TRIANGLES
)
o3d.io.write_triangle_mesh(PATH_OUTPUT, mesh_simplified)

print(f'Decimated mesh to {NB_TRIANGLES} triangles.')