import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# TODO: FOV, use cameras?
fov_angle = np.radians(60)  
max_viewing_distance = 10.0  

# Step 2: Calculate Surface Normals
mesh = trimesh.load('./meshes/Boeing747_400_M.dae', force='mesh')
face_normals = mesh.face_normals

def sample_viewpoints(mesh, num_samples,max_distance):
    points, face_index = trimesh.sample.sample_surface(mesh, num_samples)
    viewpoints=[]
    for x in range(len(points)):
        viewpoints.append([face_index[x],points[x,0]+max_distance*face_normals[x,0],points[x,1]+max_distance*face_normals[x,1],points[x,2]+max_distance*face_normals[x,2],face_normals[x,0],face_normals[x,1],face_normals[x,2]])
    return(np.array(viewpoints))

# Step 3: Compute Angles for Viewpoint Sampling
def get_pitch_yaw(n):
    return(np.vstack((np.zeros(n.shape[0]),np.arctan(n[2:],np.sqrt(np.square(n[0:])+np.square(n[1:]))),np.arctan(-n[1,:],-n[0:]))).T)

def get_areas_vertices(p,n):
    pass

num_samples = 100
sampled_viewpoints = sample_viewpoints(mesh, num_samples,5)

# Visualization Function
def visualize_viewpoints_with_model(mesh, viewpoints_with_areas):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mesh.show(scene=ax)
    x_coords, y_coords, z_coords = zip(*[point for point, _, _ in viewpoints_with_areas])
    areas = [area for _, _, area in viewpoints_with_areas]
    ax.scatter(x_coords, y_coords, z_coords, c=areas, cmap='viridis', s=50)
    for viewpoint, _, _ in viewpoints_with_areas:
        fov_rectangle = project_fov(viewpoint, fov_angle, max_viewing_distance)
        ax.add_collection3d(trimesh.visualizations.plot_wireframe(fov_rectangle, color='r'))
    cbar = plt.colorbar()
    cbar.set_label('Covered Area')
    plt.show()

viewpoints_with_areas = []
for viewpoint, angle in viewpoints_with_angles:
    fov_vertices = project_fov(viewpoint, fov_angle, max_viewing_distance)
    covered_area = calculate_covered_area(mesh, viewpoint,fov_vertices)
    viewpoints_with_areas.append((viewpoint, angle, covered_area))

visualize_viewpoints_with_model(mesh, viewpoints_with_areas)

print(viewpoints_with_areas)


