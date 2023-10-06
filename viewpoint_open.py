import open3d as o3d
import numpy as np
import random
from numpy.random import default_rng
from multiprocessing import Pool
import copy

# TODO Occlusion and collision checks, viewpoint sampling
def load_mesh(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_triangle_normals()
    return mesh

def sample_mesh_points(mesh,num_points):
    mesh_sampled = mesh.sample_points_poisson_disk(number_of_points=num_points,init_factor=1)
    return mesh_sampled

def compute_normals(mesh_sampled):
    normals = np.asarray(mesh_sampled.normals)
    return mesh_sampled,normals

def compute_new_viewpoints(points, normals, distance):
    viewpoints = points - distance * normals
    return viewpoints

def get_seen_indices(camera):
    if(type(camera)!=np.ndarray):
        camera=camera.numpy()
    tri=np.unique(camera)
    d_inx=np.where(tri == 4294967295)
    tri=np.delete(tri,d_inx)
    tri=tri.astype(int)
    return(tri)

def get_seen_mesh_PC(mesh,camera):
    tri=get_seen_indices(camera)
    seen_mesh = copy.deepcopy(mesh)
    seen_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(seen_mesh.triangles)[tri, :])
    seen_mesh.triangle_normals = o3d.utility.Vector3dVector(np.asarray(seen_mesh.triangle_normals)[tri,:])
    seen_mesh.paint_uniform_color([1, 0, 0])
    return(seen_mesh)

def get_seen_areas(mesh,seen_indices):
    if(seen_indices.shape[0]==0):
        return 0
    vertices=np.asarray(mesh.vertices)
    triangle_vertices=vertices[np.asarray(mesh.triangles)[seen_indices]]
    return(np.sum(np.linalg.norm(np.cross(triangle_vertices[:,1] - triangle_vertices[:,0], triangle_vertices[:,2] - triangle_vertices[:,0], axis=1),axis=1)/2))

def simulate_pinhole_cameras(mesh,viewpoints,centers,normals,fov,width,height):
    scene = o3d.t.geometry.RaycastingScene()
    mesh=o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh)
    cameras = []
    c=1
    for normal,viewpoint,center in zip(normals,viewpoints,centers):
        print(c)
        u=np.cross(np.array([0,0,1]),normal)
        u=np.cross(normal,u)
        camera = scene.create_rays_pinhole(fov_deg=float(fov),center=o3d.core.Tensor(center),eye=o3d.core.Tensor(viewpoint),up=o3d.core.Tensor(u),width_px=width,height_px=height)
        cameras.append(scene.cast_rays(camera,nthreads=0)['primitive_ids'])
        c+=1
    return mesh,cameras

def visualize(mesh,mesh_sampled,mesh_sampled_updated,octree,camera_seen,viewpoints,normals,cam_no=-1):
    point_cloud = o3d.geometry.PointCloud()
    if(cam_no==-1):
        point_cloud.points = o3d.utility.Vector3dVector(viewpoints)            
        point_cloud.normals = o3d.utility.Vector3dVector(normals)    
    else:
        point_cloud.points = o3d.utility.Vector3dVector(np.reshape(viewpoints[cam_no],(1,-1)))  
        point_cloud.normals = o3d.utility.Vector3dVector(np.reshape(normals[cam_no],(1,-1)))  
    o3d.visualization.draw_geometries([mesh,camera_seen,point_cloud,octree,mesh_sampled,mesh_sampled_updated])

def get_PCD(file_path="./meshes/B747.ply",num_points=1000,distance_d=15):
    mesh = load_mesh(file_path)
    mesh_sampled = sample_mesh_points(mesh,num_points)
    mesh_sampled,normals = compute_normals(mesh_sampled)
    viewpoints = compute_new_viewpoints(np.asarray(mesh_sampled.points), normals, distance_d)
    pcd = o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(viewpoints)
    pcd.normals = o3d.utility.Vector3dVector(normals) 
    return(mesh,viewpoints,pcd)

def f_traverse(node, node_info):
    early_stop = False
    List=[]
    if isinstance(node, o3d.geometry.OctreeInternalNode):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            n = 0
            for child in node.children:
                if child is not None:
                    n += 1
            print(
                "{}{}: Internal node at depth {} has {} children and {} points ({})"
                .format('    ' * node_info.depth,
                        node_info.child_index, node_info.depth, n,
                        len(node.indices), node_info.origin))

            # we only want to process nodes / spatial regions with enough points
            early_stop = len(node.indices) <1
    elif isinstance(node, o3d.geometry.OctreeLeafNode):
        if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
            List.append((node,node_info))
            print("{}{}: Leaf node at depth {} has {} points with origin {}".
                  format('    ' * node_info.depth, node_info.child_index,
                         node_info.depth, len(node.indices), node_info.origin))
    else:
        raise NotImplementedError('Node type not recognized!')

    # early stopping: if True, traversal of children of the current node will be skipped
    return early_stop

def update_weights(mesh,cameras,triangle_seen,seen):
    l=[]
    for x in range(len(cameras)):
        triangle_seen[x]=np.setdiff1d(np.array(triangle_seen[x]),seen)
        l.append(get_seen_areas(mesh,triangle_seen[x]))
    return(l)

def select_viewpoints(mesh,subspace,triangle_seen,total,ratio_min):
    cameras=[c for c in subspace[0].indices]
    totally_seen=np.array([],dtype=int)
    weights=update_weights(mesh,cameras,triangle_seen,totally_seen)
    Range=list(range(0,len(cameras)))
    chosen=[]
    ratio=get_seen_areas(mesh,totally_seen)/total
    while len(cameras)>0 and ratio<ratio_min:
        i=random.choices(Range,weights,k=1)[0]
        camera=cameras.pop(i)
        chosen.append(camera)
        _=weights.pop(i)
        _=Range.pop(len(Range)-1)
        seen=triangle_seen.pop(i)
        totally_seen=np.concatenate((totally_seen,seen))
        weights=update_weights(mesh,cameras,triangle_seen,totally_seen)
        ratio=get_seen_areas(mesh,totally_seen)/total
    return(chosen)
        
def get_selected_viewpoints(mesh,k,cameras):
    all_chosen=[]
    for x in k:
        indices=x[0].indices
        triangle_seen=np.array([],dtype=int)
        seen_camera_wise=[]
        for y in indices:
            seen=get_seen_indices(cameras[y])
            seen_camera_wise.append(seen)
            triangle_seen=np.append(triangle_seen,seen)
        triangle_seen=np.unique(triangle_seen)
        area=get_seen_areas(mesh,triangle_seen)
        chosen=select_viewpoints(mesh,x,seen_camera_wise,area,0.95)
        all_chosen.append(chosen)
    return all_chosen
    
if __name__ == "__main__":
    file_path = "./meshes/B747.ply"
    num_points=5000
    distance_d = 7

    # Load mesh
    mesh = load_mesh(file_path)
    cam_no=26
    # Sample mesh points and compute normals
    mesh_sampled = sample_mesh_points(mesh,num_points)
    mesh_sampled,normals = compute_normals(mesh_sampled)
    viewpoints = compute_new_viewpoints(np.asarray(mesh_sampled.points), normals, distance_d)
    mesh1,cameras = simulate_pinhole_cameras(mesh,viewpoints,np.asarray(mesh_sampled.points),normals,60,640,480)
    octree=o3d.geometry.Octree(N=30,x=1)
    octree1=o3d.geometry.Octree(N=30,x=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(viewpoints)
    pcd.normals = o3d.utility.Vector3dVector(normals) 
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    octree1.convert_from_point_cloud(pcd, size_expand=0.01)
    k=octree.postprocess_tree()
    
    triangles=np.asarray(mesh.triangles)
    vertices=np.asarray(mesh.vertices)
    y=vertices[triangles]
    centroids=np.mean(vertices[triangles],axis=1)
    all_chosen=get_selected_viewpoints(mesh,k,cameras)
    l=0
    new_points=np.array([[0,0,0]])
    new_normals=np.array([[0,0,0]])
    for x in all_chosen:
        new_points=np.concatenate((new_points,viewpoints[np.array(x)]))
        new_normals=np.concatenate((new_normals,normals[np.array(x)]))
        l+=len(x)
    
    pcd_new = o3d.geometry.PointCloud()
    pcd_new.points=o3d.utility.Vector3dVector(new_points[1:,:])
    pcd_new.normals = o3d.utility.Vector3dVector(new_normals[1:,:]) 
    pcd_new.paint_uniform_color(np.array([0.,0.,0.]))
    print("new",l)
    visualize(mesh,pcd,pcd_new,octree1,get_seen_mesh_PC(mesh,cameras[cam_no]),viewpoints,normals,cam_no=cam_no)
    
   
    

    
