import open3d as o3d
import numpy as np
import random
from numpy.random import default_rng
from multiprocessing import Pool
import copy
from sklearn.cluster import KMeans 
from sklearn.neighbors import NearestNeighbors
import torch

#Loading the mesh
def load_mesh(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_triangle_normals()
    return mesh

#Getting sampled points of the mesh using poisson disk sampling
def sample_mesh_points(mesh,num_points):
    mesh_sampled = mesh.sample_points_poisson_disk(number_of_points=num_points,init_factor=4)
    return mesh_sampled


#To get the normals of the sampled points
def compute_normals(mesh_sampled):
    normals = np.asarray(mesh_sampled.normals)
    return mesh_sampled,normals

#Creating a scene for Raycasting purpose
def create_scene(mesh):
    scene = o3d.t.geometry.RaycastingScene()
    mesh=o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh)
    return scene

#Computing viewpoints by mosing the sampled points by distance=7 along the normal
#viewpoints which are below the ground plane moved up by radius=0.5
def compute_new_viewpoints(points, normals, distance,radius=0.5):
    viewpoints = points - distance * normals
    mask=np.where(viewpoints[:,2]<radius)
    dn=(viewpoints[mask,2]-radius)/normals[mask,2]
    viewpoints[mask,:]=viewpoints[mask,:]-normals[mask]*dn[0][:,None]
    return viewpoints


#Removing viewpoints inside the mesh
def remove_viewpoints_inside(viewpoints,scene):
    viewpoints= o3d.core.Tensor(viewpoints.tolist(),dtype=o3d.core.Dtype.Float32)
    occupancy=np.asarray(scene.compute_occupancy(viewpoints))
    return np.where(occupancy==0)[0]


def get_seen_indices(camera):
    if(type(camera)!=np.ndarray):
        camera=camera.numpy()
    tri=np.unique(camera)
    d_inx=np.where(tri == 4294967295)
    tri=np.delete(tri,d_inx)
    tri=tri.astype(int)
    return(tri)


#For getting the triangles seen by the camera, primitive_ids gives the indices of triangles hit by the rays
def get_seen_triangles(camera):
    tri=np.asarray(camera['primitive_ids'].numpy())
    tri=np.unique(tri)
    d_inx=np.where(tri == 4294967295)
    tri=np.delete(tri,d_inx)
    tri=tri.astype(int)
    return tri

#For getting the mesh seen by the camera
def get_seen_mesh_PC(mesh,all_seen_triangles,colour=[1, 0, 0]):
    seen_mesh = copy.deepcopy(mesh)
    seen_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(seen_mesh.triangles)[all_seen_triangles, :])
    seen_mesh.triangle_normals = o3d.utility.Vector3dVector(np.asarray(seen_mesh.triangle_normals)[all_seen_triangles,:])
    seen_mesh.paint_uniform_color(colour)
    return(seen_mesh)


#Getting the area of the mesh covered by a particular camera at a viewpoint
def get_seen_areas(mesh,seen_indices):
    if(seen_indices.shape[0]==0):
        return 0
    vertices=np.asarray(mesh.vertices)
    triangle_vertices=vertices[np.asarray(mesh.triangles)[seen_indices]]
    return(np.sum(np.linalg.norm(np.cross(triangle_vertices[:,1] - triangle_vertices[:,0], triangle_vertices[:,2] - triangle_vertices[:,0], axis=1),axis=1)/2))


#For casting rays
def cast_per_point(scene,normal,center,fov,viewpoint,width,height):
    u=np.cross(np.array([0,0,1]),normal)
    u=np.cross(normal,u)
    if(np.array_equal(u,np.array([0.,0.,0.]))):
                u=np.array([1,0,0])
    camera = scene.create_rays_pinhole(fov_deg=float(fov),center=o3d.core.Tensor(center),eye=o3d.core.Tensor(viewpoint),up=o3d.core.Tensor(u),width_px=width,height_px=height)
    return(scene.cast_rays(camera,nthreads=0)['primitive_ids'])


def simulate_pinhole_cameras(mesh,viewpoints,centers,normals,fov,width,height,scene):
    mesh=o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh)
    cameras = []
    c=1
    for normal,viewpoint,center in zip(normals,viewpoints,centers):
        print(c)
        cameras.append(cast_per_point(scene,normal,center,fov,viewpoint,width,height))
        c+=1
    return mesh,cameras

def visualize(mesh,mesh_sampled_updated,octree,camera_seen,cam_no=-1,colour=[1, 0, 0]):
    mesh.paint_uniform_color(colour)
    point_cloud = o3d.geometry.PointCloud()
    if(cam_no==-1):
        point_cloud.points = o3d.utility.Vector3dVector(viewpoints)            
        point_cloud.normals = o3d.utility.Vector3dVector(normals)    
    else:
        point_cloud.points = o3d.utility.Vector3dVector(np.reshape(viewpoints[cam_no],(1,-1)))  
        point_cloud.normals = o3d.utility.Vector3dVector(np.reshape(normals[cam_no],(1,-1)))  
    o3d.visualization.draw_geometries([mesh,octree,mesh_sampled_updated])

def get_PCD(file_path="./meshes/B747.ply",num_points=1000,distance_d=15):
    mesh = load_mesh(file_path)
    mesh_sampled = sample_mesh_points(mesh,num_points)
    mesh_sampled,normals = compute_normals(mesh_sampled)
    viewpoints = compute_new_viewpoints(np.asarray(mesh_sampled.points), normals, distance_d)
    pcd = o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(viewpoints)
    pcd.normals = o3d.utility.Vector3dVector(normals) 
    return(mesh,viewpoints,pcd)

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

def new_viewpoint(triangle_center,r):
    min_value=0
    max_value=62830   #2*pi   we need 4 decimal resolution

    phi=(np.random.randint(min_value, max_value))/np.power(10, 4)
    theta=(np.random.randint(min_value, max_value/2))/np.power(10, 4)
    x=r*np.sin(theta)*np.cos(phi)
    y=r*np.sin(theta)*np.sin(phi)
    z=r*np.cos(theta)
    
    new_v=np.asarray(triangle_center+r*np.array([x,y,z]))
    u=np.cross(np.array([x,y,z]),np.array([0,0,1]))
    u=np.cross(u,np.array([x,y,z]))
    if(np.array_equal(u,np.array([0.,0.,0.]))):
                u=np.array([1,0,0])

    return new_v,u

def camera_rays(fov_deg,center,eye,up,width,height):
    rays=scene.create_rays_pinhole(fov_deg=float(fov_deg),center=o3d.core.Tensor(center),eye=o3d.core.Tensor(eye),up=o3d.core.Tensor(up),width_px=width,height_px=height)
    return rays

def occlusion(arr_missing_centroids,normals,distance_d,scene):
    tri=np.array([])
    for missing_tri_centroid,normal in zip(arr_missing_centroids,normals):            
            max=0
            max_new_seen_tri=np.array([])
            up=np.cross(normal,np.array([0,0,1]))
            up=np.cross(up,normal)
            if(np.array_equal(up,np.array([0.,0.,0.]))):
                up=np.array([1,0,0])
            viewpoint=missing_tri_centroid - distance_d * normal
            camera_1=camera_rays(float(0),missing_tri_centroid,viewpoint,up,1,1)
            occlusion_detection=scene.test_occlusions(camera_1,tfar=distance_d)  
            if(occlusion_detection or remove_viewpoints_inside(viewpoint.reshape(1,-1),scene)==1):   #if yes I will perform occlusion avoidance by finding a new viewpoint to see the triangle without being occluded
                for i in range(30):
                    new_view,u=new_viewpoint(missing_tri_centroid,distance_d)
                    camera=camera_rays(float(0),missing_tri_centroid,new_view,u,1,1)
                    occlusion_detection=scene.test_occlusions(camera,tfar=distance_d)
                    if(occlusion_detection or remove_viewpoints_inside(new_view.reshape(1,-1),scene)==1):
                        continue
                    else:
                        new_seen_tri=not_occluded(missing_tri_centroid,new_view,u)
                        if(len(new_seen_tri)>max):
                            max=len(new_seen_tri)
                            max_new_seen_tri=new_seen_tri
                        
                tri=(np.concatenate((tri,np.array(max_new_seen_tri))))
                a=len(tri)
            else:
                new_seen_tri=not_occluded(missing_tri_centroid,viewpoint,up)
                tri=(np.concatenate((tri,np.array(new_seen_tri))))
                    
    return tri
    
def not_occluded(triangle_centre,viewpoint,up):
    camera_2=camera_rays(float(30),triangle_centre,viewpoint,up,640,480)
    new_tri_seen=get_seen_triangles(scene.cast_rays(camera_2,nthreads=0))
    return new_tri_seen

def clustering1(missing_tri,mesh):
    
    triangles = np.asarray(mesh.triangles)
    unseen_triangle=triangles[missing_tri]
    vertices = np.asarray(mesh.vertices)
    triangle_centroids = np.mean(vertices[unseen_triangle], axis=1)
    k = 15

    kmeans = KMeans(n_clusters=k, random_state=0,init="k-means++").fit(triangle_centroids)
    cluster_labels = kmeans.labels_
    cluster_centroids = kmeans.cluster_centers_
    closest_triangle_centroids=np.array([0.,0.,0.]) 
    normals=np.array([0.,0.,0.]) 
    masks=np.array([],dtype=int)
    for x in range(k):
        mask=np.where(cluster_labels==x)
        cluster_points=triangle_centroids[mask]
        neigh = NearestNeighbors(n_neighbors=1,n_jobs=-1)
        neigh.fit(cluster_points)
        _,idx=neigh.kneighbors(cluster_centroids[x].reshape(1,-1))
        closest_triangle_centroids=np.vstack((closest_triangle_centroids,cluster_points[idx[0]]))
        normals=np.vstack((normals,mesh.triangle_normals[missing_tri[mask[0][idx[0]]]]))
        masks=np.append(masks,missing_tri[mask[0]])
    return closest_triangle_centroids[1:,:],normals[1:,:],masks

def complete_seen_mesh(cameras,scene):
    all_seen_triangles=np.array([],dtype=int) 
    for i in range(len(cameras)):
        all_seen_triangles=np.append(all_seen_triangles,get_seen_indices(cameras[i]))
    all_seen_triangles=np.unique(all_seen_triangles)
    total_seen_mesh=get_seen_mesh_PC(mesh,all_seen_triangles)
    triangle_indices=list(range(len(mesh.triangles)))

    print(f"TRIANGLES IN THE MESH : {len(triangle_indices)}")
    print(f"TRIANGLES SEEN BEFORE PERFORMING OCCLUSION AVOIDANCE :{len(all_seen_triangles)}")

    missing_tri = np.setdiff1d(np.asarray(triangle_indices),all_seen_triangles)
    print(f" MISSING TRIANGLES : {len(missing_tri)}")

    for i in range(5):
        clustered_triangle_centers,normals,masks=clustering1(missing_tri,mesh)
        new_tri_seen=occlusion(clustered_triangle_centers,normals,distance_d,scene=scene)
        missing_tri=np.setdiff1d(missing_tri,new_tri_seen)

    print(f"TRIANGLES SEEN AFTER PERFORMING OCCLUSION AVOIDANCE {len(np.unique(np.append(all_seen_triangles,new_tri_seen)))}")
    j=1
    return  all_seen_triangles
    #o3d.visualization.draw_geometries([total_seen_mesh])
    
if __name__ == "__main__":
    file_path = "./meshes/B747_shifted_origin.ply"
    num_points=1000
    distance_d = 7
   
    # Load mesh
    mesh = load_mesh(file_path)
    cam_no=26
    # Sample mesh points and compute normals
    mesh_sampled = sample_mesh_points(mesh,num_points)
    mesh_sampled,normals = compute_normals(mesh_sampled)
    print(cam_no)
    scene=create_scene(mesh)
    viewpoints = compute_new_viewpoints(np.asarray(mesh_sampled.points), normals, distance_d)
    viewpoint_mask=remove_viewpoints_inside(viewpoints,scene=scene)
    print(viewpoint_mask)
    viewpoints=viewpoints[viewpoint_mask]
    normals=normals[viewpoint_mask]

 
    
    mesh1,cameras = simulate_pinhole_cameras(mesh,viewpoints,np.asarray(mesh_sampled.points),normals,60,640,480,scene=scene)
    all_seen_triangles=complete_seen_mesh(cameras,scene=scene)
    octree=o3d.geometry.Octree(N=30,x=1)
    octree1=o3d.geometry.Octree(N=30,x=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(viewpoints)
    pcd.normals = o3d.utility.Vector3dVector(normals) 
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    octree1.convert_from_point_cloud(pcd, size_expand=0.01)
    k=octree.postprocess_tree()

    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(viewpoints)            
    # point_cloud.normals = o3d.utility.Vector3dVector(normals) 
    # o3d.visualization.draw_geometries([mesh,point_cloud,octree1,get_seen_mesh_PC(mesh, all_seen_triangles)])

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
    visualize(mesh,pcd_new,octree1,get_seen_mesh_PC(mesh, all_seen_triangles),cam_no=cam_no)