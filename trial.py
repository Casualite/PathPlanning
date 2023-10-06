import open3d as o3d
import numpy as np
import random
from numpy.random import default_rng
import copy
from sklearn.cluster import KMeans 
from sklearn.neighbors import NearestNeighbors
import torch

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

def compute_new_viewpoints(points, normals, distance,radius=0.5):
    viewpoints = points - distance * normals
    mask=np.where(viewpoints[:,2]<radius)
    dn=(viewpoints[mask,2]-radius)/normals[mask,2]
    # viewpoints[mask,:]=viewpoints[mask,:]-normals[mask]*dn[0][:,None]
    return viewpoints

def remove_viewpoints_inside(viewpoints,mesh,scene):
    viewpoints=o3d.core.Tensor(viewpoints)
    filtered_viewpoints=torch.where(scene.compute_occupancy(viewpoints),viewpoints,torch.tensor([0.,0.,0.]))
    return filtered_viewpoints

def get_seen_indices(camera):
    tri=np.unique(camera.numpy())
    d_inx=np.where(tri == 4294967295)
    tri=np.delete(tri,d_inx)
    tri=tri.astype(int)
    return(tri)

def get_seen_triangles(camera):
    tri=np.asarray(camera['primitive_ids'].numpy())
    tri=np.unique(tri)
    d_inx=np.where(tri == 4294967295)
    tri=np.delete(tri,d_inx)
    tri=tri.astype(int)
    return tri
'''
def get_seen_mesh_PC(mesh,camera):
    tri=get_seen_indices(camera)
    seen_mesh = copy.deepcopy(mesh)
    seen_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(seen_mesh.triangles)[tri, :])
    seen_mesh.triangle_normals = o3d.utility.Vector3dVector(np.asarray(seen_mesh.triangle_normals)[tri,:])
    seen_mesh.paint_uniform_color([1, 0, 0])
    return(seen_mesh)
'''
def get_seen_mesh_PC(mesh,all_seen_triangles,colour=[1, 0, 0]):
    seen_mesh = copy.deepcopy(mesh)
    seen_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(seen_mesh.triangles)[all_seen_triangles, :])
    seen_mesh.triangle_normals = o3d.utility.Vector3dVector(np.asarray(seen_mesh.triangle_normals)[all_seen_triangles,:])
    seen_mesh.paint_uniform_color(colour)
    return(seen_mesh)

def create_scene(mesh):
    scene = o3d.t.geometry.RaycastingScene()
    mesh=o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh)
    return scene
    
def cast_per_point(scene,normal,center,fov,viewpoint,width,height):
    u=np.cross(np.array([0,0,1]),normal)
    u=np.cross(normal,u)
    camera = scene.create_rays_pinhole(fov_deg=float(fov),center=o3d.core.Tensor(center),eye=o3d.core.Tensor(viewpoint),up=o3d.core.Tensor(u),width_px=width,height_px=height)
    return(scene.cast_rays(camera,nthreads=0)['primitive_ids'])

def simulate_pinhole_cameras(mesh,viewpoints,centers,normals,fov,width,height):
    scene = o3d.t.geometry.RaycastingScene()
    mesh=o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh)
    cameras = []
    c=1
    for normal,viewpoint,center in zip(normals,viewpoints,centers):
        print(c)
        cameras.append(cast_per_point(scene,normal,center,fov,viewpoint,width,height))
        c+=1
    return mesh,cameras

def visualize(mesh,mesh_sampled,cluster,camera_seen,viewpoints,normals,cam_no=-1):
    point_cloud = o3d.geometry.PointCloud()
    if(cam_no==-1):
        point_cloud.points = o3d.utility.Vector3dVector(viewpoints)            
        point_cloud.normals = o3d.utility.Vector3dVector(normals)    
    else:
        point_cloud.points = o3d.utility.Vector3dVector(np.reshape(viewpoints[cam_no],(1,-1)))  
        point_cloud.normals = o3d.utility.Vector3dVector(np.reshape(normals[cam_no],(1,-1)))  
    o3d.visualization.draw_geometries([mesh,camera_seen,point_cloud,cluster,mesh_sampled])

def visualize_clusters(mesh,seen,cluster):
   o3d.visualization.draw_geometries([mesh,seen,cluster])

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
            print("{}{}: Leaf node at depth {} has {} points with origin {}".
                  format('    ' * node_info.depth, node_info.child_index,
                         node_info.depth, len(node.indices), node_info.origin))
    else:
        raise NotImplementedError('Node type not recognized!')

    # early stopping: if True, traversal of children of the current node will be skipped
    return early_stop
#add areas instead
#wrong
def update_weights(cameras,triangle_seen,seen):
    l=[]
    for x in range(len(cameras)):
        triangle_seen[x]=np.setdiff1d(np.array(triangle_seen[x]),seen).to_list()
        l.append(len(triangle_seen[x]))
    return(l)

def select_viewpoints(subspace,triangle_seen,total,ratio_min):
    cameras=[c for c in subspace[0].indices]
    weights=[len(triangle_seen[c]) for c in subspace[0].indices]
    range=list(range(0,len(cameras)))
    totally_seen=np.array([])
    chosen=[]
    while len(cameras)>0 and totally_seen.shape/total>ratio_min:
        i=random.choice(range,weights,k=1)
        camera=cameras.pop(i)
        chosen.append(camera)
        _=weights.pop(i)
        _=range.pop(len(range)-1)
        seen=get_seen_indices(camera)
        totally_seen=np.concatenate((totally_seen,seen))
        weights=update_weights(cameras,triangle_seen,totally_seen)
    return(camera)
        

def get_selected_viewpoints(k,cameras):
    for x in k:
        pass

'''
def new_viewpoint(triangle_center,r,fov,width,height,scene):
    min_value=0
    max_value=62830
    #arr_tri_vertices = mesh.triangles[triangle_index]
    #triangle_center = cal_tri_center(arr_tri_vertices) 
    #print(f" Triangle centre {triangle_center}")
    phi=(np.random.randint(min_value, max_value))/np.power(10, 4)
    theta=(np.random.randint(min_value, max_value/2))/np.power(10, 4)
    x=r*np.sin(theta)*np.cos(phi)
    y=r*np.sin(theta)*np.sin(phi)
    z=r*np.cos(theta)
    
    new_v=np.asarray(triangle_center+r*np.array([x,y,z]))
    u=np.cross(np.array([x,y,z]),np.array([0,0,1]))
    #print(r*np.array([x,y,z]))
    #print(f"New point vertices {new_v}")
    camera = scene.create_rays_pinhole(fov_deg=float(fov),center=o3d.core.Tensor(triangle_center),eye=o3d.core.Tensor(new_v),up=o3d.core.Tensor(u),width_px=width,height_px=height)
    new_tri_seen=get_seen_triangles(scene.cast_rays(camera,nthreads=0))
    #print(camera['primitive_ids'].numpy())
    return new_tri_seen,theta,phi
'''

def new_viewpoint(triangle_center,r):
    min_value=0
    max_value=62830   #2*pi   we need 4 decimal resolution

    phi=(np.random.randint(min_value, max_value))/np.power(10, 4)
    theta=(np.random.randint(min_value, max_value/2))/np.power(10, 4)
    x=r*np.sin(theta)*np.cos(phi)
    y=r*np.sin(theta)*np.sin(phi)
    z=r*np.cos(theta)
    
    new_v=np.asarray(triangle_center+r*np.array([x,y,z]))
    u=np.cross(np.array([0,0,1]),np.array([x,y,z]))
    u=np.cross(np.array([x,y,z]),u)

    return new_v,u

def occlusion_avoidance(arr_missing_centroids):
    tri=np.array([])
    print("in occlusion")
    for missing_tri_centroid in arr_missing_centroids: 
        max=0
        max_new_seen_tri=None
        for i in range(100):
            #arr_tri_vertices = vertices[mesh.triangles[triangle_index]]
            new_tri_seen,theta,phi=new_viewpoint(missing_tri_centroid,distance_d)
            if(len(new_tri_seen)>max):
                max=len(new_tri_seen)
                max_new_seen_tri=new_tri_seen
        tri=np.unique(np.concatenate((tri,np.array(max_new_seen_tri))))
    a=len(tri)
    j=1
    print("done occlusion")
    return tri




def occlusion(arr_missing_centroids,normals,distance_d,scene):
    tri=np.array([])
    for missing_tri_centroid,normal in zip(arr_missing_centroids,normals):            
            max=0
            max_new_seen_tri=np.array([])
            up=np.cross(np.array([0,0,1]),normal)
            up=np.cross(normal,up)
            if(np.array_equal(up,np.array([0.,0.,0.]))):
                up=np.array([1,0,0])
            viewpoint=missing_tri_centroid - distance_d * normal
            print("before occlusion")
            camera_1=scene.create_rays_pinhole(fov_deg=float(0),center=o3d.core.Tensor(missing_tri_centroid),eye=o3d.core.Tensor(viewpoint),up=o3d.core.Tensor(up),width_px=1,height_px=1)
            occlusion_detection=scene.test_occlusions(camera_1,tfar=distance_d)   #Checking wheather the missing triangle is occluded
            if(occlusion_detection[0,0]):   #if yes I will perform occlusion avoidance by finding a new viewpoint to see the triangle without being occluded
                for i in range(10):
                    print("in occlusion")
                    new_view,up=new_viewpoint(missing_tri_centroid,distance_d)
                    camera = scene.create_rays_pinhole(fov_deg=float(0),center=o3d.core.Tensor(missing_tri_centroid),eye=o3d.core.Tensor(new_view),up=o3d.core.Tensor(up),width_px=1,height_px=1)
                    occlusion_detection=scene.test_occlusions(camera,tfar=distance_d)
                    print("after test occlusion")
                    if(occlusion_detection):
                        continue
                    else:
                        print("occuded one")
                        new_seen_tri=not_occluded(missing_tri_centroid,new_view,up)
                        if(len(new_seen_tri)>max):
                            max=len(new_seen_tri)
                            max_new_seen_tri=new_seen_tri
                        
                tri=(np.concatenate((tri,np.array(max_new_seen_tri))))
                a=len(tri)
            else:
                print("correct one",normal)
                new_seen_tri=not_occluded(missing_tri_centroid,viewpoint,up)
                tri=(np.concatenate((tri,np.array(new_seen_tri))))
                    
    return tri
    
def not_occluded(triangle_centre,viewpoint,up):
     print("inside not occluded")
     print(triangle_centre,viewpoint,up)
     camera2=scene.create_rays_pinhole(fov_deg=float(30),center=o3d.core.Tensor(triangle_centre),eye=o3d.core.Tensor(viewpoint),up=o3d.core.Tensor(-up),width_px=640,height_px=480)
     new_tri_seen=get_seen_triangles(scene.cast_rays(camera2,nthreads=0))
     print("done not occluded")
     return new_tri_seen

def clustering1(missing_tri,mesh):
    
    triangles = np.asarray(mesh.triangles)
    unseen_triangle=triangles[missing_tri]
    vertices = np.asarray(mesh.vertices)
    triangle_centroids = np.mean(vertices[unseen_triangle], axis=1)
    print(triangle_centroids)
    k = 20

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

    print(f"No of triangles in the mesh {len(triangle_indices)}")
    print(f"Old seen triangles {len(all_seen_triangles)}")

    missing_tri = np.setdiff1d(np.asarray(triangle_indices),all_seen_triangles)
    print(f"No of missing triangles {len(missing_tri)}")
    clustered_triangle_centers,normals,masks=clustering1(missing_tri,mesh)
    #print(closest_triangle_vertices)
    #visualize_clusters(mesh,total_seen_mesh,get_seen_mesh_PC(mesh,masks,[0,1,0]))
    #new_tri_seen=occlusion_avoidance(clustered_triangle_centers,normals)
    new_tri_seen=occlusion(clustered_triangle_centers,normals,distance_d,scene=scene)
    #visualize_clusters(mesh,total_seen_mesh,get_seen_mesh_PC(mesh,masks,[0,1,0]))
    #mesh1,cameras=simulate_pinhole_cameras(mesh,compute_new_viewpoints(clustered_triangle_centers,normals,15),clustered_triangle_centers,normals,30,640,480)
    print(f"New Seen Triangles {len(np.unique(np.append(all_seen_triangles,new_tri_seen)))}")
    j=1
    #o3d.visualization.draw_geometries([total_seen_mesh])

def remove_viewpoints_below_gp(viewpoints,normals):
    for viewpoint,normal in zip(viewpoints,normals):
        point=viewpoint+(normal*np.array([0,0,-1]))
        z=0
        if(z>point[3]):
            viewpoints.remove(viewpoint)
            normals.remove(normal)
    return viewpoints,normals


    

if __name__ == "__main__":
    file_path = "./meshes/B747_shifted_origin.ply"
    num_points=1000
    distance_d = 15

    # Load mesh
    mesh = load_mesh(file_path)
    cam_no=26
    scene=create_scene(mesh)
    # Sample mesh points and compute normals
    mesh_sampled = sample_mesh_points(mesh,num_points)
    mesh_sampled,normals = compute_normals(mesh_sampled)
    viewpoints = compute_new_viewpoints(np.asarray(mesh_sampled.points), normals, distance_d)
    #remove_viewpoints_inside(viewpoints,mesh,scene=scene)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points=o3d.utility.Vector3dVector(viewpoints)
    # pcd.normals = o3d.utility.Vector3dVector(normals) 
    
    # o3d.visualization.draw_geometries([mesh,pcd])

   
    mesh1,cameras = simulate_pinhole_cameras(mesh,viewpoints,np.asarray(mesh_sampled.points),normals,60,640,480)
    complete_seen_mesh(cameras,scene=scene)
    
    '''
    octree=o3d.geometry.Octree(N=3,x=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(viewpoints)
    pcd.normals = o3d.utility.Vector3dVector(normals) 
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    k=octree.postprocess_tree()
    l=[]
    for x in k:
        l=l+x[0].indices
    arr=np.array(l)
    print(arr.shape)
    arr=np.unique(arr)
    print(arr.shape)
    l=5
    '''

    #visualize(get_seen_mesh_PC(mesh,cameras[cam_no]),mesh_sampled,viewpoints,normals,cam_no=cam_no)