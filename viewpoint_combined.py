import open3d as o3d
import numpy as np
import random
from numpy.random import default_rng
from multiprocessing import Pool
import copy
from sklearn.cluster import KMeans,AgglomerativeClustering 
from sklearn.neighbors import NearestNeighbors

#Loading the mesh
def load_mesh(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_triangle_normals()
    return mesh

def remove_hidden_triangles(mesh,scene):
   triangles = np.asarray(mesh.triangles)
   vertices = np.asarray(mesh.vertices)
   triangle_centroids = np.mean(vertices[triangles], axis=1,dtype=np.float32)
   inside=np.asarray(scene.compute_occupancy(o3d.core.Tensor(triangle_centroids),nsamples=5,nthreads=12))
   mask=np.where(inside==True)
   mesh=get_seen_mesh_PC(mesh,mask[0])
   #o3d.visualization.draw_geometries([mesh])
   return(mesh)
#Getting sampled points of the mesh using poisson disk sampling
def sample_mesh_points(mesh,num_points):
    #mesh_sampled = mesh.sample_points_poisson_disk(number_of_points=num_points,init_factor=4)
    mesh_sampled = mesh.sample_points_uniformly(number_of_points=num_points)
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
def move_from_ground(viewpoints,normals,radius=0.5):
    mask=np.where(viewpoints[:,2]<radius)
    dn=(viewpoints[mask,2]-radius)/normals[mask,2]
    viewpoints[mask,:]=viewpoints[mask,:]-normals[mask]*dn[0][:,None]
    return(viewpoints)

def compute_new_viewpoints(points, normals, distance,radius=0.5):
    viewpoints = points + distance * normals
    mask=np.where(normals[:,2]!=0)
    viewpoints=move_from_ground(viewpoints[mask],normals[mask],radius)
    return viewpoints
#Removing viewpoints inside the mesh
def remove_viewpoints_inside(viewpoints,scene):
    viewpoints_tensor= o3d.core.Tensor(viewpoints.tolist(),dtype=o3d.core.Dtype.Float32)
    occupancy=np.asarray(scene.compute_occupancy(viewpoints_tensor,nsamples=21))
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
#For casting rays
def cast_per_point(scene,normal,center,fov,viewpoint,width,height):
    u=np.cross(np.array([0,0,1]),normal)
    u=np.cross(normal,u)
    if(np.array_equal(u,np.array([0.,0.,0.]))):
                u=np.array([1,0,0])
    camera = scene.create_rays_pinhole(fov_deg=float(fov),center=o3d.core.Tensor(center),eye=o3d.core.Tensor(viewpoint),up=o3d.core.Tensor(u),width_px=width,height_px=height)
    return(scene.cast_rays(camera,nthreads=0)['primitive_ids'])
#returns the mesh and primitive ids of seen triangles
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
#Getting the area of the mesh covered by a particular camera at a viewpoint
def get_seen_areas(mesh,seen_indices):
    if(seen_indices.shape[0]==0):
        return 0
    vertices=np.asarray(mesh.vertices)
    triangle_vertices=vertices[np.asarray(mesh.triangles)[seen_indices]]
    return(np.sum(np.linalg.norm(np.cross(triangle_vertices[:,1] - triangle_vertices[:,0], triangle_vertices[:,2] - triangle_vertices[:,0], axis=1),axis=1)/2))

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

#returns a list of lists of indices of sampled cameras
def get_selected_viewpoints(mesh,k,cameras):
    all_chosen=[]
    for x in k:
        indices=x[0].indices
        #print(len(indices),len(cameras))
        triangle_seen=np.array([],dtype=np.int32)
        seen_camera_wise=[]
        print(indices,len(cameras))
        for y in indices:
            seen=get_seen_indices(cameras[y-1])
            seen_camera_wise.append(seen)
            triangle_seen=np.append(triangle_seen,seen)
        triangle_seen=np.unique(triangle_seen)
        area=get_seen_areas(mesh,triangle_seen)
        chosen=select_viewpoints(mesh,x,seen_camera_wise,area,0.95)
        all_chosen.append(chosen)
        x[0].indices=chosen
    return all_chosen

def new_viewpoint(triangle_center,r,constrained=False,cur_t=0,cur_p=0,cone_angle=np.pi/6):
    min_value=0
    max_value=62830  
    if(not constrained):
        phi=(np.random.randint(min_value, max_value))/np.power(10, 4)
        theta=(np.random.randint(min_value, max_value/2))/np.power(10, 4)
    else:
        phi=(np.random.randint(min(cur_p-cone_angle,min_value), max(cur_p+cone_angle,max_value)))/np.power(10, 4)
        theta=(np.random.randint(min(cur_p-cone_angle,min_value), max(cur_p+cone_angle, max_value/2)))/np.power(10, 4)
    x=r*np.sin(theta)*np.cos(phi)
    y=r*np.sin(theta)*np.sin(phi)
    z=r*np.cos(theta)
    new_v=np.asarray(triangle_center+np.array([x,y,z]))
    u=np.cross(np.array([x,y,z]),np.array([0,0,1]))
    u=np.cross(u,np.array([x,y,z]))
    if(np.array_equal(u,np.array([0.,0.,0.]))):
                u=np.array([1,0,0])
    return new_v,np.array([x,y,z])/r,u,phi,theta

# CHOOSE POINTS IN A GRADIENT LIKE WAY SUCH THAT OCCLUSIONS ARE MINIMIZED? or CHOOSE POINTS IN SUCH A WAY(Improved by constraining it after no occlusion for rest of the points)
def occlusion(missing_triangles,arr_missing_centroids,normals,distance_d,scene,occlusion_checks=30,fov=60,width=640,height=480):
    tri=[]
    new_points=np.array([0.,0.,0.])
    new_normals=np.array([0.,0.,0.])
    for missing_tri_centroid,normal in zip(arr_missing_centroids,normals):            
            max=0
            max_new_seen_tri=np.array([])
            new_point=None
            new_normal=None
            found=False
            up=np.cross(normal,np.array([0,0,1]))
            up=np.cross(up,normal)
            if(np.array_equal(up,np.array([0.,0.,0.]))):
                up=np.array([1,0,0])
            viewpoint=missing_tri_centroid - distance_d * normal
            camera_1=scene.create_rays_pinhole(float(0),missing_tri_centroid,viewpoint,up,1,1)
            occlusion_detected=scene.test_occlusions(camera_1,tfar=distance_d)  
            if(occlusion_detected.numpy()[0,0] or remove_viewpoints_inside(viewpoint.reshape(1,-1),scene)[0]==1):   #if yes I will perform occlusion avoidance by finding a new viewpoint to see the triangle without being occluded
                constrained=False
                phi=0
                theta=0
                for i in range(occlusion_checks):
                    new_view,view_normal,u,phi,theta=new_viewpoint(missing_tri_centroid,distance_d,constrained,phi,theta)
                    # camera=scene.create_rays_pinhole(float(0),missing_tri_centroid,new_view,u,1,1)
                    # occlusion_detected=scene.test_occlusions(camera,tfar=distance_d)
                    #if(occlusion_detected.numpy()[0,0]  or remove_viewpoints_inside(new_view.reshape(1,-1),scene)[0]==1):
                    if(remove_viewpoints_inside(new_view.reshape(1,-1),scene)[0]==1):
                        continue
                    else:
                        constrained=True
                        new_seen_tri=not_occluded(scene,missing_tri_centroid,new_view,u,fov,width,height)
                        unique_seen=np.intersect1d(missing_triangles,new_seen_tri,assume_unique=True)
                        if(len(unique_seen)>max):
                            found=True
                            max=len(new_seen_tri)
                            max_new_seen_tri=new_seen_tri
                            new_point=new_view
                            new_normal=view_normal
                a=len(tri)
            else:
                new_seen_tri=not_occluded(scene,missing_tri_centroid,viewpoint,up,fov,width,height)
                unique_seen=np.intersect1d(missing_triangles,new_seen_tri,assume_unique=True)
                found=True
                new_point=viewpoint
                new_normal=normal
                max_new_seen_tri=new_seen_tri
                
            if(found):
                new_points=np.vstack((new_points,np.reshape(new_point,(1,-1))))
                new_normals=np.vstack((new_normals,np.reshape(new_normal,(1,-1))))
                #print(tri,max_new_seen_tri)
                tri.append(np.array(max_new_seen_tri))
                missing_triangles=np.setdiff1d(missing_triangles,unique_seen)

    if(new_points.ndim>1):
        return new_points[1:],new_normals[1:],tri,True
    else:
        return True,True,tri,False,missing_triangles
    
def not_occluded(scene,triangle_centre,viewpoint,up,fov,width,height):
    camera_2=scene.create_rays_pinhole(fov,triangle_centre,viewpoint,up,width,height)
    new_tri_seen=get_seen_indices(scene.cast_rays(camera_2,nthreads=0)['primitive_ids'])
    return new_tri_seen

def clustering1(missing_tri,mesh,clusters=10,d_treshold=7):
    triangles = np.asarray(mesh.triangles)
    unseen_triangle=triangles[missing_tri]
    vertices = np.asarray(mesh.vertices)
    triangle_centroids = np.mean(vertices[unseen_triangle], axis=1)
    clustering = KMeans(n_clusters=clusters, random_state=0,init="k-means++",n_init='auto').fit(triangle_centroids)
    
    cluster_labels = clustering.labels_
    cluster_centroids = clustering.cluster_centers_
    
    closest_triangle_centroids=np.array([0.,0.,0.]) 
    normals=np.array([0.,0.,0.]) 
    masks=np.array([],dtype=int)
    for x in range(clusters):
        mask=np.where(cluster_labels==x)
        cluster_points=triangle_centroids[mask]
        neigh = NearestNeighbors(n_neighbors=1,n_jobs=-1)
        neigh.fit(cluster_points)
        _,idx=neigh.kneighbors(np.reshape(cluster_centroids[x],(1,-1)))
        closest_triangle_centroids=np.vstack((closest_triangle_centroids,cluster_points[idx[0]]))
        normals=np.vstack((normals,-np.asarray(mesh.triangle_normals)[(missing_tri[mask],)][idx[0]][0]))
        pcd = o3d.geometry.PointCloud()
        pcd.points=o3d.utility.Vector3dVector(cluster_points)
        pcd.normals=o3d.utility.Vector3dVector(np.asarray(mesh.triangle_normals)[(missing_tri[mask],)])
        #o3d.visualization.draw_geometries([mesh,pcd])
        masks=np.append(masks,missing_tri[mask[0]])
    return closest_triangle_centroids[1:],normals[1:],masks
# returns newpoints as an array and all the seen triangles as indices
def after_occlusion_detection(mesh,cameras,scene,distance_d,clusters=10,iterations=100,fov=60,width=640,height=480):
    all_seen_triangles=np.array([],dtype=int) 
    for i in range(len(cameras)):
        all_seen_triangles=np.append(all_seen_triangles,get_seen_indices(cameras[i]))
    all_seen_triangles=np.unique(all_seen_triangles)
    triangle_indices=list(range(len(mesh.triangles)))

    print(f"TRIANGLES IN THE MESH : {len(triangle_indices)}")
    print(f"TRIANGLES SEEN BEFORE PERFORMING OCCLUSION AVOIDANCE :{len(all_seen_triangles)}")

    missing_tri = np.setdiff1d(np.asarray(triangle_indices),all_seen_triangles,assume_unique=True)
    print(f" MISSING TRIANGLES : {len(missing_tri)}")
    all_new_points=np.array([[0,0,0]],dtype=float)
    all_new_normals=np.array([[0,0,0]],dtype=float)
    for i in range(iterations):
        print(i+1)
        clustered_triangle_centers,normals,masks=clustering1(missing_tri,mesh,clusters=clusters,d_treshold=distance_d*np.tan(np.deg2rad(fov)))
        new_points,view_normals,new_tri_seen,gotten,missing_tri=occlusion(missing_tri,clustered_triangle_centers,normals,distance_d,scene,fov,width,height)
        if(gotten):
            all_new_points=np.concatenate((all_new_points, new_points), axis=0)
            all_new_normals=np.concatenate((all_new_normals, view_normals), axis=0)
            cameras.extend(new_tri_seen)
        # for x in new_tri_seen:
        #     missing_tri=np.setdiff1d(missing_tri,x,assume_unique=True)
    all_new_normals=all_new_normals[1:]
    all_new_points=all_new_points[1:]
    all_new_points=move_from_ground(all_new_points,all_new_normals)
    print(f"TRIANGLES SEEN AFTER PERFORMING OCCLUSION AVOIDANCE {len(mesh.triangles)-len(np.unique(missing_tri))}")
    return  all_new_points[1:],all_new_normals[1:],all_seen_triangles

def get_children_nodes(file_path="./meshes/B747_shifted_origin.ply",num_points=1000,distance_d = 7,fov=60,width=640,height=480,clusters=10,cluster_iterations=100,raycast=False):
    mesh = load_mesh(file_path) 
    mesh_sampled = sample_mesh_points(mesh,num_points)
    o3d.visualization.draw_geometries([mesh,mesh_sampled]) 
    mesh_sampled,normals = compute_normals(mesh_sampled)
    viewpoints = compute_new_viewpoints(np.asarray(mesh_sampled.points), normals, distance_d)
    scene=create_scene(mesh)
    viewpoint_mask=remove_viewpoints_inside(viewpoints,scene=scene)
    viewpoints=viewpoints[viewpoint_mask]
    normals=normals[viewpoint_mask]
    if(raycast):
        mesh1,cameras = simulate_pinhole_cameras(mesh,viewpoints,np.asarray(mesh_sampled.points),normals,fov,width,height,scene)
        new_seen_viewpoints,new_seen_normals,all_seen_triangles=after_occlusion_detection(mesh,cameras,scene,distance_d,clusters=clusters,iterations=cluster_iterations,fov=fov,width=width,height=height)
        viewpoints=np.concatenate((viewpoints,new_seen_viewpoints))
        normals=np.concatenate((viewpoints,new_seen_normals))
    pcd = o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(viewpoints)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    octree=o3d.geometry.Octree(N=30,x=1)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    leaf_nodes=octree.postprocess_tree()
    _=get_selected_viewpoints(mesh,leaf_nodes,cameras)
    return(mesh,pcd,leaf_nodes)
