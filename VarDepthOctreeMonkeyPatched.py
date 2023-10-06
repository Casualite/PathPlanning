from viewpoint_open import *
import open3d as o3d
import numpy as np
# monkey patching https://stackoverflow.com/questions/62955179/how-to-monkey-patch-an-instance-method-of-an-external-library-that-uses-other-in
def insert_point(self,point,f_init,f_update,fi_update):
    _f_init = f_init
    if(_f_init == None):
        _f_init=o3d.geometry.OctreeInternalNode.get_init_function() 

    _f_update=f_update
    if(_f_update==None):
        _f_update=o3d.geometry.OctreeInternalNode.get_update_function()
    
    if(self.root_node==None):
        if(self.max_depth == 0):
            self.root_node=_f_init()
        else:
            self.root_node=_f_init()

    root_node_info=o3d.geometry.OctreeNodeInfo(self.origin,self.size,0,0)
    def insert_point_recurse(node, node_info, point, fl_init, fl_update,
                       _f_init, _f_update):
        if(not o3d.geometry.Octree.is_point_in_bound(point,node_info.origin,node_info.size)):
            return

        # if(node_info)
        # pass
    insert_point_recurse(self.root_node,root_node_info,point,f_init,f_update,_f_init,_f_update)

k=o3d.geometry.Octree()
k.insert_point=insert_point
pcd = o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(np.array([[1,1,1]]))
pcd.normals = o3d.utility.Vector3dVector(np.array([[1,1,1]])) 
k.insert_point(pcd)
l=1