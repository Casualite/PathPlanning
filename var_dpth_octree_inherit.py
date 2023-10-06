from viewpoint_open import *
import open3d as o3d
import numpy as np

class VariableDepthOctree(o3d.geometry.Octree):
    def __init__(self):
        o3d.geometry.Octree.__init__(self)

    # def convert_from_point_cloud(self,point_cloud,size_expand=0.01):
    #     self.clear()
    #     print("monkey_patched")
    #     # min_bound=point_cloud.get_min_bound()
    #     # max_bound=point_cloud.get_max_bound()
    #     # center=(max_bound+min_bound)/2
    #     # half_sizes = center - min_bound
    #     # max_half_size = half_sizes.max()
    #     # self.origin = np.min(center - max_half_size)
    #     # if (max_half_size == 0):
    #     #     self.size = size_expand
    #     # else: 
    #     #     self.size = max_half_size * 2 * (1 + size_expand);
        
    #     # has_colors = point_cloud.HasColors()
    #     # for (size_t idx = 0; idx < point_cloud.points_.size(); idx++) {
    #     #     const Eigen::Vector3d& color =
    #     #             has_colors ? point_cloud.colors_[idx] : Eigen::Vector3d::Zero();
    #     #     InsertPoint(point_cloud.points_[idx],
    #     #                 OctreePointColorLeafNode::GetInitFunction(),
    #     #                 OctreePointColorLeafNode::GetUpdateFunction(idx, color),
    #     #                 OctreeInternalPointNode::GetInitFunction(),
    #     #                 OctreeInternalPointNode::GetUpdateFunction(idx));
    #     # }

    def insert_point(self):
        print("Monkeypatch")

pcd=get_PCD()
c=VariableDepthOctree()
c.convert_from_point_cloud(pcd)
