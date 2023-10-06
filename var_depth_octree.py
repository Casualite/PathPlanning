from viewpoint_open import *
import open3d as o3d

help(o3d)
class VariableDepthOctree:
    def __init__(self):
        self.root_node=None
        self.origin=np.zeros(3)
        self.size=0
        self.octree=o3d.geometry.Octree()

    def Clear(self):
        self.root_node_ = None
        self.origin=np.zeros(3)
        self.size = 0
    def convert_from_point_cloud(self,point_cloud,size_expand=0.01):
        self.Clear()
        min_bound=point_cloud.get_min_bound()
        max_bound=point_cloud.get_max_bound()
        center=(max_bound+min_bound)/2
        half_sizes = center - min_bound
        max_half_size = half_sizes.max()
        self.origin = np.min(center - max_half_size)
        if (max_half_size == 0):
            self.size = size_expand
        else: 
            self.size = max_half_size * 2 * (1 + size_expand);
        has_colors = point_cloud.HasColors()
        for x in range(0,point_cloud.points.size):
            pass
        

