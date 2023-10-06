from viewpoint_combined import get_children_nodes
mesh,pcd,leaf_nodes=get_children_nodes(file_path="./meshes/B747_shifted_origin_f1.ply",num_points=8000,raycast=True,cluster_iterations=30,clusters=10)