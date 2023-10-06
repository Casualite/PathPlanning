import numpy as np
def get_intrinsic(dim_x,a,fov,cx,cy,s):
    fov=np.deg2rad(fov)
    dim_y=round(dim_x*a)
    f=dim_x/(2*np.tan(fov/2))
    return(dim_y,[[f,s,cx],[0,a*f,cy],[0,0,1]])
def roll_mat(roll):
    return(np.array([[np.cos(roll),-np.sin(roll),0,0],[np.sin(roll),np.cos(roll),0,0],[0,0,1,0],[0,0,0,1]]))
def pitch_mat(pitch):
    return(np.array([[np.cos(pitch),0,np.sin(pitch),0],[0,1,0,0],[-np.sin(pitch),0,np.cos(pitch),0],[0,0,0,1]]))
def yaw_mat(yaw):
    return(np.array([[1,0,0,0],[0,np.cos(yaw),-np.sin(yaw),0],[0,np.sin(yaw),np.cos(yaw),0],[0,0,0,1]]))
def pos_mat(pos):
    return(np.array([[1,0,0,pos[0]],[0,1,0,pos[1]],[0,0,1,pos[2]],[0,0,0,1]]))
def get_homogenous_transformation(p,n):
    roll=0
    pitch=np.arctan(n[2]/np.sqrt(np.square(n[0])+np.square(n[1])))
    yaw=np.arctan(-n[1]/-n[0])
    return(np.linalg.inv(pos_mat(p)@roll_mat(roll)@pitch_mat(pitch)@yaw_mat(yaw)))