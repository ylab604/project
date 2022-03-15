##Depth_Render.py
"""
author: Ylab
data: 2022-03-11
render for Depth class
"""
import torch.nn as nn
import torch

#from engineer.render.PIFuhd import gl
# 기본으로 해줘야됨
from .Base_Render import _Base_Render

# 노말 체크
from .registry import NORMAL_RENDER

# face3D
from engineer.render.face3d import mesh
import numpy as np
from PIL import Image

######check
from math import cos, sin

def similarity_transform(vertices, s, R, t3d):
        ''' similarity transform. dof = 7.
        3D: s*R.dot(X) + t
        Homo: M = [[sR, t],[0^T, 1]].  M.dot(X)
        Args:(float32)
            vertices: [nver, 3]. 
            s: [1,]. scale factor.
            R: [3,3]. rotation matrix.
            t3d: [3,]. 3d translation vector.
        Returns:
            transformed vertices: [nver, 3]
        '''
        t3d = np.squeeze(np.array(t3d, dtype = np.float32))
        transformed_vertices = s * vertices.dot(R.T) + t3d[np.newaxis, :]

        return transformed_vertices

def to_image(vertices, h, w, is_perspective = False):
    ''' change vertices to image coord system
    3d system: XYZ, center(0, 0, 0)
    2d image: x(u), y(v). center(w/2, h/2), flip y-axis. 
    Args:
        vertices: [nver, 3]
        h: height of the rendering
        w : width of the rendering
    Returns:
        projected_vertices: [nver, 3]  
    '''
    image_vertices = vertices.copy()
    if is_perspective:
        # if perspective, the projected vertices are normalized to [-1, 1]. so change it to image size first.
        image_vertices[:,0] = image_vertices[:,0]*w/2
        image_vertices[:,1] = image_vertices[:,1]*h/2
    # move to center of image
    image_vertices[:,0] = image_vertices[:,0] + w/2
    image_vertices[:,1] = image_vertices[:,1] + h/2
    # flip vertices along y-axis.
    image_vertices[:,1] = h - image_vertices[:,1] - 1
    return image_vertices


def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch. positive for looking down.
        y: yaw. positive for looking left. 
        z: roll. positive for tilting head right. 
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x),  -sin(x)],
                 [0, sin(x),   cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, sin(y)],
                 [      0, 1,      0],
                 [-sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), -sin(z), 0],
                 [sin(z),  cos(z), 0],
                 [     0,       0, 1]])
    
    R=Rz.dot(Ry.dot(Rx))
    return R.astype(np.float32)




@NORMAL_RENDER.register_module
class Noraml_Render(_Base_Render):
    """Render class for getting normal map
    here attribute  is based on vertex normal
    we normalized value of them into [0,1]
    Note that trimesh.verts_normals normal always from [-1, 1]  
    """

    def __init__(self, width: int, height: int, render_lib="face3d", flip=False):
        super(Noraml_Render, self).__init__(width, height, render_lib)
        self.__render = self.render_lib
        self.name = "Normal_render"
        self.flip_normal = flip
        self.input_para["flip_normal"] = flip
        self.render_image = None
        self.render_mask = None

    def set_attribute(self):
        """attribute you want to set when you render
        For normal render class, we only need normal information, which means 
        """
        # vertices = self.render_mesh.vertices
        #
        # [-1,1]
        # render_mesh !!!
	#self.z = self.render_mesh.vertices[:, 2:]
        h,w=self.height,self.width
        
        R=angle2matrix([0,0,0])
        t=[0,0,0]
        self.vertices = self.render_mesh.vertices[:,:]
        s=180/(np.max(self.vertices[:,1])-np.min(self.vertices[:,1]))
        transformed_vertices = similarity_transform(self.vertices,s,R,t)
        image_vertices = to_image(transformed_vertices,h,w)
        self.z = image_vertices[:,2:]
        self.z = self.z - np.min(self.z)
        self.z = self.z / (np.max(self.z)-np.min(self.z))
        self.z=self.z-1
        #self.z=get_z_value(self.z)
        self.attribute = self.z
        

####

    









    def get_render(self):
        """capture render images 
        Parameters:
            None
        return:
            render_image [H, W, C = 3]
            render_mask [H, W] ---> 0 means background else frontground
        """
        assert self.render_image is not None, "you need call draw method, previously"
        assert self.render_mask is not None, "you need call draw method, previously"
        return self.render_image, self.render_mask

    def draw(self):
        """draw render image,
        it needs you set_mesh, previously.
        after call draw method, you will get two attribute, render image and render mask
        """
        #####face3d 노말 랜더링 사용 부분
        if self.render_lib == "face3d":
            attribute = self.attribute
            #attribute = np.concatenate(
            #    [self.attribute, np.ones([self.attribute.shape[0], 1])], axis=1
            #)
            # render_colors face에서 컬러 합치는 부분 attribute => 노말
            # normal_map = mesh.render.render_colors(self.render_mesh.vertices, self.render_mesh.faces, attribute, self.height, self.width, c=4)
            depth_image = mesh.render.render_colors(
                self.render_mesh.vertices,
                self.render_mesh.faces,
                attribute,
                self.height,
                self.width,
                c=1,
            )
            #mask = depth_image[..., 3]
            mask = depth_image[...]
            print(np.shape(depth_image))
            #print(1111111111)
            depth_image = depth_image[...]
            #depth_image = depth_image[... 2:]
            #depth_image = (depth_image + 1) / 2
            # 배열 이미지로 변환 255는 정규화? 0~255 맞추기
            depth_image=np.squeeze(depth_image,axis=2)
            self.render_image = Image.fromarray((depth_image * 255).astype(np.uint8))
            self.render_mask = mask
        else:
            raise NotImplementedError
