##Depth_Render.py
"""
author: Ylab
data: 2022-03-11
render for Depth class
"""
import torch.nn as nn
import torch

# 기본으로 해줘야됨
from .Base_Render import _Base_Render

# 노말 체크
from .registry import NORMAL_RENDER

# face3D
from engineer.render.face3d import mesh
import numpy as np
from PIL import Image


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
        self.z = self.render_mesh.vertices[:, 2:]
        self.z = self.z - np.min(self.z)
        self.z = self.z / np.max(self.z)
        self.attribute = self.z

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
            attribute = np.concatenate(
                [self.attribute, np.ones([self.attribute.shape[0], 1])], axis=1
            )
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
            mask = depth_image[..., 3]
            depth_image = depth_image[..., 2:]
            depth_image = (depth_image + 1) / 2
            # 배열 이미지로 변환 255는 정규화? 0~255 맞추기
            self.render_image = Image.fromarray((depth_image * 255).astype(np.uint8))
            self.render_mask = mask
        else:
            raise NotImplementedError
