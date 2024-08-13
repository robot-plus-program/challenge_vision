import numpy as np
import os
import cv2
from ketisdk.vision.utils.rgbd_utils_v2 import RGBD
import random

def get_default_configs():
    from ketisdk.gui.default_config import default_args

    args = default_args()

    return args



class CornellDataset():
    def infer_path(self, src_path, src_format, dst_format):
        dst_path = src_path
        for s,d in zip(src_format.split('*'), dst_format.split('*')):
            dst_path = dst_path.replace(s, d)
        return dst_path

    def show_apose(self, im, pos, neg_pose=False, color=None):
        out = np.copy(im)
        p0, p1, p2, p3 = tuple(pos[:2]), tuple(pos[2:4]), tuple(pos[4:6]), tuple(pos[6:8])
        if color is None: color = (255,0,0) if neg_pose else (0,255,0)
        color1 = (0,0,0)
        cv2.line(out, p0, p1, color1, 1)
        cv2.line(out, p1, p2, color, 2)
        cv2.line(out, p2, p3, color1, 1)
        cv2.line(out, p3, p0, color, 2)
        return out

    def get_poses(self, filepath):
        return np.loadtxt(filepath).astype('int').reshape(-1,8)

    def get_instance(self, rgbFilePath, rgb_format='*r.png', depth_format='*d.tiff',
                     neg_format='*cneg.txt', pos_format='*cpos.txt'):
        rgb = cv2.imread(rgbFilePath)[:,:,::-1]

        depth_path = self.infer_path(rgbFilePath, rgb_format, depth_format)
        if not os.path.exists(depth_path):
            print(f'{depth_path} does not exist ...')
            return
        depth = (1000*cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)).astype('uint16')

        neg_path = self.infer_path(rgbFilePath, rgb_format, neg_format)
        if not os.path.exists(neg_path):
            print(f'{neg_path} does not exist ...')
            return
        neg_poses = self.get_poses(filepath=neg_path)

        pos_path = self.infer_path(rgbFilePath, rgb_format, pos_format)
        if not os.path.exists(pos_path):
            print(f'{pos_path} does not exist ...')
            return
        pos_poses = self.get_poses(filepath=pos_path)

        return {'rgb': rgb, 'depth': depth, 'neg_poses': neg_poses, 'pos_poses': pos_poses}


    def show_instance(self, rgbFilePath, rgb_format='*r.png', depth_format='*d.tiff',
                     neg_format='*cneg.txt', pos_format='*cpos.txt'):
        instance = self.get_instance(rgbFilePath=rgbFilePath, depth_format=depth_format,
                                     neg_format=neg_format, pos_format=pos_format)
        rgbd = RGBD(rgb=instance['rgb'], depth=instance['depth'], depth_min=np.amin(instance['depth']),
                    depth_max=np.amax(instance['depth']))
        out1, out2 = rgbd.disp(), rgbd.disp(mode='depth')
        for pos in instance['neg_poses']:
            out1 = self.show_apose(out1, pos, neg_pose=True)
            out2 = self.show_apose(out2, pos, neg_pose=True)
        for pos in instance['pos_poses']:
            out1 = self.show_apose(out1, pos, neg_pose=False)
            out2 = self.show_apose(out2, pos, neg_pose=False)

        return np.concatenate((out1, out2), axis=1)

    def pose2Pose5D(self, pose):
        # Input: pose = np.array: num_pose x (x0, y0, x1, y1, x2, y2, x3, y3)
        # Output: pose5D = np.array: num_pose x (xc, yc, theta, w, h)
        X, Y = pose[:, ::2], pose[:, 1::2]
        Xc, Yc = np.mean(X, axis=1, keepdims=True), np.mean(Y, axis=1, keepdims=True)

        dX, dY = X[:,[1]]-X[:,[0]], Y[:,[1]] - Y[:,[0]]
        Theta = np.arctan(np.divide(dY, dX + 0.0001)) * 180/np.pi

        W = np.linalg.norm(pose[:,:2] - pose[:, 2:4], axis=1, keepdims=True)
        H = np.linalg.norm(pose[:,2:4] - pose[:, 4:6], axis=1, keepdims=True)

        return np.concatenate((Xc, Yc, Theta, W, H), axis=1)
    def pose5D2pose(self, pose5D):
        Xc, Yc, Theta = pose5D[:, [0]], pose5D[:, [1]], pose5D[:, [2]]
        W, H  = pose5D[:, [-2]], pose5D[:, [-1]]
        dX, dY = W/2, H/2

        P0= np.concatenate((-dX, -dY), axis=1)[..., np.newaxis]
        P1= np.concatenate((dX, -dY), axis=1)[..., np.newaxis]
        P2= np.concatenate((dX, dY), axis=1)[..., np.newaxis]
        P3= np.concatenate((-dX, dY), axis=1)[..., np.newaxis]
        P = np.concatenate((P0,P1, P2, P3), axis=2)

        CosT = np.cos(Theta * np.pi / 180)
        SinT = np.sin(Theta * np.pi / 180)

        out = []
        for c, s, p in zip(CosT.flatten(), SinT.flatten(), P):
            R = np.array([[c, -s], [s, c]])
            p1 = np.dot(R, p).transpose().reshape((1, -1))
            out.append(p1)
        out = np.concatenate(out, axis=0)
        out[:, ::2] += Xc
        out[:, 1::2] += Yc

        return out.astype('int')

    def movePose5D(self, pose5D, max_trans=10, max_rot=3):
        n = pose5D.shape[0]
        pose5D[:, 0] += max_trans*(2*np.random.rand(n)-1)
        pose5D[:, 1] += max_trans*(2*np.random.rand(n)-1)
        pose5D[:, 2] += max_rot*(2*np.random.rand(n)-1)

        mean_angle = np.mean(pose5D[:, 2])
        pose5D[:, 2] = mean_angle
        return pose5D

from kpick.base.base import DetGuiObj
class CornellDatasetGui(CornellDataset, DetGuiObj):
    def __init__(self, args=None, cfg_path=None, name='unnamed', default_args=None):
        DetGuiObj.__init__(self,args=args, cfg_path=cfg_path, name=name,
                           default_args=get_default_configs())

    def show_poses(self, rgbd, filename='unnamed.png', disp_mode='rgb', im_dir=''):
        instance = self.get_instance(rgbFilePath=os.path.join(im_dir,filename))
        out = rgbd.disp(mode=disp_mode)

        for pos in instance['neg_poses']:
            out = self.show_apose(out, pos, neg_pose=True)
        for pos in instance['pos_poses']:
            out = self.show_apose(out, pos, neg_pose=False)

        return {'im': out}

    def show_positive_poses(self, rgbd, filename='unnamed.png', disp_mode='rgb', im_dir=''):
        instance = self.get_instance(rgbFilePath=os.path.join(im_dir,filename))
        out = rgbd.disp(mode=disp_mode)

        poses = instance['pos_poses']

        for pos in poses:
            out = self.show_apose(out, pos, neg_pose=False)

        return {'im': out}

    def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', **kwargs):
        if method_ind==0:
            ret=self.show_poses(rgbd=rgbd, filename=filename, disp_mode=disp_mode, im_dir=kwargs['im_dir'])
        if method_ind==1:
            ret=self.show_positive_poses(rgbd=rgbd, filename=filename, disp_mode=disp_mode, im_dir=kwargs['im_dir'])
        return ret

def demo_cornell_gui(cfg_path=None):
    from ketisdk.gui.gui import GuiModule, GUI
    module = GuiModule(CornellDatasetGui, name='Cornell', cfg_path=cfg_path, num_method=3)
    GUI(title='Cornell GUI', modules=[module,], data_root='data/cornell',
        default_cfg_path='kpick/apps/configs/cornell_dataset.cfg',
        rgb_formats=['*/*r.png'], depth_formats=['*/*d.tiff'])

def test():
    out = CornellDataset().show_instance(rgbFilePath='/home/keti/000_workspace/000_data/cornell_grasp/01/pcd0101r.png')
    cv2.imshow('viewer', out[:, :, ::-1])
    cv2.waitKey()

if __name__=='__main__':
    # CornellDatasetGui()
    demo_cornell_gui()
