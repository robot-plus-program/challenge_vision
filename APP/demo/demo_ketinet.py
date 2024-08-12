import cv2
import kpick
import os
from ketisdk.vision.utils.rgbd_utils_v2 import RGBD
from APP.detector.grip.detector import ketinet
if __name__=='__main__':

    # configs
    depth_min = 500
    depth_max = 1000

    workspace = [(270,150), (980,150), (980,580), (270,580)]


    # load image
    rgb = cv2.imread("../data/2024_06_19_14_50_37__rgb.png")
    depth = cv2.imread("../data/2024_06_19_14_50_37__depth.png", cv2.IMREAD_UNCHANGED)
    rgbd = RGBD(rgb=rgb, depth=depth, depth_min=depth_min, depth_max=depth_max)
    rgbd.set_workspace(pts=workspace)

    # load model
    detector=ketinet()

    # parameter 설정
    detector.show_step(True)
    detector.set_width_range([(10,100)])
    detector.set_top_n(10)
    detector.set_angle_step(5)
    detector.set_threshold(0.7)
    detector.set_n_pose(400)

    #ketinet
    ret = detector.detect_and_show_grips(rgbd=rgbd)

    # show
    cv2.imshow('reviewer', ret['im'])
    cv2.waitKey(1000)
    print(ret)


