import json
import os
from ketisdk.gui.default_config import default_args
from ketisdk.utils.proc_utils import CFG
import cv2


def get_via_annotation_default_args():
    args = default_args()

    args.dataset = CFG()
    args.dataset.ann_json = 'data/grip_dataset/20220616/ann.json'
    args.dataset.im_dir = 'data/grip_dataset/20220616/disp'
    args.dataset.classes = ['grip', 'ungrip']

    args.save = CFG()
    args.save.im_dir = 'data/grip_dataset/20220616/aug0'

    args.disp.class_colors = [(0,255,0), (0,0,255)]

    return args


class ViaAnnotation():
    def init(self, ann_json, im_dir, classes=None):
        if not os.path.exists(ann_json):
            print(f'{ann_json} does not exist')
            return

        def rename(key):
            name, ext = os.path.splitext(key)
            return name + ext[:4]
        ann = json.load(open(ann_json, 'r'))
        self.ann = {}
        for key in ann:
            self.ann.update({rename(key): ann[key]})

        self.im_dir = im_dir
        self.classes = classes
        self.num_classes = len(classes) if classes else 0


    def draw_region(self, im, X,Y, disp_args, color=(0,255,0)):
        out = im.copy()
        num_el = len(X)-1
        for i in range(num_el):
            cv2.line(out, (X[i], Y[i]), (X[i+1], Y[i+1]), color, disp_args.line_thick)

        if num_el >2:
            cv2.line(out, (X[-1], Y[-1]), (X[0], Y[0]), color, disp_args.line_thick)
        return out

    def get_regions(self, filename):
        regions, indexes = [], []
        if filename not in self.ann:
            print(f'No annotation for {filename}')
        else:
            for reg in self.ann[filename]['regions']:
                X, Y = reg['shape_attributes']['all_points_x'], reg['shape_attributes']['all_points_y']
                type = reg['region_attributes']['type']
                regions.append([X,Y])
                indexes.append(int(type)-1)
        return {'regions': regions, 'indexes': indexes}

    def show_single_ann(self, rgbd, filename, disp_args, disp_mode='rgb'):
        out = rgbd.disp(mode=disp_mode)
        ret = self.get_regions(filename=filename)
        for reg, ind in zip(ret['regions'], ret['indexes']):
            color = disp_args.class_colors[ind]
            out = self.draw_region(out, reg[0], reg[1], disp_args=disp_args, color=color)
        return {'im':out}

from kpick.base.base import DetGuiObj


class ViaAnnotationGuiObj(ViaAnnotation, DetGuiObj):
    def __init__(self, args=None, cfg_path=None, name='unnamed', default_args=None):
        DetGuiObj.__init__(self, args=args, cfg_path=cfg_path, name=name,
                           default_args=get_via_annotation_default_args())
        ViaAnnotation.init(self, ann_json=self.args.dataset.ann_json, im_dir=self.args.dataset.im_dir,
                           classes=self.args.dataset.classes)

    def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', **kwargs):
        if method_ind==0:
            ret = self.show_single_ann(rgbd=rgbd, filename=filename, disp_args=self.args.disp, disp_mode=disp_mode)

        return ret

def demo_via_annocation_gui(cfg_path='configs/via_annotation.cfg', default_cfg_path='configs/default_via_annotation.cfg',
                            data_root=None, rgb_formats=None, depth_formats=None):
    from ketisdk.gui.gui import GUI, GuiModule

    module = GuiModule(ViaAnnotationGuiObj, name='Via Annotation',
                       cfg_path=cfg_path, num_method=3,
                       key_args=['dataset.ann_json', 'dataset.im_dir']
                       )

    GUI(title='Grip Detection GUI', default_cfg_path=default_cfg_path,
        modules=[module], data_root=data_root, rgb_formats=rgb_formats, depth_formats=depth_formats,
        )


if __name__ == '__main__':
    demo_via_annocation_gui()
