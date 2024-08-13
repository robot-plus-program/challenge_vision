from kpick.dataset.via_dataset import ViaAnnotation, get_via_annotation_default_args
from kpick.base.base import DetGuiObj
from ketisdk.utils.proc_utils import ProcUtils, CFG
import cv2
import math
import numpy as np
import os
import pickle


def get_grip_via_ann_default_args():
    args = get_via_annotation_default_args()

    # args.dataset.cifar_dir = 'cifar10'
    args.dataset.train_val_div = [5, 1]
    args.dataset.dber = 'GripCifar10'
    args.dataset.db_download = 0
    args.dataset.train_grip_hs = [13, 15, 17]
    args.dataset.train_grip_w_margins = [0, 2, 4]

    args.net = CFG()
    args.net.get_rgb= True
    args.net.get_depth = True
    args.net.depth2norm = True
    args.net.input_shape = 32,128,6

    return args


class GripViaAnnotation(ViaAnnotation):
    def init(self,ann_json,im_dir, input_shape, classes=None):
        ViaAnnotation.init(self,ann_json=ann_json,im_dir=im_dir, classes=classes)
        self.root_dir = os.path.split(self.im_dir)[0]
        self.cifar_dir = os.path.join(self.root_dir, 'cifar10')
        os.makedirs(self.cifar_dir, exist_ok=True)
        self.input_shape = input_shape

    def draw_region(self, im, X, Y, disp_args, color=(0, 255, 0)):
        out = im.copy()
        cv2.line(out, (X[-2], Y[-2]), (X[-1], Y[-1]), color, disp_args.line_thick)
        return out

    def to_train(self, count, train_val_div):
        dur = train_val_div[0] + train_val_div[1]
        return (count % dur) < train_val_div[0]

    def init_acc(self):
        self.acc = {'mean': 0.0, 'std': 0.0, 'num_train': 0, 'num_test': 0, 'total': 0}
        self.list_to_write = ['mean', 'std', 'num_train', 'num_test', 'total']

        # self.num_classes = len(self.args.classes)
        self.acc.update({'cls_inds': np.zeros((self.num_classes,), np.uint32)})
        self.list_to_write.append('cls_inds')

        self.acc.update({'train_array': [], 'test_array': [],
                         'train_filenames': [], 'test_filenames': [],
                         'train_labels': [], 'test_labels': []})


    def finalize_acc(self):
        # about
        about_db = open(os.path.join(self.root_dir, 'about_this_db'), 'w')
        for name in self.acc:
            if name not in self.list_to_write: continue
            about_db.write('%s:\t%s\n' % (str(name), str(self.acc[name])))
            np.save(os.path.join(self.cifar_dir, name + '.npy'), self.acc[name])
        about_db.close()

        out_dir = self.cifar_dir
        os.makedirs(out_dir, exist_ok=True)
        # train
        data = np.vstack(self.acc['train_array'])
        batch_label = 'training batch'
        data_dict = {'filenames': self.acc['train_filenames'],
                     'data': data,
                     'labels': self.acc['train_labels'],
                     'batch_label': batch_label}
        with open(os.path.join(out_dir, 'data_batch'), "wb") as f:
            pickle.dump(data_dict, f)
        # test
        data = np.vstack(self.acc['test_array'])
        batch_label = 'testing batch'
        data_dict = {'filenames': self.acc['test_filenames'],
                     'data': data,
                     'labels': self.acc['test_labels'],
                     'batch_label': batch_label}
        with open(os.path.join(out_dir, 'test_batch'), "wb") as f:
            pickle.dump(data_dict, f)
        # meta
        num_vis = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]  # 3*32*32
        label_names = self.classes
        num_cases_per_batch = self.acc['total']
        meta_dict = {'num_vis': num_vis,
                     'label_names': label_names,
                     'num_cases_per_batch': num_cases_per_batch}
        with open(os.path.join(out_dir, 'batches.meta'), "wb") as f:
            pickle.dump(meta_dict, f)

        print(f'Finished')

    def make_db(self, rgbd, filename, dataset_args, net_args, disp_args, disp_mode='rgb'):
        h, w = rgbd.height, rgbd.width
        diag = int(np.linalg.norm((h, w)))
        top, left = int((diag - h) / 2), int((diag - w) / 2)
        bottom, right = diag - h - top, diag - w - left
        center = (int(diag / 2), int(diag / 2))
        rgbd_pad = rgbd.pad(left=left, top=top, right=right, bottom=bottom)

        ret = self.get_regions(filename=filename)

        for reg, cls_ind in zip(ret['regions'], ret['indexes']):
            x1, x2, y1, y2 = reg[0][-2], reg[0][-1], reg[1][-2], reg[1][-1]
            dx, dy = x1 - x2, y1 - y2
            angle = math.atan(dy / (dx + 0.000001)) * 180 / math.pi

            # rotate
            rgbd_rot = rgbd_pad.rotate(angle=angle)

            #
            x1, y1 = x1 + left, y1 + top
            x1, y1 = ProcUtils().rotateXY(x1, y1, -angle, org=center)
            x2, y2 = x2 + left, y2 + top
            x2, y2 = ProcUtils().rotateXY(x2, y2, -angle, org=center)

            # out = self.draw_region(rgbd_rot.disp(mode=disp_mode), [x1, x2], [y1, y2], disp_args=disp_args)
            # cv2.imshow('out', out)
            # # cv2.waitKey()

            xmin, xmax = min(x1, x2), max(x1, x2)
            y = (y1 + y2) // 2
            # crop
            for grip_h in dataset_args.train_grip_hs:
                for grip_w_margin in dataset_args.train_grip_w_margins:
                    # count += 1
                    r = grip_h // 2
                    rgbd_crop = rgbd_rot.crop(left=xmin - grip_w_margin, top=y - r, right=xmax + 1 + grip_w_margin,
                                              bottom=y + r + 1)


                    # rgbd_crop.show(mode=disp_mode)
                    # cv2.waitKey()
                    # self.cls_ind = self.args.classes.index(cls)
                    # resize
                    rgbd_scaled = rgbd_rot.resize(net_args.input_shape[:2])
                    data =  rgbd_scaled.array(get_rgb=net_args.get_rgb, get_depth=net_args.get_depth,
                                      depth2norm=net_args.depth2norm)
                    h, w, num_ch = data.shape
                    data_1D_org = [data[:, :, ch].reshape((1, h * w)) for ch in range(num_ch)]

                    org_color_order = list(range(num_ch))
                    color_orders = [org_color_order, ]

                    if hasattr(dataset_args, 'aug_color_orders') and net_args.get_rgb:
                        for color_order in dataset_args.aug_color_orders:
                            color_orders.append(list(color_order) + org_color_order[3:])

                    for color_order in color_orders:
                        data_1D = [data_1D_org[ch] for ch in color_order]
                        data_1D = np.hstack(data_1D)

                        data_reorder = data[:, :, color_order]

                        # for ch in range(data.shape[2]):
                        #     ch_1D = data[:,:,ch].reshape((1, h * w))
                        #     if data_1D is None: data_1D = ch_1D
                        #     else: data_1D = np.concatenate((data_1D, ch_1D), axis=1)

                        # if hasattr(self, 'cls_ind'):
                        #     cls_ind = self.cls_ind
                        # else:
                        #     cls_ind = -1
                        self.acc['cls_inds'][cls_ind] += 1

                        # update
                        if self.to_train(self.acc['total'], dataset_args.train_val_div):
                            self.acc['train_array'].append(data_1D)
                            self.acc['train_filenames'].append(filename)
                            self.acc['train_labels'].append(cls_ind)

                            num_train = self.acc['num_train']
                            self.acc['mean'] = num_train / (num_train + 1) * self.acc['mean'] + \
                                               1 / (num_train + 1) * np.mean(data_reorder.astype('float') / 255,
                                                                             axis=(0, 1))
                            self.acc['std'] = num_train / (num_train + 1) * self.acc['std'] + \
                                              1 / (num_train + 1) * np.std(data_reorder.astype('float') / 255,
                                                                           axis=(0, 1))
                            self.acc['num_train'] += 1
                        else:
                            self.acc['test_array'].append(data_1D)
                            self.acc['test_filenames'].append(filename)
                            self.acc['test_labels'].append(cls_ind+1)

                            self.acc['num_test'] += 1

                        self.acc['total'] = self.acc['num_train'] + self.acc['num_test']




class GripViaAnnotationGuiObj(GripViaAnnotation, DetGuiObj):
    def __init__(self, args=None, cfg_path=None, name='unnamed', default_args=None):
        DetGuiObj.__init__(self, args=args, cfg_path=cfg_path, name=name,
                           default_args=get_grip_via_ann_default_args())
        GripViaAnnotation.init(self, ann_json=self.args.dataset.ann_json, im_dir=self.args.dataset.im_dir,
                               classes=self.args.dataset.classes, input_shape=self.args.net.input_shape)

    def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', **kwargs):
        if method_ind == 0:
            ret = self.show_single_ann(rgbd=rgbd, filename=filename, disp_args=self.args.disp, disp_mode=disp_mode)
        if method_ind == 1:
            ret = self.make_db(rgbd=rgbd, filename=filename, dataset_args=self.args.dataset,
                               disp_args=self.args.disp, net_args=self.args.net, disp_mode=disp_mode)

        return ret


def demo_grip_via_annocation_gui(cfg_path='configs/grip_via_annotation.cfg',
                                 default_cfg_path='configs/default_via_annotation.cfg',
                                 data_root=None, rgb_formats=None, depth_formats=None):
    from ketisdk.gui.gui import GUI, GuiModule

    module = GuiModule(GripViaAnnotationGuiObj, name='Grip Via Annotation',
                       cfg_path=cfg_path, num_method=3,
                       key_args=['dataset.ann_json', 'dataset.im_dir']
                       )

    GUI(title='Grip Via Ann GUI', default_cfg_path=default_cfg_path,
        modules=[module], data_root=data_root, rgb_formats=rgb_formats, depth_formats=depth_formats,
        )


if __name__ == '__main__':
    demo_grip_via_annocation_gui(cfg_path='configs/grip_via_dataset.cfg',
                                 default_cfg_path='configs/default_via_annotation.cfg',
                                 data_root='data/grip_dataset/20220616/', rgb_formats='rgb/*', depth_formats='depth/*')
