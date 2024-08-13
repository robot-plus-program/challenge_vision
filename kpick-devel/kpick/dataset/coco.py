import glob

from ketisdk.vision.utils.data_utilts import JsonUtils
import os
import pycocotools.mask as cocoMaskUtils
# from skimage import measure
# from shapely.geometry import Polygon, MultiPolygon
# from datetime import datetime
import cv2
import numpy as np
from copy import deepcopy
from ketisdk.gui.default_config import default_args
from ketisdk.utils.proc_utils import CFG
from scipy.ndimage import rotate
from shutil import copyfile
from kpick.base.base import DetGuiObj
from ketisdk.utils.proc_utils import ProcUtils
from ketisdk.vision.utils.image_processing import ImageProcessing

BASE_CFGS = default_args()
# BASE_CFGS.path.ann_files = ['data/coco/annotations/instances_val2017.json']
BASE_CFGS.path.ann_files = ['data/coco/annotations/person_keypoints_val2017.json']
BASE_CFGS.path.im_dirs = ['data/coco/val2017']

BASE_CFGS.save = CFG()
BASE_CFGS.save.aug_dirs = ['data/aug0', ]
BASE_CFGS.save.extend = '.png'
BASE_CFGS.save.compress_level = 0

BASE_CFGS.rotate = CFG()
BASE_CFGS.rotate.angles = [0, 30, 60, 120, 150]
BASE_CFGS.rotate.reshape = True
BASE_CFGS.rotate.order = 3
BASE_CFGS.rotate.aug_angle_step = 5

BASE_CFGS.color = CFG()
BASE_CFGS.color.brightnesss = [0.3, ]
BASE_CFGS.color.contrasts = [0.3, ]
BASE_CFGS.color.saturations = [0.3, ]
BASE_CFGS.color.hues = [0.2, ]

BASE_CFGS.flip = CFG()
BASE_CFGS.flip.axis = 0


class CocoUtils():
    def __init__(self, ann_path=None):
        self.ann_path = ann_path
        if ann_path is not None:
            if not os.path.exists(ann_path):
                print(f'{ann_path} does not exist ...')
                return
            ret = JsonUtils().read_json(ann_path)
            self.image_dict_list, self.annotation_dict_list, self.categories = ret['images'], ret['annotations'], ret[
                'categories']
            self.image_inds = [im_dict['id'] for im_dict in self.image_dict_list]
            print(f'{ann_path} loaded ...')

    def segmToRLE(self, segm, h, w):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = cocoMaskUtils.frPyObjects(segm, h, w)
            rle = cocoMaskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = cocoMaskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = segm
        return rle

    def segmToMask(self, segm, h, w):
        rle = self.segmToRLE(segm, h, w)
        return cocoMaskUtils.decode(rle)

    def get_im_path(self, image_id, im_dir):
        im_dict = self.image_dict_list[self.image_inds.index(image_id)]
        return os.path.join(im_dir, im_dict['file_name'])

    def get_im_from_ind(self, image_id, im_dir):
        im_path = self.get_im_path(image_id, im_dir)
        print(im_path)
        return cv2.imread(im_path)[:, :, ::-1]

    def get_instance_from_ind(self, ind, ann_path=None):
        if ann_path is not None:
            ret = JsonUtils().read_json(ann_path)
            annotation_dict_list = ret['annotations']
        else:
            annotation_dict_list = self.annotation_dict_list

        ann_dict = annotation_dict_list[ind]
        segm = ann_dict['segmentation']
        kpts = ann_dict['keypoints']
        bbox = ann_dict['bbox']
        image_id = ann_dict['image_id']
        category_id = ann_dict['category_id']

        return {'image_id': image_id, 'segmentation': segm, 'bbox': bbox, 'keypoints': kpts, 'category_id': category_id}

    def make_ann_categories(self, classes, kpt_labels=None, kpt_skeletons=None):
        categories = []
        for j, cls in enumerate(classes):
            category = dict()
            category.update({"supercategory": cls, "id": int(j + 1), "name": cls})
            if kpt_labels is not None: category.update({'keypoints': kpt_labels[j]})
            if kpt_skeletons is not None: category.update({'skeleton': kpt_skeletons[j]})
            categories.append(category)
        return categories

    def make_ann_images(self, image_id, im_path, im_size, image_dict_list=[]):
        _, filename = os.path.split(im_path)
        image_dict_list.append({
            "id": int(image_id),
            "license": int(1),
            "coco_url": im_path,
            "flickr_url": "keti.re.kr",
            "width": int(im_size[0]),
            "height": int(im_size[1]),
            "file_name": filename,
            "date_captured": "unknown"
        })
        return image_dict_list

    def binary_mask_to_rle(self, binary_mask):
        from itertools import groupby
        shape = [int(s) for s in binary_mask.shape]
        rle = {'counts': [], 'size': shape}
        counts = rle.get('counts')
        for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
            if i == 0 and value == 1:
                counts.append(0)
            counts.append(int(len(list(elements))))
        return rle

    def make_annotations(self, mask, bbox, ann_id, image_id, cls_id, keypoints=None, annotations=[]):
        h, w = mask.shape[:2]
        rles = cocoMaskUtils.encode(np.asfortranarray(mask))
        area = cocoMaskUtils.area(rles)
        segm = self.binary_mask_to_rle(mask)

        annotation = {
            'segmentation': segm,
            'iscrowd': int(0),
            'image_id': int(image_id),
            'category_id': int(cls_id),
            'id': int(ann_id + 1),
            'bbox': [int(p) for p in bbox],
            'area': int(area)
        }
        if keypoints is not None:
            keypoints = [int(p) for p in keypoints]
            annotation.update({'keypoints': keypoints})
        annotations.append(annotation)
        return annotations

    def visualize_instance(self, rgb, instance, categories=None):

        if categories is None: categories = self.categories
        out = np.copy(rgb)
        h, w = rgb.shape[:2]

        # mask
        if 'segmentation' in instance:
            segm = instance['segmentation']
            mask = self.segmToMask(segm, h, w)
            locs = np.where(mask > 0)
            out[locs] = 0.7 * out[locs] + (0, 75, 0)

        # bbox
        if 'bbox' in instance:
            bbox = instance['bbox']
            left, top, w, h = np.array(bbox).astype('int')
            cv2.rectangle(out, (left, top), (left + w, top + h), (0, 255, 0), 2)

            category_id = instance['category_id']
            if category_id is not None:
                for cat in self.categories:
                    if cat['id'] == category_id:
                        cv2.putText(out, cat['name'], (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        if 'keypoints' in instance:
            kpts = instance['keypoints']
            cat = categories[0]
            kpt_labels = cat['keypoints']
            kpt_skeleton = cat['skeleton']

            # keypoint
            X, Y, V = kpts[::3], kpts[1::3], kpts[2::3]
            for x, y, v in zip(X, Y, V):
                if v == 0: continue
                cv2.drawMarker(out, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 5, 2)

            # skeleton
            for link in kpt_skeleton:
                i1, i2 = link[0] - 1, link[1] - 1
                if V[i1] == 0 or V[i2] == 0: continue
                x1, y1, x2, y2 = X[i1], Y[i1], X[i2], Y[i2]
                cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return out

    def save(self, ann_path, images=None, annotations=None, categories=None, ann_info=None, ann_license=None):
        from ketisdk.utils.proc_utils import ProcUtils
        import json
        if ann_info is None:
            ann_info = {"info": {
                "description": "KETI Dataset",
                "url": "keti.re.kr",
                "version": "1.0",
                "year": int(ProcUtils().get_current_time_str('%Y')),
                "contributor": "Trung Bui",
                "data_create": '{}/{}/{}'.format(ProcUtils().get_current_time_str('%Y'),
                                                 ProcUtils().get_current_time_str('%m'),
                                                 ProcUtils().get_current_time_str('%d'))
            }}

        if ann_license is None:
            ann_license = {"licenses": [
                {"url": "keti.re.kr",
                 "id": "1",
                 "name": "Atribution license"
                 }]}

        if images is None: images = self.image_dict_list
        if categories is None: categories = self.categories
        if annotations is None: annotations = self.annotation_dict_list

        ann_dict = dict()
        ann_dict.update(ann_info)
        ann_dict.update(ann_license)
        ann_dict.update({"images": images})
        ann_dict.update({"categories": categories})
        ann_dict.update({"annotations": annotations})

        save_dir, _ = os.path.split(ann_path)
        os.makedirs(save_dir, exist_ok=True)
        instance_json_obj = open(ann_path, 'w')
        instance_json_obj.write(json.dumps(ann_dict))
        instance_json_obj.close()

        print('{} {} saved'.format('+' * 10, ann_path))

    def show_instances(self, im_dir, title='coco_viewer', im_size=(1080, 720)):

        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, im_size[0], im_size[1])

        for j, instance in enumerate(self.annotation_dict_list):
            print('ann_ind: {}'.format(j))
            rgb = self.get_im_from_ind(image_id=instance['image_id'], im_dir=im_dir)
            out = self.visualize_instance(rgb, instance)
            cv2.imshow(title, out[:, :, ::-1])
            if cv2.waitKey() == 27: exit()

    def show_ims(self, im_dir, title='coco_viewer', im_size=(1080, 720), start_from=0, step=1):
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, im_size[0], im_size[1])
        num_im = len(self.image_dict_list)
        for j in range(0, num_im, step):
            im_dict = self.image_dict_list[j]
            if j<start_from: continue
            im_id = im_dict['id']
            im_path = os.path.join(im_dir, im_dict['file_name'])
            print(f'[{j}/{num_im}] {im_path}')
            out = np.copy(cv2.imread(im_path)[:, :, ::-1])
            for ann_dict in self.annotation_dict_list:
                if im_id != ann_dict['image_id']: continue
                out = self.visualize_instance(out, ann_dict)
            cv2.imshow(title, out[:, :, ::-1])
            if cv2.waitKey() == 27: exit()

    def aug_single(self, im_path, im_id, bg_ims=None, angle_step=10, show_step=False):
        from scipy.ndimage import rotate
        from shutil import copyfile
        from ketisdk.utils.proc_utils import ProcUtils
        # makedir
        im_dir = os.path.split(im_path)[0]
        root_dir, dir_name = os.path.split(im_dir)
        save_dir = os.path.join(root_dir, f'{dir_name}_aug')
        os.makedirs(save_dir, exist_ok=True)

        # read image
        im = cv2.imread(im_path)[:, :, ::-1]
        im_height, im_width = im.shape[:2]
        if show_step: cv2.imshow('im', im[:, :, ::-1])

        # get instances
        instances = [instance for instance in self.annotation_dict_list if im_id == instance['image_id']]

        for angle in range(0, 360, angle_step):
            im_out_path = os.path.join(save_dir, ProcUtils().get_current_time_str() + '.png')

            # make image
            if angle == 0:
                copyfile(im_path, im_out_path)
                im_rot = np.copy(im)
            else:
                im_rot = np.copy(rotate(im, angle=angle, reshape=False, order=3))
                cv2.imwrite(im_out_path, im_rot[:, :, ::-1])

            self.make_ann_images(self.image_id, im_out_path, (im_width, im_height), self.out_images)

            # make annotation
            for instance in instances:
                mask, bbox = self.segmToMask(instance['segmentation'], h=im_height, w=im_width), instance['bbox']
                if angle != 0:
                    mask = rotate(mask, angle=angle, reshape=False, order=0)
                    Y, X = np.where(mask > 0)
                    if len(Y) == 0: continue
                    x, y = np.amin(X), np.amin(Y)
                    w, h = np.amax(X) - x, np.amax(Y) - y
                    bbox = [x, y, w, h]

                if show_step:
                    locs = np.where(mask > 0)
                    im_rot[locs] = 0.7 * im_rot[locs] + (0, 75, 0)
                    cv2.rectangle(im_rot, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
                    # cv2.rectangle(im_rot,(bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 2)
                    cv2.imshow('im_rot', im_rot[:, :, ::-1])
                    cv2.waitKey()

                self.make_annotations(mask, bbox, self.ann_id, self.image_id, instance['category_id'],
                                      annotations=self.out_annotations)
                self.ann_id += 1

            self.image_id += 1

    def augmentation(self, im_dir, ann_path, bg_dir=None, angle_step=10, show_step=False):
        if not os.path.exists(im_dir):
            print(f'{"+" * 10} {im_dir} not exist')
            return

        # read background images
        if bg_dir is not None:
            from glob import glob
            bgs = [cv2.imread(path)[:, :, ::-1] for path in glob(os.path.join(bg_dir, '*'))]

        self.out_annotations, self.out_images = [], []
        self.image_id, self.ann_id = 1, 1
        num_im = len(self.image_dict_list)
        # query images
        for j, im_dict in enumerate(self.image_dict_list):
            im_id = im_dict['id']
            im_path = os.path.join(im_dir, im_dict['file_name'])
            print(f'[{j}/{num_im}] {im_path}')

            self.aug_single(im_path=im_path, im_id=im_id, angle_step=angle_step, show_step=show_step)

        # save annotation

        self.save(ann_path, images=self.out_images, annotations=self.out_annotations)

    def split_trainval_(self, div=(2, 1)):

        dur = div[0] + div[1]
        train_im_dict_list, val_im_dict_list = [], []
        train_ann_dict_list, val_ann_dict_list = [], []

        # split
        num_im = len(self.image_dict_list)
        for j, im_dict in enumerate(self.image_dict_list):
            if (j % 100 == 0): print(f'[{j}/{num_im}] done')
            instances = [instance for instance in self.annotation_dict_list if im_dict['id'] == instance['image_id']]

            totrain = (j % dur < div[0])
            if totrain:
                train_im_dict_list.append(im_dict)
                train_ann_dict_list += instances
            else:
                val_im_dict_list.append(im_dict)
                val_ann_dict_list += instances

        # save
        self.save(ann_path=self.ann_path.replace('.json', '_train.json'), images=train_im_dict_list,
                  annotations=train_ann_dict_list, categories=self.categories)
        self.save(ann_path=self.ann_path.replace('.json', '_val.json'), images=val_im_dict_list,
                  annotations=val_ann_dict_list, categories=self.categories)

    def split_trainval(self, div=(2, 1)):

        num_im = len(self.image_dict_list)
        im_inds = np.arange(num_im)
        # im_inds = [d['id'] for d in self.image_dict_list]
        train_im_inds = np.random.choice(im_inds, int(len(im_inds) * div[0] / (div[0] + div[1])), replace=False)
        val_im_inds = np.setxor1d(im_inds, train_im_inds)

        train_im_dict_list = [self.image_dict_list[i] for i in train_im_inds]
        val_im_dict_list = [self.image_dict_list[i] for i in val_im_inds]

        train_im_ids = [d['id'] for d in train_im_dict_list]
        num_ann = len(self.annotation_dict_list)
        ann_inds = np.arange(num_ann)

        train_ann_inds = [j for j,ins in enumerate(self.annotation_dict_list) if ins['image_id'] in train_im_ids]
        val_ann_inds = np.setxor1d(ann_inds, train_ann_inds)

        train_ann_dict_list = [self.annotation_dict_list[i] for i in train_ann_inds]
        val_ann_dict_list = [self.annotation_dict_list[i] for i in val_ann_inds]

        # save
        self.save(ann_path=self.ann_path.replace('.json', '_train.json'), images=train_im_dict_list,
                  annotations=train_ann_dict_list, categories=self.categories)
        self.save(ann_path=self.ann_path.replace('.json', '_val.json'), images=val_im_dict_list,
                  annotations=val_ann_dict_list, categories=self.categories)

    def filename2Id(self, filename):
        for j, im_dict in enumerate(self.image_dict_list):
            if im_dict['file_name'] == filename:
                return im_dict['id']

    def filename2Instances(self, filename):
        im_id = self.filename2Id(filename)
        return [ann_dict for ann_dict in self.annotation_dict_list if ann_dict['image_id'] == im_id]

    def show_ann_single(self, im, filename):
        out = im.copy()
        instances = self.filename2Instances(filename)
        for ann_dict in instances:
            out = self.visualize_instance(out, ann_dict)
        return out

    def make_categories(self, classes, supercategories=None):
        categories = []
        for j, cl in enumerate(classes):
            cat = {'id': j + 1}
            cat.update({'name': cl})
            cat.update({'supercategory': supercategories[j] if supercategories else 'unknown'})
            categories.append(cat)
        return categories

    def init_acc(self, save_dirs, classes,aug_args=None, supercategories=None):
        self.categories = self.make_categories(classes, supercategories=supercategories)
        if aug_args is not None:
            self.im_processor = ImageProcessing()
            self.aug_params = self.im_processor.make_augmentation_params(aug_args.brightnesss,
                                                                         aug_args.contrasts,
                                                                         aug_args.saturations,
                                                                         aug_args.hues,
                                                                         range(aug_args.angle_range[0],
                                                                               aug_args.angle_range[1],
                                                                               aug_args.angle_range[2]),
                                                                         aug_args.flip_axes
                                                                         )
            print(f'1-->{len(self.aug_params[0])} augmentation params initialized ...')

        self.save_dirs = save_dirs
        for save_dir in self.save_dirs:
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(os.path.join(save_dir, 'imgs'), exist_ok=True)
        self.num_aug_dir = len(self.save_dirs)

        self.aug_images, self.aug_anns = [], []
        self.im_id, self.ann_id = 0, 0

    def finalize_acc(self):
        if len(self.aug_images) > 0:
            if hasattr(self, 'aug_params'):
                self.__delattr__('aug_params')
                print('Augmentation params removed ...')
            aug_ann_path = os.path.join(self.save_dirs[0], 'ann_aug.json')
            CocoUtils().save(aug_ann_path, images=self.aug_images, annotations=self.aug_anns,
                             categories=self.categories)
            print(f'{aug_ann_path} saved ...')

            aug_coco = CocoUtils(ann_path=aug_ann_path)
            aug_coco.split_trainval()
            aug_coco.show_ims(im_dir=os.path.join(self.save_dirs[0], 'imgs'))


    def acc_single_db(self, rgb, masks, category_ids, sub_thread=None):


        if len(masks) == 0: return

        # select saving dir
        save_dir = self.save_dirs[0] if self.num_aug_dir == 1 or sub_thread is None \
            else self.save_dirs[sub_thread % self.num_aug_dir]


        # accumalte images
        im_height, im_width = rgb.shape[:2]
        filepath = os.path.join(save_dir,'imgs',  ProcUtils().get_current_time_str() + '.png')
        cv2.imwrite(filepath, rgb[:, :, ::-1])
        self.im_id += 1
        im_id = self.im_id  # for threading
        CocoUtils().make_ann_images(im_id, filepath, (im_width, im_height), self.aug_images)

        # accumalate ann
        for mask, category_id in zip(masks, category_ids):
            Y, X = np.where(mask > 0)
            if len(X)==0:
                continue
            xmin, ymin, xmax, ymax = np.amin(X), np.amin(Y), np.amax(X), np.amax(Y)
            w, h = xmax - xmin, ymax - ymin
            bbox = [xmin, ymin, w, h]
            self.ann_id += 1
            ann_id = self.ann_id  # for threading
            CocoUtils().make_annotations(mask, bbox, ann_id, im_id, category_id,annotations=self.aug_anns)


    def aug_mask(self, mask, angle, flip_axis):
        mask_rot = mask.copy() if angle == 0 else rotate(mask, angle=angle, reshape=True, order=0)
        # cv2.imshow('mask_rot', 255*mask_rot)
        if flip_axis == 0:
            mask_flip = mask_rot[::-1, :]
        elif flip_axis == 1:
            mask_flip = mask_rot[:, ::-1]
        else:
            mask_flip = mask_rot.copy()

        return mask_flip

    def aug_category(self, category_id, angle, flip_axis):
        return category_id


    def acc_augmentation_db(self, rgb, masks, category_ids, sub_thread=None):
        count = 0
        # anns = []
        for brightness, contrast, saturation, hue, angle, ax in \
                zip(self.aug_params[0], self.aug_params[1], self.aug_params[2],
                    self.aug_params[3], self.aug_params[4], self.aug_params[5]):
            color_params_ = {'brightness': brightness,
                             'contrast': contrast,
                             'saturation': saturation,
                             'hue': hue}

            for color_params in [None, color_params_]:
                rotate_params = {'angle': angle, 'reshape': True, 'order': 3}
                flip_params = {'axis': ax}

                count += 1
                print(f'color_params: {color_params}')
                print(f'rotate_params: {rotate_params}')
                print(f'flip_params: {flip_params}')

                # augmenting image and save
                out = self.im_processor.augmentation(rgb, color_params=color_params,
                                                     rotate_params=rotate_params,
                                                     flip_params=flip_params)

                mask_outs, category_ids_out = [], []
                for mask, category_id in zip(masks, category_ids):
                    mask_out = self.aug_mask(mask, angle, ax)
                    mask_outs.append(mask_out)
                    category_out = self.aug_category(category_id, angle, ax)
                    category_ids_out.append(category_out)

                    # print(f'angle: {angle}, flip axis: {ax}, org cat: {category_id}')
                    # out = cv2.cvtColor(255*mask_out, cv2.COLOR_GRAY2BGR)
                    # cv2.putText(out, f'{category_out}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,0),2)
                    # cv2.imshow('mask', 255*mask)
                    # cv2.imshow('mask_out', out)
                    # cv2.waitKey()
                self.acc_single_db(rgb=out, masks=mask_outs, category_ids=category_ids_out)
                # anns.append([out, mask_outs, category_ids_out])
        # return anns

    def replace_background(self, im_dir, background_im, select_im_inds=None, div=(2,1)):
        # get annotations
        if select_im_inds is None: select_im_inds = np.arange(len(self.image_dict_list))

        mix_im_inds = np.array(select_im_inds.tolist() + (select_im_inds+1).tolist())


        train_im_inds = np.random.choice(mix_im_inds, int(len(mix_im_inds) * div[0] / (div[0] + div[1])), replace=False)
        val_im_inds = np.setxor1d(mix_im_inds, train_im_inds)

        train_im_dict_list = [self.image_dict_list[i] for i in train_im_inds]
        val_im_dict_list = [self.image_dict_list[i] for i in val_im_inds]

        train_im_ids = [d['id'] for d in train_im_dict_list]
        val_im_ids = [d['id'] for d in val_im_dict_list]

        train_ann_inds = [j for j, ins in enumerate(self.annotation_dict_list) if ins['image_id'] in train_im_ids]
        val_ann_inds = [j for j, ins in enumerate(self.annotation_dict_list) if ins['image_id'] in val_im_ids]

        train_ann_dict_list = [self.annotation_dict_list[i] for i in train_ann_inds]
        val_ann_dict_list = [self.annotation_dict_list[i] for i in val_ann_inds]

        self.save(ann_path=self.ann_path.replace('.json', '_bg_train.json'), images=train_im_dict_list,
                  annotations=train_ann_dict_list, categories=self.categories)
        self.save(ann_path=self.ann_path.replace('.json', '_bg_val.json'), images=val_im_dict_list,
                  annotations=val_ann_dict_list, categories=self.categories)

        # # replace background and save
        # num_im = len(select_im_inds)
        # for j,i in enumerate(select_im_inds):
        #     im_path = os.path.join(im_dir, self.image_dict_list[i]['file_name'])
        #     im = cv2.imread(im_path)[:,:,::-1]
        #     h, w = im.shape[:2]
        #     fg_mask = np.zeros((h,w), 'uint8')
        #     for ins in self.annotation_dict_list:
        #         if ins['image_id'] != self.image_dict_list[i]['id']: continue
        #         segm = ins['segmentation']
        #         mask = self.segmToMask(segm, h, w)
        #         fg_mask = np.bitwise_or(fg_mask, mask)
        #         # cv2.imshow('im', im)
        #         # cv2.imshow('mask', 255 * mask.astype('uint8'))
        #         # cv2.imshow('fg_mask', 255 * fg_mask.astype('uint8'))
        #         # cv2.waitKey()
        #     im = ImageProcessing().replace_background(im,fg_mask, background_im)
        #     cv2.imwrite(im_path, im[:,:,::-1])
        #     print(f'[{j}/{num_im}] {im_path} saved')
        #     # cv2.imshow('im', im[:,:,::-1])
        #     # cv2.imshow('fg_mask', 255 * fg_mask.astype('uint8'))
        #     # cv2.waitKey()


class CocoGui(DetGuiObj):

    def __init__(self, args=None, cfg_path=None, name='unnamed', default_args=None):
        DetGuiObj.__init__(self, args=args, cfg_path=cfg_path, name=name, default_args=BASE_CFGS)
        self.cocos = [] if not hasattr(self.args.path, 'im_dirs') else \
            [CocoUtils(ann_path=ann_file) for ann_file in self.args.path.ann_files]
        self.im_processor = ImageProcessing()

    def show_ann_single(self, rgbd, filename, im_dir):
        out = rgbd.rgb
        for coco, data_dir in zip(self.cocos, self.args.path.im_dirs):
            data_name = os.path.split(data_dir)[-1]
            # im_dir_name = os.path.split(im_dir)[0]
            im_dir_name = os.path.split(im_dir)[-1]
            if data_name != im_dir_name:
                continue
            if coco.filename2Id(filename) is None:
                continue
            out = coco.show_ann_single(out, filename)
        return out

    def concat_anns(self):
        # concat
        concat_ims_dir = os.path.join(self.args.path.concat_dir, 'imgs_concat')
        os.makedirs(concat_ims_dir, exist_ok=True)
        concat_ann_path = os.path.join(self.args.path.concat_dir, 'concat_ann.json')

        images, annotations, categories = [], [], []
        im_id, ann_id = 0, 0
        for coco, data_dir in zip(self.cocos, self.args.path.im_dirs):
            data_dir_name = os.path.split(data_dir)[-1]
            # concat images
            for image_dict_ in coco.image_dict_list:
                filename = f'{data_dir_name}_{image_dict_["file_name"]}'
                # copyfile(src=os.path.join(data_dir, 'imgs', image_dict_['file_name']),
                #          dst=os.path.join(concat_ims_dir, filename))
                image_dict = deepcopy(image_dict_)
                im_id += 1
                image_dict['id'] = im_id
                image_dict['file_name'] = filename
                images.append(image_dict)

                for instance_dict_ in coco.annotation_dict_list:
                    if instance_dict_['image_id'] != image_dict_['id']:
                        continue
                    instance_dict = deepcopy(instance_dict_)
                    instance_dict['image_id'] = im_id
                    ann_id += 1
                    instance_dict['id'] = ann_id
                    annotations.append(instance_dict)

        categories = self.cocos[0].categories

        # save
        coco_concat = CocoUtils()
        coco_concat.save(ann_path=concat_ann_path, images=images, annotations=annotations,
                         categories=categories)
        print(f'{concat_ann_path} saved...')

        concat_coco = CocoUtils(ann_path=concat_ann_path)
        concat_coco.show_ims(im_dir=concat_ims_dir)

    def init_acc(self):

        self.aug_params = self.im_processor.make_augmentation_params(self.args.color.brightnesss,
                                                                     self.args.color.contrasts,
                                                                     self.args.color.saturations,
                                                                     self.args.color.hues,
                                                                     range(0, 360, self.args.rotate.aug_angle_step),
                                                                     [-1, 0, 1]
                                                                     )
        print(f'1-->{len(self.aug_params[0])} augmentation params initialized ...')

        self.save_dirs = [os.path.join(aug_dir, 'imgs_aug') for aug_dir in self.args.save.aug_dirs]
        for save_dir in self.save_dirs:
            os.makedirs(save_dir, exist_ok=True)
        self.num_aug_dir = len(self.save_dirs)

        self.aug_images, self.aug_anns = [], []
        self.im_id, self.ann_id = 0, 0

        # # mask  to coco
        # self.m2c_images, self.m2c_anns = [], []
        # self.m2c_im_id, self.m2c_ann_id =0 ,0
        # classes =  self.args.mask2coco.classes if hasattr(self.args.mask2coco, 'classes') else ['unknown',]
        # self.m2c_categories = CocoUtils().make_ann_categories(classes=classes)

    def finalize_acc(self):
        if len(self.aug_images) > 0:
            self.__delattr__('aug_params')
            print('Augmentation params removed ...')
            aug_ann_path = os.path.join(self.args.save.aug_dirs[0], 'ann_aug.json')
            CocoUtils().save(aug_ann_path, images=self.aug_images, annotations=self.aug_anns,
                             categories=self.cocos[0].categories)
            print(f'{aug_ann_path} saved ...')

            aug_coco = CocoUtils(ann_path=aug_ann_path)
            aug_coco.split_trainval()
            aug_coco.show_ims(im_dir=self.save_dirs[0])

        # if len(self.m2c_images)>0:
        #     m2c_ann_path = os.path.join(self.m2c_data_dir, 'ann.json')
        #     CocoUtils().save(m2c_ann_path, images=self.m2c_images, annotations=self.m2c_anns,
        #                      categories=self.m2c_categories)
        #     print(f'{m2c_ann_path} saved ...')
        #     m2c_coco = CocoUtils(ann_path=m2c_ann_path)
        #     m2c_coco.show_ims(im_dir=self.m2c_rgb_dir)

    def augmentation_and_save(self, rgb, filename='unnamed.png', sub_thread=None):
        count = 0
        # self.init_acc()
        num_aug = len(self.aug_params[0])
        for brightness, contrast, saturation, hue, angle, ax in \
                zip(self.aug_params[0], self.aug_params[1], self.aug_params[2],
                    self.aug_params[3], self.aug_params[4], self.aug_params[5]):
            color_params_ = {'brightness': brightness,
                             'contrast': contrast,
                             'saturation': saturation,
                             'hue': hue}

            for color_params in [None, color_params_]:
                rotate_params = {'angle': angle, 'reshape': self.args.rotate.reshape, 'order': self.args.rotate.order}
                flip_params = {'axis': ax}

                count += 1
                # print(f'[{count}/{total_im}]')
                print(f'{"+" * 30} CPU_{sub_thread}: [{count}/{num_aug}] {filename}')
                print(f'color_params: {color_params}')
                print(f'rotate_params: {rotate_params}')
                print(f'flip_params: {flip_params}')

                save_dir = self.save_dirs[0] if self.num_aug_dir == 1 or sub_thread is None \
                    else self.save_dirs[sub_thread % self.num_aug_dir]

                # augmenting image and save
                out = self.im_processor.augmentation(rgb, color_params=color_params,
                                                     rotate_params=rotate_params,
                                                     flip_params=flip_params)
                im_height, im_width = out.shape[:2]
                im_height_org, im_width_org = rgb.shape[:2]
                filepath = os.path.join(save_dir, ProcUtils().get_current_time_str() + self.args.save.extend)
                cv2.imwrite(filepath, out[:, :, ::-1])
                self.im_id += 1
                im_id = self.im_id  # for threading
                CocoUtils().make_ann_images(im_id, filepath, (im_width, im_height), self.aug_images)
                print(f'{filepath} saved ...')

                # augmenting annotation
                for coco in self.cocos:
                    instances = coco.filename2Instances(filename)
                    if instances is None: continue

                    for instance in instances:
                        mask, bbox = CocoUtils().segmToMask(instance['segmentation'],
                                                            h=im_height_org, w=im_width_org), instance['bbox']
                        mask_rot = mask.copy() if angle == 0 else rotate(mask, angle=angle,
                                                                         reshape=self.args.rotate.reshape,
                                                                         order=0)
                        # out1 = rgb.copy()
                        # x, y, w, h = bbox
                        # cv2.rectangle(out1, (x, y), (x + w, y + h), self.args.disp.text_color,
                        #               self.args.disp.text_thick)
                        # cv2.imshow('out1', out1)

                        if ax == 0:
                            mask_flip = mask_rot[::-1, :]
                        elif ax == 1:
                            mask_flip = mask_rot[:, ::-1]
                        else:
                            mask_flip = mask_rot.copy()

                        Y, X = np.where(mask_flip > 0)
                        if len(Y) == 0: continue
                        xmin, ymin, xmax, ymax = np.amin(X), np.amin(Y), np.amax(X), np.amax(Y)
                        w, h = xmax - xmin, ymax - ymin
                        bbox = [xmin, ymin, w, h]
                        self.ann_id += 1
                        ann_id = self.ann_id  # for threading
                        # if ax in [0, 1] and angle%60==0:
                        #     print(f'im_id: {self.im_id}, ann_ind: {self.ann_id}')
                        #     out1 = out.copy()
                        #     out1[(Y,X)] = 0.7*out1[(Y,X)] + (0,75,0)
                        #     cv2.rectangle(out1, (xmin, ymin), (xmax, ymax), self.args.disp.text_color,
                        #                   self.args.disp.text_thick)
                        #     cv2.imshow('out1', out1[:,:,::-1])
                        #     cv2.imshow('mask', (255 * mask).astype('uint8'))
                        #     cv2.imshow('mask_rot', (255 * mask_rot).astype('uint8'))
                        #     cv2.imshow('mask_flip', (255*mask_flip).astype('uint8'))
                        #     cv2.waitKey()

                        CocoUtils().make_annotations(mask_flip, bbox, ann_id, im_id, instance['category_id'],
                                                     annotations=self.aug_anns)
                        # out2 = out.copy()
                        # x, y, w, h = bbox
                        # cv2.rectangle(out2, (x, y), (x + w, y + h), self.args.disp.text_color,
                        #               self.args.disp.text_thick)
                        # cv2.imshow('out2', out2)
                        # cv2.waitKey()

    def mask_to_coco(self, rgbd, filename, im_dir):
        rgb_crop = rgbd.crop_rgb()
        im_height, im_width = rgb_crop.shape[:2]
        left, top, right, bottom = rgbd.workspace.bbox

        self.m2c_data_dir = os.path.split(im_dir)[0]
        mask_dir = os.path.join(self.m2c_data_dir, self.args.mask2coco.mask_dir)
        self.m2c_rgb_dir = os.path.join(self.m2c_data_dir, 'imgs')
        os.makedirs(self.m2c_rgb_dir, exist_ok=True)

        maskfile = os.path.join(mask_dir,
                                filename.replace(self.args.mask2coco.rgb_extend, self.args.mask2coco.mask_extend))
        if not os.path.exists(maskfile):
            print(f'{maskfile} does not exist ...')
            return

        mask_crop = cv2.imread(maskfile, cv2.IMREAD_UNCHANGED)[top:bottom, left:right]
        # cv2.imshow('mask', mask)
        # cv2.waitKey()

        cv2.imwrite(os.path.join(self.m2c_rgb_dir, filename), rgb_crop[:, :, ::-1])
        self.m2c_im_id += 1
        im_id = self.m2c_im_id
        CocoUtils().make_ann_images(im_id, filename, (im_width, im_height), self.m2c_images)

        for val in np.unique(mask_crop):
            if val == 0: continue
            mm = mask_crop == val
            Y, X = np.where(mm)
            xmin, ymin, xmax, ymax = np.amin(X), np.amin(Y), np.amax(X), np.amax(Y)
            w, h = xmax - xmin, ymax - ymin

            cls_id = int(val / 1000)
            self.m2c_ann_id += 1
            ann_id = self.m2c_ann_id
            CocoUtils().make_annotations(mm.astype('uint8'), [xmin, ymin, w, h], ann_id, im_id, cls_id,
                                         annotations=self.m2c_anns)

    def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', **kwargs):
        if method_ind == 0:
            out = self.show_ann_single(rgbd=rgbd, filename=filename, im_dir=kwargs['im_dir'])
            ret = {'im': out}
        # if method_ind == 1:
        #     ret = self.concat_anns()
        if method_ind == 2:
            ret = self.augmentation_and_save(rgbd.rgb, filename=filename, sub_thread=kwargs['sub_thread'])
        if method_ind == 3:
            ret = self.mask_to_coco(rgbd, filename=filename, im_dir=kwargs['im_dir'])

        return ret


def run_coco_gui(cfg_path=None, default_cfg_path=None,
                 data_dir=None, rgb_formats=None, depth_formats=None,
                 key_args=[]):
    from ketisdk.gui.gui import GUI, GuiModule
    key_args += ['path.ann_files', 'path.im_dirs', 'save.aug_dirs', 'rotate.aug_angle_step',
                 'color.brightnesss', 'color.contrasts', 'color.saturations', 'color.hues', 'flip.axis']

    coco_module = GuiModule(CocoGui, name='coco_dataset', cfg_path=cfg_path, num_method=4,
                            default_cfg_path=default_cfg_path, key_args=key_args)
    CocoGui(cfg_path='configs/circle_cross_rect.cfg', default_args=BASE_CFGS)
    GUI(title='Coco Dataset Manipoluotion', modules=[coco_module, ], data_root=data_dir,
        rgb_formats=rgb_formats, depth_formats=depth_formats)


def run_coco_augmentation(root_dir, im_dir='imgs', ann_file='ann.json'):
    # # make augmentation data
    im_dir = os.path.join(root_dir, im_dir)
    im_aug_dir = im_dir + '_aug'
    ann_file = os.path.join(root_dir, ann_file)
    ann_aug_file = ann_file.replace('.json', '_aug.json')
    ann_aug_train_file = ann_aug_file.replace('.json', '_train.json')
    ann_aug_val_file = ann_aug_file.replace('.json', '_val.json')

    if not os.path.exists(im_dir):
        print(f'{im_dir} does not exist ...')
        return
    if not os.path.exists(ann_file):
        print(f'{ann_file} does not exist ...')
        return
    os.makedirs(im_aug_dir, exist_ok=True)

    # show
    coco = CocoUtils(ann_path=ann_file)
    # coco.show_instances(im_dir=os.path.join(root_dir, 'imgs'))
    coco.show_ims(im_dir=im_dir)

    # augmentation
    coco.augmentation(im_dir=im_dir, ann_path=ann_aug_file, angle_step=1)

    # split data
    coco = CocoUtils(ann_path=ann_aug_file)
    coco.split_trainval()

    # show train data
    coco = CocoUtils(ann_path=ann_aug_train_file)
    coco.show_ims(im_dir=im_aug_dir)

    # show val data
    coco = CocoUtils(ann_path=ann_aug_val_file)
    coco.show_ims(im_dir=im_aug_dir)

def concat_cocos(ann_files, im_dirs, out_file):
    out_dir = os.path.split(out_file)[0]
    os.makedirs(os.path.join(out_dir, 'imgs'), exist_ok=True)
    for im_dir in im_dirs:
        for im_path in glob.glob(os.path.join(im_dir, '*')):
            filename = os.path.split(im_path)[-1]
            os.system(f'ln -s {im_path} {os.path.join(out_dir, "imgs", filename)}')

    cocos = [CocoUtils(f) for f in ann_files]
    images, annotations = [], []
    start_im_id, start_ann_id = 0,0
    for j, coco in enumerate(cocos):
        if start_im_id!=0 or start_ann_id!=0:
            for i in range(len(coco.image_dict_list)):
                coco.image_dict_list[i]['id'] += start_im_id

            for i in range(len(coco.annotation_dict_list)):
                coco.annotation_dict_list[i]['id'] += start_ann_id
                coco.annotation_dict_list[i]['image_id'] += start_im_id

        images += coco.image_dict_list
        annotations += coco.annotation_dict_list

        # start_ann_id += len(coco.image_dict_list)
        # start_ann_id += len(coco.annotation_dict_list)
        if j!=len(cocos)-1:
            start_im_id += int(np.amax([d['id'] for d in coco.image_dict_list]))
            start_ann_id += int(np.amax([d['id'] for d in coco.annotation_dict_list]))

    coco_concat = CocoUtils()
    coco_concat.save(ann_path=out_file, images=images, annotations=annotations,
                     categories=cocos[0].categories)

    coco_concat = CocoUtils(out_file)
    coco_concat.split_trainval()
    coco_concat.show_ims(os.path.join(out_dir, "imgs"))


    print(f'{out_file} saved...')

def replace_background(ann_file, im_dir, background_path,div=[2,1], select_stride=None):
    coco = CocoUtils(ann_file)
    background = cv2.imread(background_path)[:,:,::-1]
    select_inds = np.arange(0, len(coco.image_dict_list),select_stride) if select_stride is not None else None
    coco.replace_background(im_dir,background, select_im_inds=select_inds,div=div)

def change_category_id(ann_file, out_file, old_ids, new_ids):
    coco = CocoUtils(ann_file)
    coco.categories[0]['id']=new_ids[0]
    for ann in coco.annotation_dict_list:
        ann['category_id'] = new_ids[0]

    coco.save(out_file)


if __name__ == '__main__':
    # CocoGui()

    run_coco_gui(
        # cfg_path='data/coco/annotations/person_keypoints_val2017.json',
        data_dir=BASE_CFGS.path.im_dirs[0], rgb_formats=['*', ], depth_formats='')
