import cv2
import numpy as np
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from .modeling.utils import reset_cls_test


def get_clip_embeddings(vocabulary, prompt='a '):
    from detic.modeling.text.text_encoder import build_text_encoder
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

BUILDIN_CLASSIFIER = {
    'lvis': 'inst_seg/datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': 'inst_seg/datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': 'inst_seg/datasets/metadata/oid_clip_a+cname.npy',
    'coco': 'inst_seg/datasets/metadata/coco_clip_a+cname.npy',
    'order': 'inst_seg/datasets/metadata/order_picking_clip_a+cname.npy'
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}

class Predictor(object):
    def __init__(self, cfg, args,
                 instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        if args.vocabulary == 'custom':
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = args.custom_vocabulary.split(',')
            classifier = get_clip_embeddings(self.metadata.thing_classes)
            # class 단어 임베딩 된 데이터 저장 for Orin
            """
            classifier_npy = classifier.t().numpy()
            classifier_npy = classifier_npy.astype(np.float16)
            np.save('inst_seg/datasets/metadata/order_picking_clip_a+cname.npy', classifier_npy)
            print('save npy')
            """
        else:
            if args.vocabulary == 'order':
                self.metadata = MetadataCatalog.get("order")
                self.metadata.thing_classes = args.custom_vocabulary.split(',')
            else:
                self.metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[args.vocabulary])
            classifier = BUILDIN_CLASSIFIER[args.vocabulary]

        num_classes = len(self.metadata.thing_classes)
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.predictor = DefaultPredictor(cfg)
        reset_cls_test(self.predictor.model, classifier, num_classes)

        self.imsize = args.input_size
        self.acceptable_size_min = args.acceptable_size_min
        self.acceptable_size_max = args.acceptable_size_max

    def remove_small_and_large(self, masks: np.ndarray) -> np.ndarray:
        """
        image shape: [h, w, ch]             e.g) [960, 1280, 3]
        masks shape: [instance num, h, w]   e.g) [num, 480, 640]
        """

        num_instances = masks.shape[0]
        h, w = masks.shape[1:]

        mmin = h * w * self.acceptable_size_min
        mmax = h * w * self.acceptable_size_max

        add_mask = [True if mmin <= np.count_nonzero(masks[i, :, :]) <= mmax else False for i in range(num_instances)]

        masks = np.array([masks[i, :, :] for i in range(num_instances) if add_mask[i]])
        return masks

    def remove_small_in_bigger(self, masks: np.ndarray, overlap_ratio:float) -> np.ndarray:
        remove_list = []

        for i1 in range(masks.shape[0]):
            if i1 not in remove_list:

                mask1 = masks[i1, :, :]
                mask1_size = np.count_nonzero(mask1)

                for i2 in range(masks.shape[0]):
                    if i1 != i2:
                        mask2 = masks[i2, :, :]
                        mask2_size = np.count_nonzero(mask2)
                        overlap_size = np.count_nonzero(np.logical_and(mask1, mask2))

                        # mask1과 mask2의 겹치는 영역이 존재하고, mask1이 mask2보다 크다면,
                        if overlap_size and mask1_size > mask2_size:
                            # 두 마스크의 겹치는 영역이 마스크1 크기 * 비율 보다 작으면
                            if overlap_size < mask1_size * overlap_ratio and overlap_size > mask2_size * (1 - overlap_ratio):
                                # mask2를 제거할 물체 목록에 추가
                                remove_list.append(i2)
                            # 한 객체를 여러개로 인식했을 경우 (두 객체의 영역 겹침이 95% 이상일 때)
                            if overlap_size > mask1_size * 0.8 or overlap_size > mask2_size * 0.8:
                                remove_list.append(i2)

        filtered_masks = np.delete(masks, list(set(remove_list)), axis=0)

        return filtered_masks

    def __call__(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        # BGR -> RGB
        # image = image[:, :, ::-1]

        image = cv2.resize(image, dsize=self.imsize)

        predictions = self.predictor(image)

        predictions["instances"] = predictions["instances"].to(self.cpu_device)

        filtered_masks = None

        if predictions['instances'].pred_masks.shape[0]:
            try:
                filtered_masks = self.remove_small_and_large(np.array(predictions['instances'].pred_masks))
                f1 = filtered_masks.shape[0]
                # filtered_masks = self.remove_small_in_bigger(filtered_masks, 0.3)
                # f2 = filtered_masks.shape[0]
                # print(f"Total Objects: {predictions['instances'].pred_masks.shape[0]}, "
                #       f"1st Filter: {f1}, "
                #       f"2nd Filter: {f2} ")
                if not filtered_masks.shape[0]:
                    filtered_masks = None
            except:
                pass

        # predictions["instances"] = predictions["instances"].to(self.cpu_device)
        # masks = np.array(predictions['instances'].pred_masks)
        # plt.imshow(masks[0, :, :])
        # plt.show()

        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)

        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, filtered_masks, vis_output

COUNT = 0
class VisualizationDemo(object):
    def __init__(self, cfg, args, 
        instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        global COUNT
        COUNT += 1
        if args.vocabulary == 'custom':
            self.metadata = MetadataCatalog.get(f"__{COUNT}")
            self.metadata.thing_classes = args.custom_vocabulary.split(',')
            classifier = get_clip_embeddings(self.metadata.thing_classes)

            # classifier_npy = classifier.t().numpy()
            # classifier_npy = classifier_npy.astype(np.float16)
            # np.save('inst_seg/datasets/metadata/order_picking_clip_a+cname.npy', classifier_npy)

        else:
            if args.vocabulary == 'order':
                self.metadata = MetadataCatalog.get("order")
                self.metadata.thing_classes = args.custom_vocabulary.split(',')
            else:
                self.metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[args.vocabulary])
            classifier = BUILDIN_CLASSIFIER[args.vocabulary]

        num_classes = len(self.metadata.thing_classes)
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor = DefaultPredictor(cfg)
        reset_cls_test(self.predictor.model, classifier, num_classes)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """

        predictions = self.predictor(image)

        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)

        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output
