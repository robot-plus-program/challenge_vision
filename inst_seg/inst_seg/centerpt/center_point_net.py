import timm
import torch
from torch import nn
from torchvision import transforms


class CenterPointNet:
    def __init__(self, baseline='mobilenet', pretrained=None):
        self.model = self.load_model(baseline)

        if pretrained:
            try:
                self.model.load_state_dict(torch.load(pretrained, map_location='cuda:0'),
                                           strict=False)
                print('Pretrained model load done.')
            except Exception as e:
                print('ValueError. Failed to load weight')

        self.input_size = [224, 224]

        self.model.eval()
        # print(self.model)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.input_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 데이터셋의 평균
                                 std=[0.229, 0.224, 0.225])   # ImageNet 데이터셋의 표준편차
        ])

    def __call__(self, images):
        shape_list = [i.shape[:2][::-1] for i in images]

        batch = torch.stack(list(map(lambda img: self.transform(img.copy()), images)))

        with torch.no_grad():
            output = self.model(batch.cuda())

        output = output.cpu().numpy()
        output = np.array([shape_list[i] * xy for i, xy in enumerate(output)], dtype=np.uint16)

        return output

    @staticmethod
    def load_model(baseline):
        if baseline == 'mobilenet':
            model = timm.create_model('mobilenetv3_small_100', pretrained=True)
        elif baseline == 'efficientnet':
            model = timm.create_model('efficientnet_b0', pretrained=True)
        elif baseline == 'efficientnet-light':
            model = timm.create_model('efficientnet_light0', pretrained=True)
        else:
            raise AttributeError('model name error.')

        in_feature = model.get_classifier().in_features
        # out_feature = model.get_classifier().out_features
        out_feature = 500

        model.classifier = nn.Sequential(
            nn.Linear(in_feature, out_feature),
            nn.ReLU(),
            nn.Linear(out_feature, 2)
        )

        return model.cuda()

    def get_model(self):
        return self.model


"""
class CustomMobileNetV3(nn.Module):
    def __init__(self):
        super(CustomMobileNetV3, self).__init__()

        self.baseline = timm.create_model('mobilenetv3_small_100', pretrained=True)

        in_feature = self.baseline.get_classifier().in_features
        out_feature = self.baseline.get_classifier().out_features

        self.baseline.classifier = nn.Sequential(
            nn.Linear(in_feature, out_feature),
            nn.ReLU(),
            nn.Linear(out_feature, 2)
        )

    def forward(self, x):
        return self.baseline(x)


class CustomEfficientNet(nn.Module):
    def __init__(self):
        super(CustomEfficientNet, self).__init__()

        self.baseline = timm.create_model('efficientnet_b0', pretrained=True)

        in_feature  = self.baseline.get_classifier().in_features
        out_feature = self.baseline.get_classifier().out_features

        self.baseline.classifier = nn.Sequential(
            nn.Linear(in_feature, out_feature),
            nn.ReLU(),
            nn.Linear(out_feature, 2)
        )

    def forward(self, x):
        return self.baseline(x)
"""


def open_label(path):
    try:
        with open(path, 'r') as f:
            data = f.readline()
        if data:
            data = [int(i) for i in data.split(',')]
            return data
        else:
            return False
    except:
        return False

if __name__ == "__main__":
    import os
    import cv2
    import numpy as np
    import natsort

    BASELINE = 'mobilenet'
    PRETRAINED = 'checkpoint/20240321_mobilenet_epoch124_lr0.0061.pth'
    # BASELINE = 'efficientnet'
    # PRETRAINED = 'checkpoint/20240321_efficientnet_epoch26_lr0.0200.pth'

    det = CenterPointNet(baseline=BASELINE, pretrained=PRETRAINED)

    img_path = 'dataset/valid/image'
    lbl_path = 'dataset/valid/label'
    img_list = natsort.natsorted(os.listdir(img_path))

    images = []
    labels = []
    for i in img_list:
        name = i.split('.')[0]

        images.append(np.array(cv2.imread(f"{img_path}/{name}.png")[:, :, ::-1]))
        labels.append(open_label(f"{lbl_path}/{name}.txt"))
    result = det(images)

    # a *= img1.shape[0]
    # a = a.astype(np.int64)
    # print(lbl1, a[0])
    # print(lbl2, a[1])
    print(labels)

    show = []
    for i in range(len(images)):
        show_img = cv2.circle(np.array(images[i]), labels[i], radius=3, color=(255, 0, 0), thickness=2)
        show_img = cv2.circle(show_img, result[i], radius=3, color=(0, 0, 255), thickness=2)
        show.append(show_img)

    show = np.hstack(show)

    cv2.imshow('result', show[:,:,::-1])
    cv2.waitKey(0)




