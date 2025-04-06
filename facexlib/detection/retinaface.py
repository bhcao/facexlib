import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter as IntermediateLayerGetter

from facexlib.detection.align_trans import get_reference_facial_points, warp_and_crop_face
from facexlib.detection.retinaface_net import FPN, SSH, MobileNetV1, make_bbox_head, make_class_head, make_landmark_head
from facexlib.detection.retinaface_utils import (PriorBox, batched_decode, batched_decode_landm, py_cpu_nms)
from facexlib.utils.image_dto import ImageDTO


def generate_config(network_name):

    cfg_mnet = {
        'name': 'mobilenet0.25',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': True,
        'batch_size': 32,
        'ngpu': 1,
        'epoch': 250,
        'decay1': 190,
        'decay2': 220,
        'image_size': 640,
        'return_layers': {
            'stage1': 1,
            'stage2': 2,
            'stage3': 3
        },
        'in_channel': 32,
        'out_channel': 64
    }

    cfg_re50 = {
        'name': 'Resnet50',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': True,
        'batch_size': 24,
        'ngpu': 4,
        'epoch': 100,
        'decay1': 70,
        'decay2': 90,
        'image_size': 840,
        'return_layers': {
            'layer2': 1,
            'layer3': 2,
            'layer4': 3
        },
        'in_channel': 256,
        'out_channel': 256
    }

    if network_name == 'mobile0.25':
        return cfg_mnet
    elif network_name == 'resnet50':
        return cfg_re50
    else:
        raise NotImplementedError(f'network_name={network_name}')


class RetinaFace(nn.Module):

    def __init__(self, network_name='resnet50', half=False, phase='test', device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        super(RetinaFace, self).__init__()
        self.half_inference = half
        cfg = generate_config(network_name)
        self.backbone = cfg['name']

        self.model_name = f'retinaface_{network_name}'
        self.cfg = cfg
        self.phase = phase
        self.target_size, self.max_size = 1600, 2150
        self.mean_tensor = (104., 117., 123.)
        self.reference = get_reference_facial_points(default_square=True)
        # Build network.
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            self.body = IntermediateLayerGetter(backbone, cfg['return_layers'])
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=False)
            self.body = IntermediateLayerGetter(backbone, cfg['return_layers'])

        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]

        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

        self.to(self.device)
        self.eval()
        if self.half_inference:
            self.half()

    def forward(self, inputs):
        out = self.body(inputs)

        if self.backbone == 'mobilenet0.25' or self.backbone == 'Resnet50':
            out = list(out.values())
        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        tmp = [self.LandmarkHead[i](feature) for i, feature in enumerate(features)]
        ldm_regressions = (torch.cat(tmp, dim=1))

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output

    def detect_faces(
        self,
        images,
        conf_threshold=0.8,
        nms_threshold=0.4,
        use_origin_size=True,
        pad_to_same_size=False,
    ):
        """
        Args:
            images: a list of PIL.Image, np.array, or image file paths.
            conf_threshold: confidence threshold.
            nms_threshold: nms threshold.
            use_origin_size: whether to use origin size.
        
        Returns:
            list of np.array ([n_boxes, 15], type=np.float32) in (xyxy, conf, landmarks) format.
        """
        # prepare input
        only_one_img = False
        if not isinstance(images, (list, tuple)):
            images = [images]
            only_one_img = True
        
        if use_origin_size:
            images = [ImageDTO(img) for img in images]
        else:
            images = [ImageDTO(img).resize(
                self.target_size, keep_ratio=True, align_shorter_edge=True, max_size=self.max_size
            ) for img in images]
        
        same_shape = len({x.image.shape for x in images}) == 1
        if not same_shape:
            if pad_to_same_size:
                max_shape = np.max(np.array([x.image.shape for x in images]), axis=0)
                images = [x.pad(max_shape) for x in images]
            else:
                raise ValueError('Images have different shapes, set pad_to_same_size=True.')
        
        image_tensors = torch.cat([image.to_tensor(
            rgb2bgr=True,
            mean=self.mean_tensor,
            std=1.0
        ) for image in images]).to(
            device=self.device,
            dtype=torch.half if self.half_inference else torch.float32,
        )

        # inference
        b_loc, b_conf, b_landmarks = self.forward(image_tensors)

        # get priorbox
        priorbox = PriorBox(self.cfg, image_size=image_tensors.shape[2:])
        priors = priorbox.forward().to(self.device).unsqueeze(0)

        # post-process
        boxes = batched_decode(b_loc, priors, self.cfg['variance'])
        boxes = torch.stack([images[i].restore_keypoints(box, uniformed=True) for i, box in enumerate(boxes)])

        scores = b_conf[:, :, 1]

        landmarks = batched_decode_landm(b_landmarks, priors, self.cfg['variance'])
        landmarks = torch.stack([images[i].restore_keypoints(landmark, uniformed=True) for i, landmark in enumerate(landmarks)])

        # index for selection
        b_indice = scores > conf_threshold

        # concat
        b_preds = torch.cat((boxes, scores.unsqueeze(-1)), dim=2).float()

        final_results = []

        for pred, landm, inds in zip(b_preds, landmarks, b_indice):

            # ignore low scores
            pred, landm = pred[inds, :], landm[inds, :]

            if pred.shape[0] == 0:
                final_results.append(np.array([], dtype=np.float32))
                continue

            # sort
            # order = score.argsort(descending=True)
            # box, landm, score = box[order], landm[order], score[order]

            # to CPU
            bounding_boxes, landm = pred.cpu().numpy(), landm.cpu().numpy()

            # NMS
            keep = py_cpu_nms(bounding_boxes, nms_threshold)
            bounding_boxes, landmarks = bounding_boxes[keep, :], landm[keep]

            # append
            final_results.append(np.concatenate((bounding_boxes, landmarks), axis=1))
        # self.t['forward_pass'].toc(average=True)
        # self.batch_time += self.t['forward_pass'].diff
        # self.total_frame += len(frames)
        # print(self.batch_time / self.total_frame)

        if only_one_img:
            final_results = final_results[0]

        return final_results


    def __align_multi(self, image, boxes, landmarks, limit=None):

        if len(boxes) < 1:
            return [], []

        if limit:
            boxes = boxes[:limit]
            landmarks = landmarks[:limit]

        faces = []
        for landmark in landmarks:
            facial5points = [[landmark[2 * j], landmark[2 * j + 1]] for j in range(5)]

            warped_face = warp_and_crop_face(np.array(image), facial5points, self.reference, crop_size=(112, 112))
            faces.append(warped_face)

        return np.concatenate((boxes, landmarks), axis=1), faces

    def align_multi(self, img, conf_threshold=0.8, limit=None):

        rlt = self.detect_faces(img, conf_threshold=conf_threshold)
        boxes, landmarks = rlt[:, 0:5], rlt[:, 5:]

        return self.__align_multi(img, boxes, landmarks, limit)

    def batched_detect_faces(self, frames, conf_threshold=0.8, nms_threshold=0.4, use_origin_size=True):
        warnings.warn('This function is deprecated, use detect_faces instead.', DeprecationWarning)

        results = self.detect_faces(frames, conf_threshold=conf_threshold, nms_threshold=nms_threshold, 
                                    use_origin_size=use_origin_size)
        
        final_bounding_boxes = [result[:, 0:5] for result in results]
        final_landmarks = [result[:, 5:] for result in results]

        return final_bounding_boxes, final_landmarks
