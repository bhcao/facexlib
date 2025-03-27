import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import cv2

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResBlock, self).__init__()

        self.downsample = downsample

        if downsample:
            self.skip_layers = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2 if self.downsample else 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU()


    def forward(self, x):
        if self.downsample:
            residual = self.skip_layers(x)
        else:
            residual = x
        
        x = self.layers(x)
        x = x + residual
        x = self.relu(x)
        return x


class OutputBlock(nn.Module):
    def __init__(self):
        super(OutputBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(56, 80, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(80, 80, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(80, 80, 3, padding=1),
            nn.ReLU(),
        )

        self.conv_score = nn.Conv2d(80, 2, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

        self.conv_bbox = nn.Conv2d(80, 8, kernel_size=3, padding=1)
        self.scale = nn.Parameter(torch.tensor(0.0))

        self.conv_landmark = nn.Conv2d(80, 20, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.layers(x)

        score = self.sigmoid(self.conv_score(x).permute(2, 3, 0, 1).reshape(-1, 1))
        bbox = (self.conv_bbox(x) * self.scale).permute(2, 3, 0, 1).reshape(-1, 4)
        landmark = self.conv_landmark(x).permute(2, 3, 0, 1).reshape(-1, 10)

        return score, bbox, landmark


class InsightRetina(nn.Module):
    '''InsightFace RetinaFace'''
    def __init__(self):
        super(InsightRetina, self).__init__()

        # backbone
        self.input_block = nn.Sequential(
            nn.Conv2d(3, 28, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(28, 28, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(28, 56, 3, padding=1),
            nn.ReLU(),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layers_1 = nn.Sequential(
            ResBlock(56, 56),
            ResBlock(56, 56),
            ResBlock(56, 56),
            ResBlock(56, 88, downsample=True),
            ResBlock(88, 88),
            ResBlock(88, 88),
            ResBlock(88, 88),
        )
        self.conv_1 = nn.Conv2d(88, 56, kernel_size=1)

        self.layers_2 = nn.Sequential(
            ResBlock(88, 88, downsample=True),
            ResBlock(88, 88),
        )
        self.conv_2 = nn.Conv2d(88, 56, kernel_size=1)

        self.layers_3 = nn.Sequential(
            ResBlock(88, 224, downsample=True),
            ResBlock(224, 224),
            ResBlock(224, 224)
        )
        self.conv_3 = nn.Conv2d(224, 56, kernel_size=1)

        # fpn
        self.conv_fpn_1 = nn.Conv2d(56, 56, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_fpn_2 = nn.Conv2d(56, 56, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_fpn_3 = nn.Conv2d(56, 56, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_fpn_4 = nn.Conv2d(56, 56, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv_fpn_5 = nn.Conv2d(56, 56, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv_fpn_6 = nn.Conv2d(56, 56, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_fpn_7 = nn.Conv2d(56, 56, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # output block
        self.output_block_1 = OutputBlock()
        self.output_block_2 = OutputBlock()
        self.output_block_3 = OutputBlock()

        # insightface variables
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.input_size = (640, 640)
        self.input_mean = 127.5
        self.input_std = 128.0
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True


    def forward(self, x):
        # backbone
        x = self.input_block(x)
        x = self.max_pool(x)

        x = self.layers_1(x)
        feature_1 = self.conv_1(x)
        x = self.layers_2(x)
        feature_2 = self.conv_2(x)
        x = self.layers_3(x)
        feature_3 = self.conv_3(x)

        # fpn
        feature_23 = feature_2 + F.interpolate(feature_3, size=feature_2.shape[2:4])
        feature_out_1 = self.conv_fpn_1(feature_1 + F.interpolate(feature_23, size=feature_1.shape[2:4]))
        feature_123 = self.conv_fpn_2(feature_23) + self.conv_fpn_4(feature_out_1)
        feature_out = self.conv_fpn_3(feature_3) + self.conv_fpn_5(feature_123)
        feature_out_2 = self.conv_fpn_6(feature_123)
        feature_out_3 = self.conv_fpn_7(feature_out)
        
        # output block
        score_1, bbox_1, landmark_1 = self.output_block_1(feature_out_1)
        score_2, bbox_2, landmark_2 = self.output_block_2(feature_out_2)
        score_3, bbox_3, landmark_3 = self.output_block_3(feature_out_3)

        return (score_1, score_2, score_3, bbox_1, bbox_2, bbox_3, landmark_1, landmark_2, landmark_3)

    # below are copied from insightface
    def prepare(self, nms_thresh=None, input_size=None):
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        if input_size is not None:
            self.input_size = input_size

    # insightface forward
    def detect(self, img, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0/self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True)

        blob = torch.from_numpy(blob).to(next(self.parameters()).device, next(self.parameters()).dtype)
        net_outs = self.forward(blob)

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx].cpu().numpy()
            bbox_preds = net_outs[idx+fmc].cpu().numpy()
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = net_outs[idx+fmc*2].cpu().numpy() * stride
            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)

                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                if self._num_anchors>1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
                if len(self.center_cache)<100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores>=threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                #kpss = kps_preds
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    # insightface detect
    def detect_faces(self, img, threshold=0.5, max_num=0, metric='default'):
        
        input_size = self.input_size
            
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.detect(det_img, threshold)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order,:,:]
            kpss = kpss[keep,:,:]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                    det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric=='max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        
        # to make it consistent with another retinaface
        return np.concatenate((det, kpss.reshape(-1, 10)), axis=1)

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

