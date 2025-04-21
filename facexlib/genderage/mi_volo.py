from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from facexlib.utils.image_dto import ImageDTO
from facexlib.utils.misc import get_root_logger
from facexlib.genderage.utils import create_model, crop_object
from timm.data import resolve_data_config

has_compile = hasattr(torch, "compile")

MIVOLO_CFG = {
    'mivolo_d1': {
        'min_age': 1,
        'max_age': 95,
        'avg_age': 48.0,
        'no_gender': False,
        'with_persons_model': True,
        'input_size': 224
    },
    'volo_d1': {
        'min_age': 21,
        'max_age': 60,
        'avg_age': 40.5,
        'no_gender': False,
        'with_persons_model': False,
        'input_size': 224
    }
}

class Meta:
    def __init__(self, cfg, disable_faces: bool = False, use_persons: bool = True):
        self.min_age = cfg['min_age']
        self.max_age = cfg['max_age']
        self.avg_age = cfg['avg_age']
        self.only_age = cfg['no_gender']
        self.with_persons_model = cfg['with_persons_model']
        self.input_size = cfg['input_size']

        self.num_classes = 1 if self.only_age else 3
        self.in_chans = 3 if not self.with_persons_model else 6

        self.num_classes_gender = 2

        self.disable_faces = disable_faces
        self.use_persons = use_persons and self.with_persons_model

        if not self.with_persons_model and self.disable_faces:
            raise ValueError("You can not use disable-faces for faces-only model")
        if self.with_persons_model and self.disable_faces and not self.use_persons:
            raise ValueError(
                "You can not disable faces and persons together. "
                "Set --with-persons if you want to run with --disable-faces"
            )

    def __str__(self):
        attrs = vars(self)
        attrs.update({"use_person_crops": self.use_person_crops, "use_face_crops": self.use_face_crops})
        return ", ".join("%s: %s" % item for item in attrs.items())

    @property
    def use_person_crops(self) -> bool:
        return self.with_persons_model and self.use_persons

    @property
    def use_face_crops(self) -> bool:
        return not self.disable_faces or not self.with_persons_model


class MiVOLO(nn.Module):
    def __init__(
        self,
        model_type: str,
        device: str = "cuda",
        half: bool = True,
        disable_faces: bool = False,
        use_persons: bool = True,
        verbose: bool = False,
        torchcompile: Optional[str] = None,
    ):
        super().__init__()
        self.verbose = verbose
        self.device = torch.device(device)
        self.fp16 = half and self.device.type != "cpu"

        self.meta: Meta = Meta(MIVOLO_CFG[model_type], disable_faces=disable_faces, use_persons=use_persons)

        if self.verbose:
            get_root_logger().info(f"Model meta:\n{str(self.meta)}")

        model_name = f"mivolo_d1_{self.meta.input_size}"
        self.model = create_model(
            model_name=model_name,
            num_classes=self.meta.num_classes,
            in_chans=self.meta.in_chans,
            pretrained=False,
        )
        self.param_count = sum([m.numel() for m in self.model.parameters()])
        get_root_logger().info(f"Model {model_name} created, param count: {self.param_count}")

        self.data_config = resolve_data_config(
            model=self.model,
            verbose=verbose,
            use_test_size=True,
        )

        self.data_config["crop_pct"] = 1.0
        c, h, w = self.data_config["input_size"]
        assert h == w, "Incorrect data_config"
        self.input_size = w

        if torchcompile:
            assert has_compile, "A version of torch w/ torch.compile() is required for --compile, possibly a nightly."
            torch._dynamo.reset()
            self.model = torch.compile(self.model, backend=torchcompile)

        if self.fp16:
            self.model = self.model.half()

    def inference(self, model_input: torch.tensor, keep_pre_logits: bool = False) -> torch.tensor:

        if self.fp16:
            model_input = model_input.half()
        
        features = self.model.forward_features(model_input)
        output = self.model.forward_head(features)
        if keep_pre_logits:
            pre_logits = self.model.forward_head(features, pre_logits=True)
            return output, pre_logits

        return output, None

    def predict(self, image: Union[np.ndarray, str, torch.Tensor], bboxes: np.ndarray, keep_pre_logits: bool = False):
        '''
        Predict age and gender for faces and persons in the image.

        Args:
            image: Input image, ImageDTO accepted type.
            bboxes: Bounding boxes of faces and persons, shape (n, 10).
            keep_pre_logits: Whether to return pre-logits or not.
        
        Returns:
            A tuple of (ages, genders, gender_scores), where ages, genders, and gender_scores are lists of length n.
            If `keep_pre_logits` is True, will also return pre-logits.
        '''
        image = ImageDTO(image)

        if len(bboxes) > 0:
            assert bboxes.shape[1] == 10, "Results should have both persons and faces"

        n_faces = np.sum(np.logical_not(np.isnan(bboxes[:, 0])))
        n_persons = np.sum(np.logical_not(np.isnan(bboxes[:, 5])))
        if (
            (len(bboxes) == 0)
            or (not self.meta.use_persons and n_faces == 0)
            or (self.meta.disable_faces and n_persons == 0)
        ):
            # nothing to process
            return None

        zero_img = ImageDTO(np.zeros((1, 1, 3), dtype=np.uint8))
        assert self.meta.use_face_crops or self.meta.use_person_crops, "Must specify at least one of use_persons and use_faces"

        # crop faces and persons
        faces_crops, person_crops = [], []
        inds = []
        for ind in range(len(bboxes)):
            face_image = crop_object(bboxes, image, ind) if self.meta.use_face_crops else zero_img
            person_image = crop_object(bboxes, image, ind, is_person=True) if self.meta.use_person_crops else zero_img

            if face_image is not None or person_image is not None:
                faces_crops.append(face_image.resize(self.input_size, keep_ratio=True).pad(self.input_size, fill=0)
                                   .to_tensor(mean=self.data_config["mean"], std=self.data_config["std"], timm_form=True))
                person_crops.append(person_image.resize(self.input_size, keep_ratio=True).pad(self.input_size, fill=0)
                                    .to_tensor(mean=self.data_config["mean"], std=self.data_config["std"], timm_form=True))
                inds.append(ind)

        person_input = torch.concat(person_crops).to(self.device)
        faces_input = torch.concat(faces_crops).to(self.device)

        if faces_input is None and person_input is None:
            # nothing to process
            return None

        if self.meta.with_persons_model:
            model_input = torch.cat((faces_input, person_input), dim=1)
        else:
            model_input = faces_input
        
        output, pre_logits = self.inference(model_input, keep_pre_logits)

        # write gender and age results
        results = self.fill_in_results(output, inds, len(bboxes))

        if keep_pre_logits:
            return *results, pre_logits
        else:
            return results


    def fill_in_results(self, output, inds, length) -> Tuple[List[int], List[str], List[float]]:
        ages = [None for _ in range(length)]
        genders = [None for _ in range(length)]
        gender_scores = [None for _ in range(length)]

        if self.meta.only_age:
            age_output = output
            gender_probs, gender_indx = None, None
        else:
            age_output = output[:, 2]
            gender_output = output[:, :2].softmax(-1)
            gender_probs, gender_indx = gender_output.topk(1)

        assert output.shape[0] == len(inds)

        # per face
        for index in range(output.shape[0]):
            ind = inds[index]

            # get_age
            age = age_output[index].item()
            age = age * (self.meta.max_age - self.meta.min_age) + self.meta.avg_age
            age = round(age, 2)

            ages[ind] = age

            if gender_probs is not None:
                gender = "male" if gender_indx[index].item() == 0 else "female"
                gender_score = gender_probs[index].item()

                genders[ind] = gender
                gender_scores[ind] = gender_score
        
        return ages, genders, gender_scores


if __name__ == "__main__":
    model = MiVOLO("../pretrained/checkpoint-377.pth.tar", half=True, device="cuda:0")