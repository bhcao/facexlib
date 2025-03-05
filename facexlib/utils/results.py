from facexlib.utils.data_tensor import BaseTensor, data_tensor


@data_tensor
class BBox(BaseTensor):
    r'''
    Initialize a BBox object.

    Args:
        x: x-coordinate of the top-left corner of the bounding box.
        y: y-coordinate of the top-left corner of the bounding box.
        w: width of the bounding box.
        h: height of the bounding box.
        confidence: Confidence score of the detection.
    '''
    x: float
    y: float
    w: float
    h: float
    confidence: float


@data_tensor
class Landmarks(BaseTensor):
    r'''
    Initialize a Landmarks object.
    
    Args:
        left_eye: The coordinates (x, y) of the left eye with respect to the person instead of observer.
        right_eye: The coordinates (x, y) of the right eye.
        nose: The coordinates (x, y) of the nose.
        mouth_left: The coordinates (x, y) of the left mouth corner.
        mouth_right: The coordinates (x, y) of the right mouth corner.
    '''
    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    nose: tuple[float, float]
    mouth_left: tuple[float, float]
    mouth_right: tuple[float, float]


@data_tensor
class FacialAreaRegion(BaseTensor):
    r"""
    Initialize a Face object. Contains a bounding box and facial landmarks.

    Args:
        bbox: Bounding box.
        landmarks: Facial landmarks.
    """
    bbox: BBox
    landmarks: Landmarks


@data_tensor
class GenderAge(BaseTensor):
    r"""
    Initialize a GenderAge object. Contains gender and age.

    Args:
        gender: Gender of the person.
        age: Age of the person.
    """
    gender: tuple[float, float]
    age: float

    def get_gender(self):
        if self.gender[0] > self.gender[1]:
            return 'male'
        return 'female'


@data_tensor(element_type=float, element_num=512)
class FaceEmbedding(BaseTensor):
    r"""
    Initialize a FaceEmbedding object. Contains a 512-dimensional face embedding.
    """
    pass