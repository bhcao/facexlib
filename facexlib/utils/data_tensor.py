from typing import Union
import functools
import numpy as np
import torch

# Modified from ultralytics.engine.results.BaseTensor
class BaseTensor:
    """
    Base tensor class with additional methods for easy manipulation and device handling.

    Attributes:
        data (torch.Tensor | np.ndarray): Prediction data such as bounding boxes, masks, or landmarks.

    Methods:
        cpu: Return a copy of the tensor stored in CPU memory.
        numpy: Returns a copy of the tensor as a numpy array.
        cuda: Moves the tensor to GPU memory, returning a new instance if necessary.
        to: Return a copy of the tensor with the specified device and dtype.
        type_as: Return a copy of the tensor with the same device and dtype as the other tensor.
    """

    element_num: int = 0
    element_type: type = None
    data: torch.Tensor | np.ndarray = None

    def __init__(self, data: Union[torch.Tensor, np.ndarray, 'BaseTensor'] = None, **kwargs) -> None:
        '''Create a new tensor object.
        
        Args:
            data (torch.Tensor | np.ndarray, optional): Initializes the tensor with the given data.
            **kwargs: Initializes the tensor with the given attributes.
        '''
        if data is None:
            self.data = np.zeros((self.element_num), dtype=self.element_type)
            for k, v in kwargs.items():
                setattr(self, k, v)
            return
        
        if isinstance(data, BaseTensor):
            data = data.data

        def matches(dtype, element_type):
            if dtype == element_type:
                return True
            if dtype in (torch.float16, torch.float32, torch.float64,
                         np.float16, np.float32, np.float64) and element_type == float:
                return True
            if dtype in (torch.int8, torch.int16, torch.int32, torch.int64, 
                         np.int8, np.int16, np.int32, np.int64) and element_type == int:
                return True
            if dtype in (torch.bool, np.bool_) and element_type == bool:
                return True
            return False

        # Check data type and shape
        assert isinstance(data, (torch.Tensor, np.ndarray)), "'data' must be 'torch.Tensor' or 'np.ndarray'"
        assert data.shape[-1] == self.element_num, f"Invalid data shape: {data.shape}, expected {self.element_num} elements"
        assert matches(data.dtype, self.element_type), f"Invalid data type: {data.dtype}, expected {self.element_type.__name__}"
        self.data = data

    @property
    def shape(self):
        return self.data.shape

    def cpu(self):
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu())

    def numpy(self):
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy())

    def cuda(self):
        return self.__class__(torch.as_tensor(self.data).cuda())

    def to(self, *args, **kwargs):
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs))
    
    def type_as(self, other: 'BaseTensor'):
        if isinstance(other.data, np.ndarray):
            return self.numpy()
        return self.to(device=other.data.device, dtype=other.data.dtype)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.__class__(self.data[idx])
    
    def __setitem__(self, idx, value):
        if isinstance(value, BaseTensor):
            value = value.type_as(self).data
        self.data[idx] = value
    
    # avoid add new attributes after initialization
    def __setattr__(self, name, value):
        if name in self.__dir__():
            super().__setattr__(name, value)
        else:
            raise AttributeError(
                f"Cannot create new attribute '{name}' to an instance of '{type(self).__name__}' object"
            )


def concat(tensors: list[BaseTensor]) -> BaseTensor:
    """
    Concatenate a list of tensors along the batch size dimension.

    Args:
        tensors (list[BaseTensor]): A list of tensors to be concatenated.

    Returns:
        (BaseTensor): The concatenated tensor.
    """
    if tensors is None or len(tensors) == 0:
        return None
    if len(tensors) == 1:
        return tensors[0]
        
    # Check tensors type
    tensor_types = set([type(i) for i in tensors])
    assert len(tensor_types) == 1, f"All tensors must be the same type, but got {tensor_types.pop().__name__} and {tensor_types.pop().__name__}"
    tensor_type = tensor_types.pop()
    assert issubclass(tensor_type, BaseTensor), f"All tensors must be subclasses of 'BaseTensor', but got {tensor_type.__name__}"
        
    data = []
    # Unsqueeze tensors without batch dimension
    for t in tensors:
        if len(t.data.shape) == 1:
            data.append(t.data.reshape(1, -1))
        else:
            data.append(t.data)

    if isinstance(tensors[0].data, np.ndarray):
        data = np.concatenate(data, axis=0)
    else:
        data = torch.cat(data, dim=0)
    return tensors[0].__class__(data)


def data_tensor(cls: type = None, element_type: type = None, element_num: int = 0) -> type:
    """
    Decorator that converts class attributes into aliases for fields in the `data` array,
    allowing attribute-style access to the elements of the `data` array. 

    Args:
        cls (type): The class to be decorated, which should have attributes with same type.
        element_type (type, optional): The type of the elements in the `data` array. If sepecified,
                                       types of all fields must be consistent with this type.
        element_num (int, optional): The number of elements in the `data` array. If fields are provided,
                                     this argument will be ignored.

    Returns:
        (type): The decorated class with tensor-based data handling.
    
    Example:
        >>> @data_tensor
        >>> class MyTensor(BaseTensor):
        >>>     attr1: int
        >>>     attr2: tuple[int, int]

        >>> @data_tensor(element_type=float, element_num=512)
        >>> class MyTensor2(BaseTensor):
        >>>     pass

        >>> dt = MyTensor(data=np.array([1, 2, 3]))
        >>> dt.attr1 = 4
        >>> print(dt.data)
        [4 2 3]
        >>> print(dt.attr2)
        (2, 3)
    """

    # For decorator with arguments, return a new decorator function
    if cls is None:
        return functools.partial(data_tensor, element_type=element_type, element_num=element_num)

    assert issubclass(cls, BaseTensor), f"Class '{cls.__name__}' must be a subclass of 'BaseTensor'"
    annotations = cls.__annotations__
    assert annotations.get('data', None) is None, f"Cannot use 'data' as an attribute name of type '{cls.__name__}'"
    cls.__annotations__ = {}

    # Ignore element_num if annotations are provided
    if len(annotations) > 0:
        element_num = 0

    for name, dtype in annotations.items():
        # Get all element types
        if dtype in (int, float, bool):
            element_types = (dtype,)
            current_element_num = 1
        elif hasattr(dtype, '__args__'):
            # tuple or list
            element_types = dtype.__args__
            current_element_num = len(element_types)
        elif issubclass(dtype, BaseTensor):
            element_types = (dtype.element_type,)
            current_element_num = dtype.element_num
        else:
            raise ValueError(f"Unsupported data type: {dtype.__name__}")
        element_types += (element_type,) if element_type is not None else ()
        
        # Check if all elements have the same type
        element_type_set = set(element_types)
        if len(element_type_set) > 1:
            raise ValueError(f"Elements of '{name}' must have the same type, but got {element_type_set.pop().__name__} and {element_type_set.pop().__name__}")
        element_type = element_type_set.pop()
        assert element_type in (int, float, bool), f"Unsupported data type: {element_type.__name__}"

        if current_element_num == 1:
            # Getter method, pass element_num to avoid closure
            def getter(self, dtype=dtype, element_num=element_num):
                return dtype(self.data[..., element_num])
            
            # Setter method
            def setter(self, value, element_num=element_num):
                if isinstance(value, BaseTensor):
                    value = value.type_as(self).data
                self.data[..., element_num] = value
        
        else:
            # Getter method
            def getter(self, dtype=dtype, element_num=element_num, current_element_num=current_element_num):
                if hasattr(dtype, '__args__'):
                    return dtype([dtype.__args__[i](self.data[..., element_num+i]) for i in range(current_element_num)])
                return dtype(self.data[..., element_num:element_num+current_element_num])
            
            # Setter method
            def setter(self, value, element_num=element_num, current_element_num=current_element_num):
                if isinstance(value, BaseTensor):
                    value = value.type_as(self).data
                self.data[..., element_num:element_num+current_element_num] = value
        
        # Add getter and setter to the class
        setattr(cls, name, property(getter, setter))
        element_num += current_element_num

    # used for determing the shape and type of nested tensors
    cls.element_num = element_num
    cls.element_type = element_type

    return cls
