'''
Simplified version of basicsr.models.sr_model and basicsr.models.base_model without training,
save, distribution and metric logic.
'''

from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from facexlib.utils.image_dto import ImageDTO
from facexlib.utils.misc import get_root_logger

class SRModel(nn.Module):
    """Base SR model for single image super-resolution."""

    def __init__(self, net_g, device='cuda', half=False):
        super(SRModel, self).__init__()
        self.device = torch.device(device)
        self.dtype = torch.float16 if half else torch.float32

        # define network
        self.net_g = net_g.to(self.device, dtype=self.dtype)
        self.print_network(self.net_g)


    def print_network(self, net):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        net_cls_str = f'{net.__class__.__name__}'

        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        logger = get_root_logger()
        logger.info(f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        logger.info(net_str)


    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()
    
    def inference(self, image: np.ndarray | Image.Image):
        self.lq = ImageDTO(image).to_tensor().to(device=self.device, dtype=self.dtype)
        self.test()

        visuals = self.get_current_visuals()

        # tentative for out of GPU memory
        del self.lq
        del self.output
        torch.cuda.empty_cache()
    
        return ImageDTO(visuals['result'], min_max=(0, 1))


    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        return out_dict
