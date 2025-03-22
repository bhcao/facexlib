'''
Simplified version of basicsr.models.sr_model and basicsr.models.base_model without training,
save, distribution and metric logic.
'''

from copy import deepcopy
from collections import OrderedDict
import numpy as np
import torch
from PIL import Image

from facexlib.utils.misc import get_root_logger, img2tensor, tensor2img

class SRModel:
    """Base SR model for single image super-resolution."""

    def __init__(self, net_g, device='auto', load_path=None, param_key='params', strict_load=True, half=False):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.dtype = torch.float16 if half else torch.float32

        # define network
        self.net_g = net_g.to(self.device, dtype=self.dtype)
        self.print_network(self.net_g)

        # load pretrained models
        if load_path is not None:
            self.load_network(self.net_g, load_path, strict_load, param_key)

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


    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print keys with different name or different size when loading models.

        1. Print keys with different names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        logger = get_root_logger()
        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

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
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.
        
        # HWC to CHW, numpy to tensor
        self.lq = img2tensor(image, bgr2rgb=False, float32=True).to(
            device=self.device, dtype=self.dtype
        ).unsqueeze(0)
        self.test()

        visuals = self.get_current_visuals()

        # tentative for out of GPU memory
        del self.lq
        del self.output
        torch.cuda.empty_cache()
    
        return tensor2img([visuals['result']])


    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        return out_dict
