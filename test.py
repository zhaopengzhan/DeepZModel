import warnings

import torch

warnings.filterwarnings("ignore")
import logging
from ptflops import get_model_complexity_info

logging.basicConfig(
    level=logging.INFO,  # 控制最低输出等级
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]  # 输出到控制台
)

import models


def test_list_models():
    model_names = models.list_models()
    print(model_names)
    pass


def test_build_all_models(in_channels=4, image_size=224, num_classes=17):
    model_names = models.list_models()
    for model_name in model_names:
        model = models.build_model(model_name=model_name,
                                   in_channels=in_channels,
                                   num_classes=num_classes)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        macs, params = get_model_complexity_info(model, (in_channels, image_size, image_size),
                                                 as_strings=True, print_per_layer_stat=False,
                                                 verbose=False)
        print(f'model_name: {model_name}')
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        print()
    pass


if __name__ == '__main__':
    # test_list_models()
    test_build_all_models()
