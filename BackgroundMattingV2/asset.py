from pathlib import Path
from typing import Literal
import gdown


def get_weights_path(model_backbone: Literal['resnet101', 'resnet50', 'mobilenetv2']='resnet50'):
    CACHE_DIR = Path.home() / '.cache' / 'BackgroundMattingV2'
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True)

    model_links = {
        'resnet101': 'https://drive.google.com/uc?id=1zysR-jW6jydA2zkWfevxD1JpQHglKG1_',
        'resnet50': 'https://drive.google.com/uc?id=1ErIAsB_miVhYL9GDlYUmfbqlV293mSYf',
        'mobilenetv2': 'https://drive.google.com/uc?id=1b2FQH0yULaiBwe4ORUvSxXpdWLipjLsI',
    }

    # could be downloaded in this repo: https://github.com/italojs/facial-landmarks-recognition/tree/master
    weights_path = CACHE_DIR / f'pytorch_{model_backbone}.pth'
    weights_url = model_links[model_backbone]
    if not weights_path.exists():
        print(f"Downloading from {weights_url} to {weights_path}...")
        gdown.download(weights_url, str(weights_path))

    return weights_path