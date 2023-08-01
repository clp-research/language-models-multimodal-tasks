from lavis.models import load_model_and_preprocess
import torch
from PIL import Image

class LavisCaptioningModel:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Loading LAVIS model...')
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="large_coco", is_eval=True,
                                                             device=self.device)
        print('Done.')

    def generate_caption(self, image_path):
        raw_image = Image.open(image_path).convert("RGB")
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        caption = self.model.generate({"image": image})
        return caption[0]