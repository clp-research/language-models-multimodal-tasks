from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
import torch

class OFACaptioningModel:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ckpt_dir = 'OFA-huge'

        print('Loading model...')
        self.tokenizer = OFATokenizer.from_pretrained(ckpt_dir, device_map='auto')
        self.model = OFAModel.from_pretrained(ckpt_dir, device_map='auto')
        self.model.to(self.device)

        txt = " what does the image show?"
        self.inputs = self.tokenizer([txt], return_tensors="pt").input_ids.to(self.device)
        print('Done')

    def generate_caption(self, image_path, txt=None):

        if txt is not None:
            self.inputs = self.tokenizer([txt], return_tensors="pt").input_ids.to(self.device)

        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 256
        patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        img = Image.open(image_path)
        patch_img = patch_resize_transform(img).unsqueeze(0).to(self.device)

        gen = self.model.generate(self.inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3)
        caption = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        return caption