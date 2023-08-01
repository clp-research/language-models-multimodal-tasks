from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor, AutoModelForCausalLM, BlipForConditionalGeneration, VisionEncoderDecoderModel, BlipProcessor
import torch
from PIL import Image
from src import file_utils


class ImageCaptioningModel:
    def __init__(self):
        print('Loading image captioning models ...')

        if not file_utils.dir_exists('huggingface_cache'):
            file_utils.create_dirs('huggingface_cache')

        # self.git_processor_large_coco = AutoProcessor.from_pretrained("microsoft/git-large-coco", cache_dir='huggingface_cache')
        # self.git_model_large_coco = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco", cache_dir='huggingface_cache')
        #
        # self.git_processor_large_textcaps = AutoProcessor.from_pretrained("microsoft/git-large-r-textcaps", cache_dir='huggingface_cache')
        # self.git_model_large_textcaps = AutoModelForCausalLM.from_pretrained("microsoft/git-large-r-textcaps", cache_dir='huggingface_cache')

        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir="huggingface_cache")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir="huggingface_cache").to("cuda")

        # self.blip_processor_large = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir='huggingface_cache')
        # self.blip_model_large = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir='huggingface_cache')

        # self.vitgpt_processor = AutoImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir='huggingface_cache')
        # self.vitgpt_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir='huggingface_cache')
        # self.vitgpt_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir='huggingface_cache')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.git_model_large_coco.to(self.device)
        # self.git_model_large_textcaps.to(self.device)
        # self.blip_model_large.to(self.device)
        # self.vitgpt_model.to(self.device)

        print('Done.')

    def generate_caption(self, processor, model, image, tokenizer=None):

        inputs = processor(images=image, return_tensors="pt").to(self.device)

        generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)

        if tokenizer is not None:
            generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_caption


    def generate_captions(self, image_path):

        print('Extracting captions')

        image = Image.open(image_path)

        inputs = self.blip_processor(image, return_tensors="pt").to("cuda")

        output = self.blip_model.generate(**inputs)
        blip_caption = self.blip_processor.decode(output[0], skip_special_tokens=True)
        return blip_caption

        # try:
        #     caption_git_large_coco = self.generate_caption(self.git_processor_large_coco, self.git_model_large_coco, image)
        # except:
        #     caption_git_large_coco = ''
        #
        # try:
        #     caption_git_large_textcaps = self.generate_caption(self.git_processor_large_textcaps, self.git_model_large_textcaps, image)
        # except:
        #     caption_git_large_textcaps = ''

        # try:
        # caption_blip_large = self.generate_caption(self.blip_processor_large, self.blip_model_large, image)
        # # except:
        # #     caption_blip_large = ''
        #
        # try:
        #     caption_vitgpt = self.generate_caption(self.vitgpt_processor, self.vitgpt_model, image, self.vitgpt_tokenizer)
        # except:
        #     caption_vitgpt = ''
        #
        # print('Done.')
        #
        # # return caption_git_large_coco, caption_git_large_textcaps, caption_blip_large, caption_vitgpt
        #
        # return caption_blip_large, caption_vitgpt
