import argparse
import sys
import os
from src.image_as_text_representation.VITGPT2CaptioningModel import VITGPT2CaptioningModel
from src.image_as_text_representation.VisualTagExtractor import VisualTagExtractor
from src.image_as_text_representation.LAVISModel import LavisCaptioningModel
from src import file_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    image_dir = args.dir

    image_files = file_utils.get_files_in_directory(image_dir, ['jpg'])

    lavis_captioning_model = LavisCaptioningModel()
    vitgpt2_captioning_model = VITGPT2CaptioningModel()
    visual_tag_extractor = VisualTagExtractor()

    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)

        caption_blip = lavis_captioning_model.generate_caption(image_path)
        caption_vitgpt2 = vitgpt2_captioning_model.generate_caption(image_path)
        visual_tags_as_string = visual_tag_extractor.generate_visual_tags(image_path)

        print(image_path)
        print(caption_blip, ' === ', caption_vitgpt2,  '===', visual_tags_as_string)
        print('\n')