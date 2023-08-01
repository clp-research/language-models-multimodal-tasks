# Images in Language Space: Exploring the Suitability of Large Language Models for Vision & Language Tasks

## Publication

This is the official Git repository page for the paper:

> Hakimov, S., and Schlangen, D., (2023).  Images in Language Space: Exploring the Suitability of Large Language Models for Vision & Language Tasks. Findings of the Association for Computational Linguistics: ACL 2023 [PDF](https://aclanthology.org/2023.findings-acl.894.pdf)


## Code

### Installation

Create a virtual environment in Python 3 and run the following scripts
```
pip install -r requirements.txt

pip install git+https://github.com/openai/CLIP.git
```

To use OFA Captioning model, please check the instructions from their [official Git repository](https://github.com/OFA-Sys/OFA).


### Extraction of Image as Text Representation

Run the following code snippet that extracts visual tags, captions from BLIP and VITGPT2 models for images in the 'images' directory
```
python extract_image_as_text_representation.py --dir images
```

### Prompting Large Language Models

The prompt templates for each dataset are given in the 'resource/prompt_templates' directory.



## Citation
If you find the resources or the code useful, please cite us:
```
@inproceedings{hakimov-schlangen-2023-images,
    title = "Images in Language Space: Exploring the Suitability of Large Language Models for Vision {\&} Language Tasks",
    author = "Hakimov, Sherzod  and
      Schlangen, David",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.894",
    pages = "14196--14210",
    abstract = "Large language models have demonstrated robust performance on various language tasks using zero-shot or few-shot learning paradigms. While being actively researched, multimodal models that can additionally handle images as input have yet to catch up in size and generality with language-only models. In this work, we ask whether language-only models can be utilised for tasks that require visual input {--} but also, as we argue, often require a strong reasoning component. Similar to some recent related work, we make visual information accessible to the language model using separate verbalisation models. Specifically, we investigate the performance of open-source, open-access language models against GPT-3 on five vision-language tasks when given textually-encoded visual information. Our results suggest that language models are effective for solving vision-language tasks even with limited samples. This approach also enhances the interpretability of a model{'}s output by providing a means of tracing the output back through the verbalised image content.",
}
```