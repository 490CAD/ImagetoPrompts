# ImagetoPrompts

`ImagetoPrompts` is used to [Stable Diffusion - Image to Prompts](https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/overview), and now is 237/1231. Public Score 0.58145, Private Score 0.57859.

## Project Structure
    |___data_output
        |_____output.csv
    |___temp_test
    |___test_models
        |____test_clip_interrogator.py
        |____test_coca.py
        |____test_ofa.py
        |____test_vit_gpt2.py
        |____test.ipynb
    |___cal_cv_model.py
    |___main.ipynbk
    |___README.md

- data_output save the output.
- temp_test just for test, useless.
- test_models use different pretrain models to generate the prompts.
- cal_cv_model.py is find best weighted to combine the feature.
- main.ipynb is the upload code.

## Project Core

- Generate the images from prompts by using stable diffusion. 
- Train our ViT model through the dataset.
- Fine-tuning pretrain BLIP CLIP OFA model through the dataset.
- Using data augementation methods at predict time.
- Using a feature combination method to enhance the score. (Weighted accumulation ✅, MLP[TODO])


## Reference

- [OFA](https://github.com/OFA-Sys/OFA)

- [CoCa](https://github.com/UKPLab/sentence-transformers)

- [pharmapsychotic/interrogator](https://github.com/pharmapsychotic/clip-interrogator)

- And other public notes & codes from kaggle, click [this](https://www.kaggle.com/competitionsstable-diffusion-image-to-prompts) to find more.

- ```
    @misc{stable-diffusion-image-to-prompts,
        author = {Ashley Chow, inversion, Will Cukierski},
        title = {Stable Diffusion - Image to Prompts},
        publisher = {Kaggle},
        year = {2023},
        url = {https://kaggle.com/competitions/stable-diffusion-image-to-prompts}
    }
    ```
