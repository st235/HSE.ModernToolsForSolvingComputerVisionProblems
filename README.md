# Original model

[The original model](https://developers.google.com/mediapipe/solutions/vision/image_segmenter#selfie-model) performs selfie
segmentation.

The model is an exported **TensorFlow Lite** model. The features of the model are:
- **Input**: the model expects **normalised** 256x256 RGB image.
- **Recognises 6 classes**: background, hair, body-skin, face-skin, clothes, and accessories.
- **Output**: a tensor of probabilities of size _256x256x6_. 

You can get an idea of the segmentation masks for every class from the images below:

| Original                                | Background                                  | Hair                            |
|-----------------------------------------|---------------------------------------------|---------------------------------|
| ![Original](./docs/model_original.jpeg) | ![Background](./docs/model_background.jpeg) | ![Hair](./docs/model_hair.jpeg) |

| Body Skin                                 | Face Skin                                 | Clothes                               | Accessories                                   |
|-------------------------------------------|-------------------------------------------|---------------------------------------|-----------------------------------------------|
| ![Body Skin](./docs/model_body_skin.jpeg) | ![Face Skin](./docs/model_face_skin.jpeg) | ![Clothes](./docs/model_clothes.jpeg) | ![Accessories](./docs/model_accessories.jpeg) |

is stored under [`original_model` folder](./original_model).

# Working with TVM

The model has been converted to TVM IR, then tuned, compiled for `x86_64-linux-gnu` and exported.
`x86_64-linux-gnu` is a target for the linux image that runs `python:3.9-slim-bullseye` Docker container.

The entire history of tuning is listed in [`TFLite_Converation.ipynb` notebook file](./tools/TFLite_Convertation.ipynb),
that has been used at the preparation step. 

## Tuning

The key parameters for tuning are given below:

| Parameter      | Value | Notes                                                                                                         |
|----------------|-------|---------------------------------------------------------------------------------------------------------------|
| tuner          | xgb   | XGBoost works really good and much faster than GridSearch.                                                    |
| early_stopping | False | Instead of stopping earlier I limit trials.                                                                   |
| n_trial        | 333   | Empirically tuned value: not big to finish within a few hours, not small to actually find optimal parameters. |

Tuning helped and optimised the model for the specific hardware. I was able to spare about **350 ms** on average.
You can find the final optimisation statistic in [`tvm_model/tuning_statistics.log`](./tools/tuning_statistics.log) and
the evaluation summary from **Collab** below:

![Tuning results](./docs/tuning_results.png)

## Exporting the artifacts

I considered multiple options before exporting the model: 
1. Using TVMC as a standalone archive
2. Using build-in TVM methods to get a dynamic library

### TVMC Package

TVMC packs model in a really handy archive. Though there is a wee issue with TVMC: 
the parameters are not baked with the model, and you need to provide them separately. 
It seemed it was necessary to serialise them and keep separately. So to work with a model with weights one needs to
provide 2 files per model: model archive and the parameters.

I tried to export files and work with them (you can find the code in the notebook linked above).
Exported model lays under [`tvm_model` folder](./tvm_model) and is called `tvm_not_tuned_selfie_multiclass.tar`.
The file with parameters can be found under the same folder and is called [`tvmc_serialised_parameters.npy`](./tvm_model/tvmc_serialised_parameters.npy).


### TVM built-in methods

The pair [`lib.export_library`](https://tvm.apache.org/docs/reference/api/python/runtime.html?highlight=export_library#tvm.runtime.Module.export_library) -
[`tvm.runtime.load_module`](https://tvm.apache.org/docs/reference/api/python/runtime.html?highlight=runtime#tvm.runtime.load_module) can export and then load the model
as a dynamic library specifically compiled for the hardware.

I preferred this way over **TVMC** as it allows to bake-in the parameters of the model and provide them within the same
file. This approach is used in the final solution.

# Bot

## How to use this bot 

The bot is really easy to control and interact with. It supports 3 commands:
1. `/segmentation`
2. `/change-hair-color`
3. `/change-background`

When the command is executed it returns an image or series of images and performance information.
Performance compares **original** and **tuned** model and is similar to the example below:

```text
Inference of default model took 264 ms.
Inference of tuned model took 185 ms.
```

Let's take a closer look at these commands.

### Segmentation

![Bot segmentation](docs/bot_segmentation.png)

In the segmentation mode you need to send a selfie in a **quick send** mode.

The result is 7 images: 6 corresponding to the classes masks and the original image.

### Change Hair Color

![Bot change hair color](docs/bot_change_color_hair.png)

Receives a selfie and changes the color of the person's hair.

This command is a two steps command:
1. Waits for a selfie image
2. Asks to send a new color value as a hexadecimal rgb string

The result is a modified rgb image with new hair color.

### Change Background

![Bot change background](docs/bot_change_background.png)

Changes background in selfie, similar to what modern video call apps are doing.

This command also works in two steps:
1. Waits for a selfie image
2. Asks to provide a new background

The command returns a modified selfie with replace background.

## Deployment

And the last but not least, the instructions how to deploy the bot.
The bot is using `docker-compose`.

### Create an environment file with a token

First of all, it is required to create an `.env` file and declare `TELEGRAM_BOT_TOKEN` variable in there.
The content of a file should look similar to the snippet below:

```bash
TELEGRAM_BOT_TOKEN=0000000000:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
```

The real token can be obtained from the [`@botfather`](https://t.me/botfather).

#### Build docker-compose image

Run the command below to build the docker image.

`docker-compose --project-name hse_tvm_bot build`

#### Run with environment

To run the image one should use the command below.

`docker-compose --project-name hse_tvm_bot --env-file .env up`

After running the command, the bot will go online and it would be possible to use it from Telegram.
