import os
import numpy as np
from tifffile import imread, imsave
from napari_convpaint import conv_paint_utils, conv_paint, conv_paint_param
from catboost import CatBoostClassifier


def main():
    # Gather tif files from all folders in the current directory
    all_lifeact = []
    for folder in ["blebbistatin", "latrunculinb", "dmso", "ycompound"]:
        if os.path.isdir(folder):
            for file in os.listdir(folder):
                if file.endswith(".tif"):
                    all_lifeact.append(os.path.join(folder, file))

    # Determine which file to process
    image_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    input_filename = all_lifeact[image_index]
    current_condition = input_filename.split("/")[0]
    output_filename = os.path.join("seg", current_condition, os.path.basename(input_filename))

    # Read the image
    lifeact = imread(input_filename)
    all_seg = np.zeros_like(lifeact, dtype=np.uint8)

    # Set up parameter objects for feature extraction with convpaint
    param_dino = conv_paint_param.Param(
        fe_name="dinov2_vits14_reg",
        fe_scalings=[1],
        fe_order=0,
        image_downsample=1,
        fe_use_cuda=False,
    )
    param_vgg = conv_paint_param.Param(
        fe_name="vgg16",
        fe_scalings=[1],
        fe_order=0,
        fe_layers=[
            "features.0 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
            "features.2 Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
            "features.5 Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
            "features.7 Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
            "features.10 Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
            "features.12 Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
            "features.14 Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
        ],
        image_downsample=1,
        fe_use_cuda=False,
    )

    print("Creating feature extractor models...")
    model_dino = conv_paint.create_model(param_dino)
    model_vgg = conv_paint.create_model(param_vgg)

    # Load the CatBoost model
    print("Loading CatBoost model...")
    model = CatBoostClassifier()
    model.load_model("convpaint_model")

    # Process the image (slice by slice if 3D)
    if lifeact.ndim == 3:
        for i in range(lifeact.shape[0]):
            mean, std = conv_paint_utils.compute_image_stats(lifeact[0], ignore_n_first_dims=0)
            img_norm = conv_paint_utils.normalize_image(lifeact[i], mean, std)

            features_dino, _ = conv_paint.get_features_current_layers(
                img_norm, np.ones_like(img_norm), model_dino, param_dino
            )
            features_vgg, _ = conv_paint.get_features_current_layers(
                img_norm, np.ones_like(img_norm), model_vgg, param_vgg
            )
            features = np.concatenate((features_dino, features_vgg), axis=1)

            predicted = model.predict(features)
            all_seg[i] = predicted.reshape(img_norm.shape)
            print(f"Processed slice {i + 1}/{lifeact.shape[0]}")
    else:
        mean, std = conv_paint_utils.compute_image_stats(lifeact, ignore_n_first_dims=0)
        img_norm = conv_paint_utils.normalize_image(lifeact, mean, std)

        features_dino, _ = conv_paint.get_features_current_layers(
            img_norm, np.ones_like(img_norm), model_dino, param_dino
        )
        features_vgg, _ = conv_paint.get_features_current_layers(
            img_norm, np.ones_like(img_norm), model_vgg, param_vgg
        )
        features = np.concatenate((features_dino, features_vgg), axis=1)

        predicted = model.predict(features)
        all_seg = predicted.reshape(img_norm.shape)
        print("Processed 2D image")

    # Save the result
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    imsave(output_filename, all_seg.astype(np.uint8), compression="zlib")
    print(f"Segmentation saved to {output_filename}")


if __name__ == "__main__":
    main()
