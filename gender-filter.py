from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
import tqdm
import os
import glob

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'


def get_args():
    parser = argparse.ArgumentParser(description="Script to filter images by gender",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--input-folder", help="Path to folder file")
    parser.add_argument("--output-folder", help="Folder to save output visualizations")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

def main():
    args = get_args()
    depth = args.depth
    k = args.width
    weight_file = args.weight_file
    margin = args.margin

    if not weight_file:
        weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                               file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

    folder_images = os.path.expanduser(args.input_folder)
    images = []
    images = glob.glob(os.path.join(folder_images, "*.png"))
    images.extend(glob.glob(os.path.join(folder_images, "*.jpg")))
    images.extend(glob.glob(os.path.join(folder_images, "*.jpeg")))

    # for face detection
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    img_size = 64
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)

    for path in tqdm.tqdm(images):
        img_name = os.path.splitext(os.path.basename(path))[0] # without file suffix
        img = cv2.imread(path)
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        segment_img = cv2.imread(os.path.join(args.output_folder, "segmented/", img_name + ".png"))

        # if not img_name == "0377":
        #     continue

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        # predict ages and genders of the detected faces
        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

            results = model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

            # process results
            # remove all female detected segments
            for i, d in enumerate(detected):
                gender = "M" if predicted_genders[i][0] < 0.5 else "F"
                label = "{}, {}".format(int(predicted_ages[i]), gender)
                print(path, label)
                draw_label(img, (d.left(), d.top()), label)
                if gender == "F":
                    centerX = int(d.left() + d.width()/2)
                    centerY = int(d.top() + d.height()/2)
                    segment_color = segment_img[centerY, centerX]
                    segment_pixels_indices = np.where(np.all(segment_img == segment_color, axis=-1))
                    segment_img[segment_pixels_indices] = [0,0,0]

        # make all remaining segements white
        non_black_pixels_indices = np.any(segment_img != [0, 0, 0], axis=-1)
        segment_img[non_black_pixels_indices] = [255,255,255]

        # save new binary mask
        out_filename = os.path.join(args.output_folder, "age_gender_masked_binary/", img_name + ".png")
        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
        cv2.imwrite(out_filename, segment_img)

        # save debug image
        out_filename = os.path.join(args.output_folder, "age_gender/", img_name + ".png")
        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
        cv2.imwrite(out_filename, img)

if __name__ == '__main__':
    main()
