from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

from videotransforms import (
    Stack, GroupNormalize, ToTorchFormatTensor, GroupScale, GroupCenterCrop
)

from models.pytorch_i3d import InceptionI3d

from dataset import I3DDataSet


def get_frame_list(video_list, data_root):
    frame_list = []
    for video in video_list:
        for f in sorted(Path(data_root+video+'/').glob('img_*.jpg')):
            frame_list.append(str(f))
    return frame_list


def load_model(modality, state_dict_file):

    channels = 3 if modality == "RGB" else 2
    model = InceptionI3d(51, in_channels=channels)
    state_dict = torch.load(state_dict_file)
    model.load_state_dict(state_dict)

    model = torch.nn.DataParallel(model).cuda()

    return model


def load_image(frame_file):
    try:
        img = Image.open(frame_file).convert('RGB')
        return img
    except:
        print("Couldn't load image:{}".format(frame_file))
        return None


def construct_input(frames):

    transform = torchvision.transforms.Compose([
                    GroupScale(256),
                    GroupCenterCrop(224),
                    Stack(),
                    ToTorchFormatTensor(),
                    GroupNormalize(),
                ])

    frame_list = []
    for frame in frames:
        frame_list.append(load_image(frame))
    process_data = transform(frame_list)
    return process_data.unsqueeze(0)


def predict_input(model, input):
    input = input.cuda(non_blocking=True)
    output = model(input)
    output = torch.mean(output, dim=2)
    return output


def predict_over_video(video_frame_list, window_width=9, stride=1):

    if window_width < 9:
        raise ValueError("window_width must be 9 or greater")

    print("Loading model...")

    model = load_model(
        modality="RGB",
        state_dict_file="pretrained_chkpt/rgb_hmdb_split1.pt"
    )

    model.eval()

    print("Predicting actions over {0} frames".format(len(video_frame_list)))

    with torch.no_grad():

        window_count = 0

        for i in range(stride+window_width-1, len(video_frame_list), stride):
            window_frame_list = [video_frame_list[j] for j in range(i-window_width, i)]
            batch = construct_input(window_frame_list)
            window_predictions = predict_input(model, batch)
            window_proba = F.softmax(window_predictions, dim=1)
            window_top_pred = window_proba.max(1)
            print(("Window:{0} Class pred:{1} Class proba:{2}".format(
                window_count,
                window_top_pred.indices.cpu().numpy()[0],
                window_top_pred.values.cpu().numpy()[0])
            ))
            window_count += 1



if __name__ == "__main__":

    video_list = [
        'brush_hair/Aussie_Brunette_Brushing_Long_Hair_brush_hair_u_nm_np1_ba_med_3',
        'brush_hair/Brushing_Her_Hair__[_NEW_AUDIO_]_UPDATED!!!!_brush_hair_h_cm_np1_fr_goo_0',
        'brush_hair/Brushing_my_long_hair_brush_hair_u_nm_np1_ba_goo_2',
        'cartwheel/Aerial_Cartwheel_Tutorial_By_Jujimufu_cartwheel_f_nm_np1_ri_med_0',
        'cartwheel/Beim_Radschlag_Hose_gerissen_(Nick)_xD_cartwheel_f_cm_np1_ri_med_0',
        'cartwheel/CarTwHeeL_PerFecT_cartwheel_f_cm_np1_le_med_0',
        'catch/Fangen_und_Werfen_catch_u_nm_np1_fr_bad_0',
        'catch/Florian_Fromlowitz_beim_Training_der_U_21_Nationalmannschaft_catch_f_cm_np1_ri_med_0',
        'catch/Goalkeeper_Training_Day_#_2_catch_f_cm_np1_ba_bad_3',
    ]

    frame_list = get_frame_list(video_list, '/datadir/rawframes/')

    predict_over_video(frame_list, window_width=64, stride=32)