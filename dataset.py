import torch.utils.data as data
import torch

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
from pathlib import Path

from itertools import cycle


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(
            len([x for x in Path(
                self._data[0]).glob('img_*')])-1)

    @property
    def label(self):
        return int(self._data[1])


class I3DDataSet(data.Dataset):
    def __init__(self, data_root, split=1, sample_frames=64, 
            modality='RGB', transform=lambda x:x,
            train_mode=True, test_clips=10,
            sample_frames_at_test=False):

        self.data_root = data_root
        self.split = split
        self.sample_frames = sample_frames
        self.modality = modality
        self.transform = transform
        self.train_mode = train_mode
        self.sample_frames_at_test = sample_frames_at_test
        if not self.train_mode:
            self.num_clips = test_clips

        self._parse_split_files()


    def _load_image(self, directory, idx):
        if self.modality == 'RGB':
            img_path = os.path.join(directory, 'img_{:05}.jpg'.format(idx))
            try:
                img = Image.open(img_path).convert('RGB')
            except:
                print("Couldn't load image:{}".format(img_path))
                return None
            return [img]
        else:
            try:
                img_path = os.path.join(directory, 'flow_x_{:05}.jpg'.format(idx))
                x_img = Image.open(img_path).convert('L')
            except:
                print("Couldn't load image:{}".format(img_path))
                return None
            try:
                img_path = os.path.join(directory, 'flow_y_{:05}.jpg'.format(idx))
                y_img = Image.open(img_path).convert('L')
            except:
                print("Couldn't load image:{}".format(img_path))
                return None
            x_img = np.array(x_img, dtype=np.float32)
            y_img = np.array(y_img, dtype=np.float32)
            img = np.asarray([x_img, y_img]).transpose([1, 2, 0])
            img = Image.fromarray(img.astype('uint8'))
            return [img]


    def _parse_split_files(self):
            file_list = sorted(Path('./data/hmdb51_splits').glob('*'+str(self.split)+'.txt'))
            video_list = []
            for class_idx, f in enumerate(file_list):
                class_name = str(f).strip().split('/')[2][:-16]
                class_count = 0
                for line in open(f):
                    tokens = line.strip().split(' ')
                    video_path = self.data_root+class_name+'/'+tokens[0][:-4]
                    record = (video_path, class_idx)
                    if self.train_mode & (tokens[-1] == '1'):
                        video_list.append(VideoRecord(record))
                        class_count += 1
                    elif (self.train_mode == False) & (tokens[-1] == '2'):
                        video_list.append(VideoRecord(record))
                        class_count += 1
                
                #print("class: ", class_name, " count: ", class_count, " label: ", class_idx)
            self.video_list = video_list


    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        if record.num_frames > self.sample_frames:
            start_pos = randint(record.num_frames - self.sample_frames + 1)
            offsets = range(start_pos, start_pos + self.sample_frames, 1)
        else:
            offsets = [x for x in range(record.num_frames)]
        offsets = [int(v)+1 for v in offsets]  # images are 1-indexed
        if len(offsets) < self.sample_frames:
            self._loop_indices(offsets)
        return offsets


    def _get_test_indices(self, record):
        offsets = [v+1 for v in range(record.num_frames)]
        return offsets


    def _loop_indices(self, indices):
        indices_cycle = cycle(indices)
        while len(indices) < self.sample_frames:
            indices.append(next(indices_cycle))


    def __getitem__(self, index):
        record = self.video_list[index]
        if self.train_mode or self.sample_frames_at_test:
            segment_indices = self._sample_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        process_data, label = self.get(record, segment_indices)
        if process_data is None:
            raise ValueError('sample indices:', record.path, segment_indices)
        
        return process_data, label


    def get(self, record, indices):

        images = list()
        for ind in indices:
            seg_img = self._load_image(record.path, ind)
            if seg_img is None:
                return None,None
            images.extend(seg_img)

        process_data = self.transform(images)
        return process_data, record.label


    def __len__(self):
        return len(self.video_list)


if __name__ == '__main__':
    dat = I3DDataSet(
        data_root='/datadir/rawframes/',
        split=1,
        sample_frames = 64,
        modality='RGB',
        train_mode=False,
        sample_frames_at_test=False
    )
    item = dat.__getitem__(10)
    print(item[1])
    print(len(item[0]))
    print(item[0][0].size)

    for x in dat:
        print(len(x[0]))
        pass