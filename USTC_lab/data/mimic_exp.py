"""
exp class for imitation learning
"""
import numpy as np
import os, time
import torch
from multiprocessing import Queue
from typing import List, Tuple, Generator, Union
from torch.utils.data import Dataset, DataLoader

"""
    ################### Writer ##########################
"""
# TODO np.save delay by f.flush(), I don't know why, use try except .


class MimicExpWriter:
    """
        all process share this instance.


        if y: label is tensor, like classical or mujoco, save label as {}_{}_{}_y.npy
        if y: label is a number, do not save .

        x: inputs, maybe pictures, shape like: [4(batch),4(frame),84(weight),84(high)],
        x: inputs, maybe tensors, shape like: [4(batch), 16(observation * frame)], for convenience, we reshape 16 to [4, 4]

    """
    def __init__(self,
                 task_name: str,
                 save_file_dir: str,
                 process_num: int = 1,
                 frame_stack: int = 1):
        self.task_name = task_name
        self.save_file_dir = save_file_dir
        if not os.path.exists(self.save_file_dir):
            try:
                os.makedirs(self.save_file_dir)
            except:
                pass
        self.sf = open(save_file_dir + "dataset.txt", "w")
        self.sf.write(save_file_dir + "\n")
        self.q = Queue()

        self.index = 0

        self.pnum = process_num
        # self. = [0] * process_num
        self.step_ = [0] * self.pnum
        self.frame_stack = frame_stack

    def put(self, *args) -> None:
        self.q.put(args)

    def deal_inputs(self, inputs: np.ndarray, p_index: int = 0) -> np.ndarray:
        return inputs

    def deal_labels(self, labels: np.ndarray, p_index: int = 0) -> Union[List[str], np.ndarray]:
        return labels

    def save_inputs(self, inputs: np.ndarray, p_index: int = 0) -> Union[List[str], np.ndarray]:
        out = []
        i = 0
        # loop batch
        for input in inputs:
            frame_flag = ["{}_{}_{}.npy".format(p_index, i, self.step_[p_index] + j) for j in range(self.frame_stack)]
            # save first frame
            np.save(self.save_file_dir + frame_flag[0], input[0])
            out.append(",".join(frame_flag))
            i += 1
        return out

    def save_labels(self, labels: np.ndarray, p_index: int = 0) -> List[str]:
        i = 0
        out = []
        # loop batch
        for label in labels:
            label_flag = "{}_{}_{}_y.npy".format(p_index, i, self.step_[p_index])
            np.save(self.save_file_dir + label_flag, label)
            out.append(label_flag)
            i += 1
        return out

    def deal_raw(self, data: Tuple[np.ndarray, np.ndarray], p_index:int = 0) -> Generator:
        """
                    save pic frame as {}_{}_{}.npy : 0_3_12.npy
                    first number is process index
                    second number is batch index
                    third number is step index
        """

        x_, y_ = data
        x_ = self.deal_inputs(x_, p_index)
        y_ = self.deal_labels(y_, p_index)
        self.step_[p_index] += 1
        for x, y in zip(x_, y_):
            yield x + "||" + str(y)

    def write(self) -> None:
        size = self.q.qsize()
        for i in range(size):
            x = self.q.get()
            data, p_index = x[:2], x[-1]
            lines_gen = self.deal_raw(data, p_index)
            for line in lines_gen:
                self.sf.write(line + '\n')
                # self.sf.flush()


class MimicExpTensorFrameWriter(MimicExpWriter):
    def __init__(self, *args):
        super(MimicExpTensorFrameWriter, self).__init__(*args)

    def deal_inputs(self, x_: np.ndarray, p_index: int = 0) -> np.ndarray:
        # x_.shape: [batch_num, observation_space * frame_stack] -> [batch_num, observation_space, frame_stack]
        x_ = x_.reshape(x_.shape[0], self.frame_stack, x_.shape[1]//self.frame_stack)
        out = self.save_inputs(x_, p_index)
        return out

    def deal_labels(self, labels: np.ndarray, p_index: int = 0) -> List:
        """because the action is tensor, not simple number like atari,
           we need to save labels as .npy"""
        out = self.save_labels(labels, p_index)
        return out


class MimicExpPicFrameWriter(MimicExpWriter):
    def __init__(self, *args):
        super(MimicExpPicFrameWriter, self).__init__(*args)

    def deal_inputs(self, x_: np.ndarray, p_index: int = 0) -> np.ndarray:
        """x: inputs, maybe pictures, shape like: [4(batch),4(frame),84(weight),84(high)]"""
        out = self.save_inputs(x_, p_index)
        return out


class MimicExpClassicalWriter(MimicExpTensorFrameWriter):
    def __init__(self, *args):
        super(MimicExpClassicalWriter, self).__init__(*args)


class MimicExpMujocoWriter(MimicExpTensorFrameWriter):
    def __init__(self, *args):
        super(MimicExpMujocoWriter, self).__init__(*args)


class MimicExpAtariWriter(MimicExpPicFrameWriter):
    def __init__(self, *args):
        super(MimicExpAtariWriter, self).__init__(*args)


"""
    ################### Reader ##########################
"""


class MimicExpReader(Dataset):
    def __init__(self,
                 save_dir: str,
                 module_type: torch.dtype = torch.float32,
                 device: Union[torch.device, str] = 'cpu'):
        super(MimicExpReader, self).__init__()
        self.dataset_path = save_dir + "/dataset.txt"
        self.f = open(self.dataset_path, "r")
        self.save_dir = save_dir
        self.tmp_line_list = [i.strip() for i in self.f.readlines()[1:]]
        self.line_list = []

        # load hard disk data to memory
        self.data = {}
        self.to_memory()
        self.f.close()

        # tensor type
        self.dtype = module_type
        # device
        self.device = device

    # def update(self) -> None:
    #     self.f = open(self.dataset_path, "r")
    #     self.tmp_line_list = [i.strip() for i in self.f.readlines()[1:]]
    #     self.f.close()

    def to_memory(self) -> None:
        raise NotImplemented

    def _tensorlabel(self, label: str) -> torch.Tensor:
        raise NotImplemented

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        line = self.line_list[index]
        x_fs, y_f = line.split("||")
        # convert label to tensor
        label = self._tensorlabel(y_f)
        t = []
        for x in x_fs.split(","):
            t.append(self.data[x])
        # concat_frame = torch.tensor(np.concatenate(t, axis=0), dtype=self.dtype)
        concat_frame = np.concatenate(t, axis=0)
        # should change dtype to module dtype in Network Instance
        return concat_frame, label

    def __len__(self) -> int:
        return len(self.line_list)


class MimicExpClassificationReader(MimicExpReader):
    def __init__(self, *args):
        super(MimicExpClassificationReader, self).__init__(*args)

    def _tensorlabel(self, label: str) -> torch.Tensor:
        label = torch.tensor([int(float(label))], dtype=self.dtype)
        return label


class MimicExpRegressionReader(MimicExpReader):
    def __init__(self, *args):
        super(MimicExpRegressionReader, self).__init__(*args)

    def to_memory(self):
        for line in self.tmp_line_list:
            pic_filenames, label = line.split("||")
            try:
                for pic_file in pic_filenames.split(","):
                    if self.data.get(pic_file) is None:
                        self.data[pic_file] = np.load(self.save_dir + pic_file)
                        self.data[pic_file] = np.load(self.save_dir + pic_file)[None]
                self.data[label] = np.load(self.save_dir + label)
                self.line_list.append(line)
            except:
                pass

    def _tensorlabel(self, label: str) -> torch.Tensor:
        return self.data[label]


class MimicExpAtariReader(MimicExpClassificationReader):
    def __init__(self, *args):
        super(MimicExpAtariReader, self).__init__(*args)

    def to_memory(self):
        for line in self.tmp_line_list:
            pic_filenames, label = line.split("||")
            for pic_file in pic_filenames.split(","):
                if self.data.get(pic_file) is None:
                    # a.[None] is expand a's dim. such as [84, 84] -> [1, 84, 84]
                    try:
                        self.data[pic_file] = np.load(self.save_dir + pic_file)[None]
                        self.line_list.append(line)
                    except:
                        self.data[pic_file] = -1

class MimicExpMujocoReader(MimicExpRegressionReader):
    def __init__(self, *args):
        super(MimicExpMujocoReader, self).__init__(*args)


class MimicExpClassicalReader(MimicExpRegressionReader):
    def __init__(self, *args):
        super(MimicExpClassicalReader, self).__init__(*args)


"""
    ################### Factory ##########################
"""


class MimicExpFactory:
    writer_register = {
        "atari": MimicExpAtariWriter,
        "mujoco": MimicExpMujocoWriter,
        "classical": MimicExpClassicalWriter,
    }

    reader_register = {
        "atari": MimicExpAtariReader,
        "mujoco": MimicExpMujocoReader,
        "classical": MimicExpClassicalReader,
    }

    def mimic_writer(self, task_type: str, *args) -> MimicExpWriter:
        return self.writer_register[task_type](*args)

    def mimic_reader(self, task_type: str, *args) -> MimicExpReader:
        return self.reader_register[task_type](*args)


if __name__ == "__main__":
    dataset = MimicExpFactory().mimic_reader("atari", "/home/drl/drlnav_frame/mimic/Pong-v4-21-v1-16env/")
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=1)
    s = time.time()
    for batch in dataloader:
        # pass
        # print(batch)
        batch[0] = batch[0].to(torch.float32)
        #print(batch[0].dtype)
        # print(batch[1].shape)
        print(time.time() - s)
        s = time.time()