import numpy as np
import struct
import math
import torch
import redis
import marshal

from typing import Any, Dict, List, Tuple, Union, Optional, Callable, Generator


def mul(x):
    out = 1
    for i in x:
        out *= i
    return out


class EasyBytes():


    d = {
        1 : (1, np.uint8),
        2 : (2, np.float16),
        3 : (4, np.float32),
        4 : (8, np.float64),
    }

    def __init__(self, machine="127.0.0.1"):
        t_ = machine.split(".")

        self.machine_bytes: bytes = b""
        for t in t_:
            assert 0 <= int(t) < 256
            self.machine_bytes += struct.pack(">H", int(t))

    def _get_t(self, data_dtype: np.dtype):
        t = None
        if data_dtype == np.uint8: t = 1
        elif data_dtype == np.float16: t = 2
        elif data_dtype == np.float32: t = 3
        elif data_dtype == np.float64: t = 4
        else:
            print("EasyBytes: Match data type error !", data_dtype, flush=True)
            raise ValueError
        return struct.pack(">h", t)

    def decode_data(self, bytes_data: bytes) -> List[np.ndarray]:
        index, length = 0, len(bytes_data)
        list_np_data = []
        while index < length:
            c_type = struct.unpack(">h", bytes_data[index: index+2])[0]
            tb, t = self.d[c_type]
            count, shape_dim = struct.unpack(">II", bytes_data[index+2: index+10])

            shape = struct.unpack(">" + "I" * shape_dim,
                                  bytes_data[index + 10: index + 10 + shape_dim * 4])
            index = index + 10 + shape_dim * 4
            each_bytes_data = bytes_data[index: index + count*tb]
            list_np_data.append(np.frombuffer(each_bytes_data, dtype=t).reshape(*shape))
            index += count*tb
        return list_np_data

    def encode_data(self, np_list_data: List[np.ndarray]) -> bytes:
        bytes_out = b""
        for np_data in np_list_data:
            shape = np_data.shape
            shape_dim = len(shape)
            count = mul(shape)

            bytes_out += self._get_t(np_data.dtype)
            bytes_out += struct.pack(">II", count, shape_dim)
            bytes_out += struct.pack(">" + "I" * shape_dim, *shape)

            bytes_out += np_data.tobytes()
        return bytes_out

    def encode_forward_return_data(self,
                                   list_forward_return_np_data: List[Union[np.ndarray, torch.Tensor]],
                                   list_env_batch_num: List[int],
                                   ) -> List[bytes]:
        """
            encode data(action, logp) from forward net to bytes, and send it to message queue for agent process.
        """
        # print("data", data, flush=True)
        # change tensor to np
        for i in range(len(list_forward_return_np_data)):
            if isinstance(list_forward_return_np_data[i], torch.Tensor):
                list_forward_return_np_data[i] = list_forward_return_np_data[i].cpu().detach().numpy()

        list_env_bytes = []
        index = 0
        for int_env_batch_num in list_env_batch_num:
            # the bytes data return to agent.
            i = 0
            list_each_env_forward_np_data: List[np.ndarray] = []
            for np_forward_data in list_forward_return_np_data:
                # each env process get correct data with env_dict
                if i == 2:
                    # values .
                    index_data = np_forward_data[:, index: index+int_env_batch_num]
                else:
                    index_data = np_forward_data[index: index+int_env_batch_num]
                list_each_env_forward_np_data.append(index_data)
                i += 1

            list_env_bytes.append(self.encode_data(list_each_env_forward_np_data))
            index += int_env_batch_num

        return list_env_bytes

    def decode_forward_states(self, byte_states: bytes) -> Tuple[List[str], List[np.ndarray]]:
        """
        decode batched forward states from predicting deque,
        return listed process_env_id and listed states
        """
        len_byte_states = len(byte_states)
        index = 0
        list_np_forward_states: List[np.ndarray] = []
        list_process_env_ids = []
        tmp_list_env_states = []
        while index < len_byte_states:
            length = struct.unpack(">Q", byte_states[index: index+8])[0]
            index += 8
            ip = map(str, struct.unpack(">HHHH", byte_states[index: index+8]))
            index += 8
            process_env_id = struct.unpack(">I", byte_states[index: index+4])[0]
            index += 4
            tmp_list_env_states.append(self.decode_data(byte_states[index: index+length]))
            list_process_env_ids.append(".".join(list(ip)) + "_" + str(process_env_id))
            index += length
        # TODO 测试predicting queue batch 之后在这里的时间消耗
        for i in range(len(tmp_list_env_states[0])):
            xxx = []
            for j in range(len(tmp_list_env_states)):
                xxx.append(tmp_list_env_states[j][i])

            list_np_forward_states.append(np.concatenate(xxx, axis=0))

        return list_process_env_ids, list_np_forward_states

    def encode_forward_states(self, process_env_id: int,
                              list_np_states: List[np.ndarray]) -> bytes:
        out = b""
        out += self.machine_bytes # 8 bytes
        out += struct.pack(">I", process_env_id) # 4 bytes
        out += self.encode_data(list_np_states)

        length = struct.pack(">Q", len(out) - 12 )
        return length + out

    def encode_backward_data(self, list_np_data: List[Union[List[np.ndarray], np.ndarray]], dict_logger: Dict
                             ) -> bytes:
        """
        encode training data to bytes
            [state, r, a, logp, v]
        """
        bytes_out = b""
        bytes_states = self.encode_data(list_np_data[0])
        bytes_out += struct.pack(">Q", len(bytes_states)) + bytes_states
        bytes_other4 = self.encode_data(list_np_data[1:])
        bytes_out += struct.pack(">Q", len(bytes_other4)) + bytes_other4
        return bytes_out + marshal.dumps(dict_logger)

    def decode_backward_data(self, bytes_data: bytes) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
        int_len_states = struct.unpack(">Q", bytes_data[:8])[0]
        list_np_states = self.decode_data(bytes_data[8: 8+int_len_states])
        int_len_other4 = struct.unpack(">Q", bytes_data[8+int_len_states: 16+int_len_states])[0]
        list_np_other4 = self.decode_data(bytes_data[16+int_len_states: 16+int_len_states+int_len_other4])

        dict_logger: Dict = marshal.loads(bytes_data[16+int_len_states+int_len_other4:])
        return list_np_states, list_np_other4, dict_logger



def test_decode_backward_data():
    s = EasyBytes()
    x = s.encode_backward_data([[np.array([[1,2,3],[4,5,6]], dtype=np.uint8), np.array([4,6], dtype=np.float32)],
      np.array([[4,5],[6,7]], dtype=np.float32), np.array([[4,5],[6,7]],dtype=np.float32), np.array([[4,5],[6,7]],dtype=np.float32), np.array([[4,5],[6,7]],dtype=np.float32)
    ],{"c":np.mean(6.6)})
    decoded_data = s.decode_backward_data(x)
    from USTC_lab.data import Experience
    tensor_data = Experience(decoded_data[0], *decoded_data[1])
    tensor_data.to_tensor()
    print(tensor_data.get_xrapv())
    print(decoded_data[2])



def test_forward_states():
    s = EasyBytes()
    x = np.array([[1,2,3,4]],dtype=np.float32)
    y = s.encode_forward_states(1, [x])
    print(y)
    print(s.decode_forward_states(y))

if __name__ == "__main__":
    # test_forward_states()
    test_decode_backward_data()
    # s = EasyBytes()
    # a = s.encode_forward_states(0, [np.array([[1,2,3],[4,5,6]], dtype=np.uint8), np.array([4,6], dtype=np.float32)])
    # b = s.encode_forward_states(1, [np.array([[1,2,3],[4,5,6]], dtype=np.uint8), np.array([4,6], dtype=np.float32)])
    #
    # print(s.decode_forward_states(a+b))
    # x = s.encode_data([np.array([[1,2,3],[4,5,6]], dtype=np.uint8), np.array([4,5,6], dtype=np.float32)])
    # print(x)
    # print(s.decode_data(x))