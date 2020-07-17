from __future__ import absolute_import

import random
import math
import numpy as np


class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = list(frame_indices)

        while len(out) < self.size:
            for index in out:
                if len(out) >= self.size:
                    break
                out.append(index)

        return out


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size, padding=True, pad_method='loop'):
        self.size = size
        self.padding = padding
        self.pad_method = pad_method

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = list(frame_indices[begin_index:end_index])

        if self.padding == True:
            if self.pad_method == 'loop':
                while len(out) < self.size:
                    for index in out:
                        if len(out) >= self.size:
                            break
                        out.append(index)
            else:
                while len(out) < self.size:
                    for index in out:
                        if len(out) >= self.size:
                            break
                        out.append(index)
                out.sort()

        return out


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size=4, stride=8):
        self.size = size
        self.stride = stride

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        frame_indices = list(frame_indices)

        if len(frame_indices) >= self.size * self.stride:
            rand_end = len(frame_indices) - (self.size - 1) * self.stride - 1
            begin_index = random.randint(0, rand_end)
            end_index = begin_index + (self.size - 1) * self.stride + 1
            out = frame_indices[begin_index:end_index:self.stride]
        elif len(frame_indices) >= self.size:
            index = np.random.choice(len(frame_indices), size=self.size, replace=False)
            index.sort()
            out = [frame_indices[index[i]] for i in range(self.size)]
        else:
            index = np.random.choice(len(frame_indices), size=self.size, replace=True)
            index.sort()
            out = [frame_indices[index[i]] for i in range(self.size)]

        return out


class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size=4):
        self.size = size
        
    def __call__(self, frame_indices):
        frame_indices = list(frame_indices)

        if len(frame_indices) >= 25:
            out = frame_indices[0:25:8]
        elif len(frame_indices) >= 13:
            out = frame_indices[0:13:4]
        elif len(frame_indices) >= 7:
            out = frame_indices[0:7:2]
        elif len(frame_indices) >= 4:
            out = frame_indices[0:4:1]
        else:
            out = frame_indices[0:4]
            while len(out) < 4:
                for index in out:
                    if len(out) >= 4:
                        break
                    out.append(index)

        return out

# class TemporalBeginCrop(object):
#     """Temporally crop the given frame indices at a beginning.

#     If the number of frames is less than the size,
#     loop the indices as many times as necessary to satisfy the size.

#     Args:
#         size (int): Desired output size of the crop.
#     """

#     def __init__(self, size=4):
#         self.size = size
        
#     def __call__(self, frame_indices):
#         frame_indices = list(frame_indices)

#         if len(frame_indices) >= 4:
#             out = frame_indices[0:4:1]
#         else:
#             out = frame_indices[0:4]
#             while len(out) < 4:
#                 for index in out:
#                     if len(out) >= 4:
#                         break
#                     out.append(index)

#         return out