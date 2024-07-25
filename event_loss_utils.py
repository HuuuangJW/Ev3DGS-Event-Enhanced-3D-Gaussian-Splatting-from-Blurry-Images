import torch
import math
import random

def lin_log(x, threshold=20):
    """
    linear mapping + logarithmic mapping.
    :param x: float or ndarray the input linear value in range 0-255
    :param threshold: float threshold 0-255 the threshold for transisition from linear to log mapping
    """
    # converting x into np.float32.
    if x.dtype is not torch.float64:
        x = x.double()
    f = (1./threshold) * math.log(threshold)
    y = torch.where(x <= threshold, x*f, torch.log(x))

    return y.float()


def image2gray(rgb):
    r, g, b = rgb[0, :, :], rgb[1, :, :], rgb[2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def event_loss_call(all_rgb, event_data, combination, rgb2gray, resolution_h, resolution_w):
    '''
    simulate the generation of event stream and calculate the event loss
    '''
    # if rgb2gray == "rgb":
    #     rgb2grey = torch.tensor([0.299, 0.587, 0.114]).to("cuda:0")
    # elif rgb2gray == "ave":
    #     rgb2grey = torch.tensor([1/3, 1/3, 1/3]).to("cuda:0")
    loss = []

    chose = random.sample(combination, 10)
    for its in range(10):
        start = chose[its][0]
        end = chose[its][1]

        thres_pos = (lin_log(image2gray(all_rgb[end]) * 255) - lin_log(image2gray(all_rgb[start]) * 255)) / 0.3
        thres_neg = (lin_log(image2gray(all_rgb[end]) * 255) - lin_log(image2gray(all_rgb[start]) * 255)) / 0.2

        event_cur = event_data[start].view(resolution_h, resolution_w)
        for j in range(start + 1, end):
                event_cur += event_data[j].view(resolution_h, resolution_w)

        pos = event_cur > 0
        neg = event_cur < 0

        loss_pos = torch.mean(((thres_pos * pos) - ((event_cur + 0.5) * pos)) ** 2)
        loss_neg = torch.mean(((thres_neg * neg) - ((event_cur - 0.5) * neg)) ** 2)
        if loss_pos + loss_neg <= 10:
            loss.append(loss_pos + loss_neg)

    event_loss = torch.mean(torch.stack(loss, dim=0), dim=0)
    return event_loss
