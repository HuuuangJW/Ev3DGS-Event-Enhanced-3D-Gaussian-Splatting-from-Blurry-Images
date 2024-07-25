import torch
import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
import cv2 as cv
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def addeventbins(event_data, k, resolution_h, resolution_w):

    if k == 0:
        return 1
    else:
        event_cur = event_data[0].view(resolution_h, resolution_w)
        for m in range(1, k):
            event_cur += event_data[m].view(resolution_h, resolution_w)
        event_cur = torch.exp(the * event_cur)
        return event_cur + addeventbins(event_data, k-1, resolution_h, resolution_w)

def addeventbin(event_data, k, resolution_h, resolution_w):
    if k == 0:
        return 1
    else:
        event_cur = event_data[0].view(resolution_h, resolution_w)
        for m in range(1, k):
            event_cur += event_data[m].view(resolution_h, resolution_w)
        event_cur = torch.exp(the * event_cur)

        return event_cur

def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    input_tensor = cv.cvtColor(input_tensor, cv.COLOR_RGB2BGR)
    input_tensor = input_tensor[:, :, [2,1,0]]
    cv.imwrite(filename, input_tensor)


if __name__ == '__main__':

    input_path = "./process/input/toys"
    output_path = "./process/output/toys"
    the = -0.15
    frame_num = 5
    view_num = 30
    white_background = False
    b = 4
    resolution_h = 260
    resolution_w = 346
    event_infos = []

    print("Loading Event Data")
    event_infos = torch.load(os.path.join(input_path, "events.pt"))
    print("event size: " + str(event_infos.size()))
    event_infos = event_infos.to(device)

    for i in range(0, view_num):
        x = 1
        event_data = event_infos[i]
        blur_image_path = os.path.join(input_path, "images/{0:03d}.jpg".format(i * 5))
        blur_image = cv.imread(blur_image_path)
        transf = transforms.ToTensor()
        blur_image = transf(blur_image).to(device)
        y = addeventbins(event_data, 4, resolution_h, resolution_w)
        for j in range(0, 5):
            x = addeventbin(event_data, j, resolution_h, resolution_w)
            image = blur_image * x / y
            image = (b + 1) * image
            image = image.unsqueeze(0)
            filename = "{0:03d}.jpg".format(i * 5 + j)
            image_path = os.path.join(output_path, filename)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            save_image_tensor2cv2(image, image_path)



