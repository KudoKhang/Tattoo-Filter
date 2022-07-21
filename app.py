import torch
from collections import OrderedDict
import torchvision.transforms as transforms

import networks
from utilss.transforms import transform_logits
from utilss.transforms import get_affine_transform
from utilss.dataset_settings import dataset_settings
from utilss.inference_funcs import *
from modules.human_detection import Detection
from modules.face import Ibug_Parsing
from modules.tattoo import AddTattoo
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class HumanParsing():
    def __init__(self, dataset='atr'):
        self.dataset = dataset
        self.input_size = dataset_settings[dataset]['input_size']
        self.aspect_ratio = self.input_size[1] * 1.0 / self.input_size[0]
        self.num_classes = dataset_settings[dataset]['num_classes']
        self.path_pretrained = dataset_settings[dataset]['path_pretrained']

        # Init model
        self.model = networks.init_model('resnet101', num_classes=self.num_classes, pretrained=None)
        self.model_detection = Detection()
        self.Face_parsing_predictor = Ibug_Parsing()
        self.addTattoo = AddTattoo()

        state_dict = torch.load(self.path_pretrained)['state_dict']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def check_type(self, img_path):
        if type(img_path) == str:
            if img_path.endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(img_path)
            else:
                raise Exception("Please input a image file")
        elif type(img_path) == np.ndarray:
            img = img_path
        return img

    def preprocessing(self, img_path):
        img = self.model_detection.run(self.check_type(img_path))
        self.face_mask = self.Face_parsing_predictor.run(img)
        self.img_copy = img.copy()
        h, w, _ = img.shape
        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.input_size)
        input = cv2.warpAffine(
            img,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        input = self.transform(input)
        input = torch.unsqueeze(input, 0)
        meta = {
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        return input, meta

    def make_color(self, masks, color=(0, 255, 0)):
        for i in range(3):
            masks[:, :, i][np.where(masks[:, :, i] == 255)] = color[i]
        return masks

    def remove_noise(self, mask):
        mask = np.uint8(mask)
        """
            IDEA: Find Contour --> Remove all area < max_area
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        list_areas = np.array(list(map(lambda c: (cv2.contourArea(c), cv2.boundingRect(c)), contours)))
        index_max_area = np.where(list_areas[:,0] == np.max(list_areas[:,0]))
        list_areas = list(list_areas)
        list_areas.pop(int(index_max_area[0]))
        list_areas = np.array(list_areas)

        for coord in list_areas[:,1]:
            x, y, w, h = coord
            points = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
            cv2.fillPoly(mask, pts=[points], color=(0, 0, 0))
        return mask

    def run(self, img_path, tattoo_path):
        image, meta = self.preprocessing(img_path)
        c = meta['center']
        s = meta['scale']
        w = meta['width']
        h = meta['height']

        output = self.model(image.to(self.device))
        upsample = torch.nn.Upsample(size=self.input_size, mode='bilinear', align_corners=True)
        upsample_output = upsample(output[0][-1][0].unsqueeze(0))
        upsample_output = upsample_output.squeeze()
        upsample_output = upsample_output.permute(1, 2, 0)

        logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=self.input_size)
        parsing_result = np.argmax(logits_result, axis=2)

        index = dataset_settings[self.dataset]['label'].index('Face')
        parsing_result[np.where(parsing_result == index)] = 255
        parsing_result[np.where(parsing_result != 255)] = 0

        parsing_result = self.remove_noise(parsing_result - self.face_mask)

        img = self.addTattoo.run(self.img_copy, parsing_result, tattoo_path=tattoo_path)

        return img

if __name__ == '__main__':
    args = get_args()
    img = image(args.input, args.tattoo ,args.save, args.plot, args.savedir)
    # video()
    # webcam(args.tattoo)
