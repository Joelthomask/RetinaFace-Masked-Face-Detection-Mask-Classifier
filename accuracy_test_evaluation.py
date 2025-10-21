import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from retinaface.models.retinaface import RetinaFace
from retinaface.utils.box_utils import decode, decode_landm
from retinaface.layers.functions.prior_box import PriorBox
from retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from retinaface.data import cfg_mnet, cfg_re50

# --- Configuration ---
RETINAFACE_MODEL_PATH = 'retinaface/weights/ResNet50_Final.pth'
MASK_MODEL_PATH       = 'checkpoints/mobilenet_mask_best.pth.tar'
TEST_DIR              = 'data/Dataset/test'
DEVICE                = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NETWORK               = 'resnet50'      # or 'mobile0.25'
CONF_THRESH           = 0.6
NMS_THRESH            = 0.4
PADDING_RATIO         = 0.1

# select RetinaFace config
cfg = cfg_re50 if NETWORK == 'resnet50' else cfg_mnet

# --- Load RetinaFace detector ---
detector = RetinaFace(cfg=cfg, phase='test')
ckpt_rf = torch.load(RETINAFACE_MODEL_PATH, map_location=DEVICE)
raw_rf  = ckpt_rf.get('state_dict', ckpt_rf)
rf_state = {k.replace('module.', ''): v for k, v in raw_rf.items()}
detector.load_state_dict(rf_state, strict=False)
detector.to(DEVICE).eval()

# --- Load MobileNetV2 mask classifier ---
class MaskClassifier(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, x):
        return self.model(x)

def load_classifier(path):
    base = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
    base.classifier[1] = torch.nn.Linear(base.last_channel, 2)
    model = MaskClassifier(base)
    ckpt = torch.load(path, map_location=DEVICE)
    raw = ckpt.get('state_dict', ckpt)
    model.load_state_dict(raw)
    model.to(DEVICE).eval()
    return model

classifier = load_classifier(MASK_MODEL_PATH)

# --- Preprocessing transform for classifier ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_face(face_img):
    pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    return transform(pil).unsqueeze(0).to(DEVICE)

def detect_and_classify(image_path):
    """Returns (predicted_label, detected_flag)."""
    img_raw = cv2.imread(image_path)
    if img_raw is None:
        return None, False

    h, w = img_raw.shape[:2]

    # prepare input for RetinaFace
    img = img_raw.astype(np.float32)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    tensor = torch.from_numpy(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        loc, conf, landms = detector(tensor)

    # decode boxes & scores
    priorbox = PriorBox(cfg, image_size=(h, w))
    priors   = priorbox.forward().to(DEVICE)
    boxes    = decode(loc.squeeze(0), priors.data, cfg['variance'])
    boxes    = (boxes * torch.Tensor([w, h, w, h]).to(DEVICE)).cpu().numpy()
    scores   = conf.squeeze(0)[:, 1].detach().cpu().numpy()

    # filter by confidence
    inds   = np.where(scores > CONF_THRESH)[0]
    if inds.size == 0:
        return None, False

    boxes  = boxes[inds]
    scores = scores[inds]

    # sort & NMS
    order = scores.argsort()[::-1]
    boxes  = boxes[order]
    scores = scores[order]
    dets   = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
    keep   = py_cpu_nms(dets, NMS_THRESH)
    dets   = dets[keep]

    if dets.size == 0:
        return None, False

    # use highest-score detection
    x1, y1, x2, y2, _ = dets[0]
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    bw, bh = x2 - x1, y2 - y1
    px = int(bw * PADDING_RATIO)
    py = int(bh * PADDING_RATIO)

    # apply padding and crop
    x1c = max(0, x1 - px)
    y1c = max(0, y1 - py)
    x2c = min(w, x2 + px)
    y2c = min(h, y2 + py)
    face_crop = img_raw[y1c:y2c, x1c:x2c]

    # classify mask/no-mask
    tensor_face = preprocess_face(face_crop)
    with torch.no_grad():
        logits = classifier(tensor_face)
        probs  = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    idx   = int(np.argmax(probs))
    label = 'Mask' if idx == 0 else 'No Mask'

    return label, True

# --- Evaluation loop ---
classes = {
    'Mask':    'Mask',     # folder name -> true label
    'No_mask': 'No Mask'
}

results = {}
for folder, true_label in classes.items():
    folder_path = os.path.join(TEST_DIR, folder)
    all_files   = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    total     = len(all_files)
    correct   = 0
    wrong     = 0
    skipped   = 0

    for fname in all_files:
        img_path = os.path.join(folder_path, fname)
        pred, detected = detect_and_classify(img_path)
        if not detected:
            skipped += 1
        elif pred == true_label:
            correct += 1
        else:
            wrong += 1

    results[folder] = {
        'total':   total,
        'correct': correct,
        'wrong':   wrong,
        'skipped': skipped
    }

# --- Report ---
print("\n=== Mask Classifier Evaluation Report ===\n")
overall = {'total': 0, 'correct': 0, 'wrong': 0, 'skipped': 0}
for folder, stats in results.items():
    print(f"{folder:10s}: {stats['correct']:2d}/{stats['total']:2d} correct, "
          f"{stats['wrong']:2d} wrong, {stats['skipped']:2d} skipped")
    for k in overall:
        overall[k] += stats[k]

accuracy = overall['correct'] / overall['total'] * 100 if overall['total'] > 0 else 0.0
print(f"\nOverall: {overall['correct']}/{overall['total']} correct, "
      f"{overall['wrong']} wrong, {overall['skipped']} skipped")
print(f"Accuracy: {accuracy:.1f}%\n")