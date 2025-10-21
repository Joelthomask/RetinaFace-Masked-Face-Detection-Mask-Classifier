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

# --- Config ---
RETINAFACE_MODEL_PATH = 'retinaface/weights/ResNet50_Final.pth'
MASK_MODEL_PATH       = 'checkpoints/mobilenet_mask_best.pth.tar'
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
raw_rf = ckpt_rf.get('state_dict', ckpt_rf)
rf_state = {k.replace('module.', ''): v for k, v in raw_rf.items()}
detector.load_state_dict(rf_state, strict=False)
detector.to(DEVICE).eval()

# --- Load Mask Classifier ---
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
    raw_cls = ckpt.get('state_dict', ckpt)
    model.load_state_dict(raw_cls)
    model.to(DEVICE).eval()
    return model

classifier = load_classifier(MASK_MODEL_PATH)

# --- Preprocessing for classifier ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_face(face_img):
    pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    return transform(pil).unsqueeze(0).to(DEVICE)

def classify(face_img):
    tensor = preprocess_face(face_img)
    with torch.no_grad():
        logits = classifier(tensor)
        probs  = torch.softmax(logits, dim=1)[0]\
                     .detach().cpu().numpy()
    idx       = int(np.argmax(probs))
    label     = 'Mask' if idx == 0 else 'No Mask'
    confidence= float(probs[idx])
    return label, confidence

# --- Run live webcam inference ---
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # prepare tensor for RetinaFace
        img = frame.astype(np.float32)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            loc, conf, landms = detector(img_tensor)

        # decode
        scale = torch.Tensor([w, h, w, h]).to(DEVICE)
        priorbox = PriorBox(cfg, image_size=(h, w))
        priors = priorbox.forward().to(DEVICE)

        boxes = decode(loc.squeeze(0), priors.data, cfg['variance'])
        boxes = (boxes * scale).cpu().numpy()
        scores = conf.squeeze(0)[:, 1].detach().cpu().numpy()

        landms = decode_landm(landms.squeeze(0), priors.data, cfg['variance'])
        scale1 = torch.Tensor([w, h] * 5).to(DEVICE)
        landms = (landms * scale1).cpu().numpy()

        # filter & nms
        keep_inds = np.where(scores > CONF_THRESH)[0]
        boxes = boxes[keep_inds]
        scores= scores[keep_inds]
        landms= landms[keep_inds]

        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores= scores[order]
        landms= landms[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
        keep = py_cpu_nms(dets, NMS_THRESH)
        dets = dets[keep]

        # annotate frame
        for det in dets:
            if det[4] < CONF_THRESH:
                continue
            x1, y1, x2, y2 = map(int, det[:4])
            bw, bh = x2 - x1, y2 - y1
            px = int(bw * PADDING_RATIO)
            py = int(bh * PADDING_RATIO)

            # apply padding
            x1c = max(0, x1 - px)
            y1c = max(0, y1 - py)
            x2c = min(w, x2 + px)
            y2c = min(h, y2 + py)

            face_crop = frame[y1c:y2c, x1c:x2c]
            label, conf_score = classify(face_crop)

            color = (0,255,0) if label=='Mask' else (0,0,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            text = f"{label}: {conf_score*100:.1f}%"
            cv2.putText(frame, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Live Mask Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()