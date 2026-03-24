import numpy as np

from interactive_seg_backend.extensions import ExpertSegClassifier
from interactive_seg_backend.file_handling import load_image, load_labels, save_segmentation
from interactive_seg_backend.configs import TrainingConfig, FeatureConfig, ClassInfo

from interactive_seg_backend import featurise, apply

img = load_image("tests/data/1.tif")
labels = load_labels("tests/data/1_labels.tif")
gt = load_labels("tests/data/1_ground_truth.tif")

classes = np.unique(gt)
vfs = {int(i): float(np.mean(np.where(gt == i, 1, 0))) for i in classes}

# class_infos = [ClassInfo(name=f"{k}", value=k, desired_volume_fraction=v) for k, v in vfs.items()]
class_infos = [
    ClassInfo(name="background", value=0, desired_volume_fraction=0.1),
    ClassInfo(name="object", value=1, desired_volume_fraction=0.9),
]

feature_cfg = FeatureConfig()
training_cfg = TrainingConfig(feature_config=feature_cfg, class_infos=class_infos)
feats = featurise(img, training_cfg)

model = ExpertSegClassifier(
    class_infos=class_infos, n_epochs=100, lambd_vf=2.0, extra_args={"max_depth": 2, "eta": 0.1}
)
print(feats.shape, labels.shape)
model.fit(feats, labels)
preds = model.predict(feats) - 1
print(preds.shape)


pred_vfs = {int(i): float(np.mean(np.where(preds == i, 1, 0))) for i in classes}
print(vfs)
print(pred_vfs)
save_segmentation(preds, "./examples/1_expertseg.tif")
