# Robotic Bin-Picking Perception

This project builds a computer vision, 3D geometry, and machine learning pipeline for robotic bin-picking scenes. The system takes cluttered industrial bin images with depth data, predicts or uses visible object masks, extracts object-level 3D geometry, and ranks visible parts as pick candidates.

The final pipeline combines:

1. visible instance segmentation
2. depth-based point-cloud geometry
3. object-level feature extraction
4. a learned PyTorch pickability ranker
5. visual and quantitative evaluation

## Final Integrated Example

The example below shows the full comparison used in the final evaluation. The first panel shows the heuristic ranking with dataset masks. The second panel shows the learned ranker with dataset masks. The third panel shows the integrated YOLO segmentation plus learned ranker result.

<p align="center">
  <img src="reports/figures/scene_000000_image_000000_three_way_visual_comparison.png" alt="Integrated YOLO segmentation and pickability ranker result" width="100%">
</p>

## Key Results

### Learned ranker with dataset masks

The learned pickability model is a PyTorch multilayer perceptron that replaces the hand-designed heuristic during inference. It takes one object candidate at a time and predicts a pickability score from depth, visibility, geometry, image-position, and part-type features.

| Metric | Value |
|---|---:|
| Candidate rows | 2016 |
| Validation images | 75 |
| R2 | 0.9307 |
| Pearson correlation | 0.9661 |
| Top-1 agreement with heuristic | 0.7200 |
| Heuristic #1 in model top 3 | 0.9733 |
| NDCG@3 | 0.9965 |
| Mean regret | 0.0034 |

The learned ranker fits the heuristic target closely and usually preserves the same high-quality candidate group, even when the exact order changes.

### Integrated YOLO segmentation plus ranker

The integrated pipeline replaces dataset masks with YOLO-predicted masks and part classes. Each predicted mask is converted into depth and geometry features, then scored by the same learned PyTorch ranker.

| Comparison | Top-1 agreement | Integrated #1 in reference top 3 | Reference #1 in integrated top 3 | NDCG@3 | Mean regret |
|---|---:|---:|---:|---:|---:|
| Integrated vs heuristic | 0.5467 | 0.8187 | 0.7840 | 0.9818 | 0.0170 |
| Integrated vs known-mask ranker | 0.5733 | 0.8320 | 0.8453 | 0.9861 | 0.0117 |

The integrated top pick appears in the heuristic top 3 in 81.9 percent of images and in the known-mask ranker top 3 in 83.2 percent of images. This shows that the integrated system usually selects from the same high-quality pick region, even when upstream segmentation changes the candidate set.

## Method

### Object-level geometry

For each visible object candidate, the pipeline extracts pixels from the object mask and back-projects the corresponding depth values into a 3D point cloud using the camera intrinsics. The extracted candidate features include:

- visible pixel count
- valid 3D point count
- median, minimum, and maximum depth
- object centroid
- approximate 3D extents
- image-space center
- bounding-box position
- PCA orientation axis
- part type

### Heuristic baseline

The heuristic score ranks candidates using visible area, visibility, depth coverage, camera-relative depth, image position, and geometry confidence. This creates an interpretable baseline and a training target for the learned ranker.

### Learned pickability model

The learned model is a PyTorch multilayer perceptron trained to predict the heuristic pickability score. The model receives one object-candidate feature vector at a time.

```text
object mask + depth values
→ object-level 3D geometry features
→ PyTorch MLP
→ predicted pickability score
```

Candidates are ranked within each image by the predicted score.

The model is used in two settings:

1. **Known-mask learned ranker**: features are computed from dataset-provided visible masks.
2. **Integrated YOLO plus ranker**: YOLO predicts visible masks and part classes, then the same learned ranker scores the predicted candidates.

### Segmentation integration

The segmentation stage uses Ultralytics YOLO for class-aware visible instance segmentation. The predicted mask and class for each visible part are passed into the same feature extraction and ranking pipeline.

YOLO inference used:

```text
confidence = 0.25
IoU threshold = 0.50
image size = 640
retina masks = enabled
```

## Results Figures

### Learned ranker validation

<p align="center">
  <img src="reports/figures/pickability_ranker_score_summary.png" alt="Learned pickability ranker validation summary" width="100%">
</p>

### Feature importance

<p align="center">
  <img src="reports/figures/feature_importance.png" alt="Permutation feature importance" width="100%">
</p>

### Integrated pipeline evaluation

<p align="center">
  <img src="reports/figures/integrated_comparison_summary.png" alt="Integrated YOLO segmentation plus ranker evaluation" width="100%">
</p>

## Technical Report

A detailed technical report with method details, validation metrics, feature-importance analysis, integrated-pipeline evaluation, and qualitative examples is available here:

[Technical Report](reports/technical_report.md)

## Local Generated Outputs

Generated files are written locally and are ignored by Git. Curated report files are copied into `reports/` for GitHub rendering.

Typical local output locations:

```text
outputs/
  figures/      visualizations, validation plots, model comparison figures
  reports/      CSV and JSON summaries from ranking and evaluation scripts
  models/       trained PyTorch ranker checkpoints
  pointclouds/  extracted object point clouds

runs/
  segment/      Ultralytics YOLO training outputs and weights

data/
  xyzibd/                local dataset files
  segmentation_yolo/     converted YOLO segmentation training set
```

Downloaded or trained `.pt` model files are ignored by Git.

## Setup

Create and activate a Python environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install CUDA PyTorch for NVIDIA GPU training:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining packages:

```powershell
python -m pip install -r requirements.txt
```

## Main Scripts

```text
scripts/analyze_scene_objects.py              heuristic ranking visualization
scripts/build_candidate_dataset.py            object-candidate feature table
scripts/train_pickability_ranker.py           learned PyTorch ranker training
scripts/score_pickability_ranker.py           ranker validation metrics
scripts/analyze_feature_importance.py         permutation feature importance
scripts/prepare_yolo_segmentation_dataset.py  YOLO segmentation label conversion
scripts/train_segmentation_yolo.py            YOLO segmentation training
scripts/run_integrated_yolo_ranker.py         three-way visual comparison
scripts/score_integrated_yolo_ranker.py       integrated pipeline metrics
```

## Dataset and Attribution

This project uses the XYZ Industrial Bin-Picking Dataset in BOP format. The dataset page lists the license as CC BY-NC-SA 4.0. Dataset files are not included in this repository.

```bibtex
@article{huang2025xyzibd,
  title={XYZ-IBD: A High-precision Bin-picking Dataset for Object 6D Pose Estimation Capturing Real-world Industrial Complexity},
  author={Huang, Junwen and Liang, Jizhong and Hu, Jiaqi and Sundermeyer, Martin and Yu, Peter KT and Navab, Nassir and Busam, Benjamin},
  journal={arXiv preprint arXiv:2506.00599},
  year={2025}
}
```

The segmentation phase uses Ultralytics YOLO. YOLO checkpoints and training outputs are not included in this repository. Users should review Ultralytics licensing terms before training, distributing, or deploying YOLO models.

```bibtex
@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```

## License

The code and documentation in this repository are released under the MIT License. The MIT License applies to the code and documentation created for this project. It does not relicense the XYZ-IBD dataset, Ultralytics YOLO, YOLO checkpoints, or dataset-derived artifacts.
