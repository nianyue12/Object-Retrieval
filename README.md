# CLIP-based Open-set 3D Object Retrieval

This project contains experiments for open-set 3D object retrieval using
multi-view RGB images and rendered depth maps with a CLIP model.

---

## Experiments

### 1. RGB Multi-view CLIP 

**Backbone**
- CLIP ViT-B/32 

**Input**
- 12-view RGB images rendered per 3D object
- Image resolution: 224 × 224

**Feature Extraction**
- Each view is encoded independently by the CLIP image encoder
- View-level features are aggregated via mean pooling

**Evaluation**
- Open-set 3D object retrieval
- Metrics:
  - AUROC
  - FPR@95TPR

---

### 2. Depth Multi-view CLIP 

**Input**
- 12-view rendered depth maps aligned with RGB camera parameters
- Depth maps are converted to 3-channel images for CLIP compatibility

**Feature Extraction**
- The same CLIP image encoder is used 
- Depth features are aggregated via mean pooling

**Evaluation**
- Identical open-set protocol as RGB experiments

---

## Open-set Retrieval Setting

- Object categories are split into known (seen) and unknown (unseen) classes
- Gallery contains only known-class objects
- Queries include both known and unknown objects
- Open-set score is defined as the maximum cosine similarity to the gallery

---


