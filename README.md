# CLIP-based Open-set 3D Object Retrieval

This project implements open-set 3D object retrieval using multi-view RGB images and depth maps rendered from 3D models, with a CLIP backbone.

---

## Experiments

### 1️⃣ RGB Multi-view CLIP

- **Backbone:** CLIP ViT-B/32  
- **Input:** 12-view RGB images per 3D object, 224 × 224 resolution  
- **Feature Extraction:**  
  - Each view encoded independently via CLIP image encoder  
  - View-level features aggregated using mean pooling (`multi_view=False` in scripts)  
- **Evaluation:**  
  - Open-set 3D object retrieval  
  - Metrics: AUROC, FPR@95TPR  

---

### 2️⃣ Depth Multi-view CLIP

- **Input:** 12-view rendered depth maps aligned with RGB cameras, converted to 3-channel images  
- **Feature Extraction:**  
  - Same CLIP image encoder as RGB  
  - Aggregated using mean pooling   
- **Evaluation:**  
  - Open-set 3D object retrieval  
  - Metrics: AUROC, FPR@95TPR  

---

### 3️⃣ RGB + Depth Feature Fusion

- **Fusion Method:** Weighted sum of RGB and Depth features:  

  $$fused = \alpha \cdot rgb\_feat + (1 - \alpha) \cdot depth\_feat$$

- **Feature Settings:**  
  - RGB feature: `multi_view=False`  
  - Depth feature: `multi_view=False` 
- **Evaluation:**  
  - Same open-set retrieval metrics  
  - Fusion weight tested: e.g., $\alpha = 0.3$

---

## Open-set Retrieval Protocol

- **Total categories:** 50  
- **Known (seen) classes:** 40  
- **Unknown (unseen) classes:** 10  

- **Gallery:** 70% of samples from known classes only  
- **Queries:** Remaining 30% of known class samples + all unknown class samples  
- **Open-set score:** Maximum cosine similarity to gallery