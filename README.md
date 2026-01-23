# 3D Data Processing Toolkit
Convert **GLB → Multi-view Images**, **Mesh → Point Clouds**, **OBJ → Multi-view Images**

This repository provides lightweight tools for converting and processing 3D models, including multi-view rendering and point cloud sampling.

---

## Features
### GLB → Multi-view Rendering
Scripts:
- `batch_render.sh`
- `render_single_glb.py`

Example (BlenderProc):
```bash
blenderproc run \
  --custom-blender-path {BLD_DIR} \
  --blender-install-path {BLD_DIR} \
  render_single_glb.py \
  --object_path {glb_path} \
  --output_dir {out_folder} \
  --engine CYCLES \
  --num_images 12 \
  --camera_dist 1.2
```

https://github.com/panmari/stanford-shapenet-renderer
https://github.com/Colin97/OpenShape_code/blob/master/render_single_glb.py


### Mesh → Point Cloud

Files:
- Outputs `.npy` point cloud files
- `misc_utils.py`

https://github.com/Colin97/OpenShape_code/issues/16

### OBJ → Multi-view Rendering

- Load `.obj` models with or without textures

Example (Blender 2.79b):
```bash
/home/qrhan/datasets/missing/3D-FUTURE-ToolBox/blender-2.79b-linux-glibc219-x86_64/blender \
  --background \
  --python demo_render_object_mv_without_texture.py -- \
  --views 12 \
  --output_folder ./f4a7af7a-d1e1-43db-8d66-c7285bf13053
```
  https://github.com/3D-FRONT-FUTURE/3D-FUTURE-ToolBox/tree/master