# Dataset Notes

This repository does not store raw 3D datasets.

The current codebase expects large datasets and generated artifacts to live outside the repo under the local workspace, for example:

- `D:/1Ahaha/AA3d/ShapeNet`
- `D:/1Ahaha/AA3d/output_224`
- `D:/1Ahaha/AA3d/ShapeNet_PointClouds`
- `D:/1Ahaha/AA3d/depth_maps`

This `datasets/` directory is reserved for lightweight assets that help document or reproduce experiments, such as:

- dataset statistics
- class lists and split manifests
- benchmark notes for HGM2R datasets
- small adapter scripts that do not include raw dataset files

Suggested subfolders:

- `datasets/shapenet/`
- `datasets/hgm2r_benchmarks/`
