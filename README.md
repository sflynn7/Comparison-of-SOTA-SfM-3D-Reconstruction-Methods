# Comparison of SOTA SfM 3D Reconstruction Methods

## Abstract
3D reconstruction from 2D images is a fundamental problem in computer vision with applications in robotics, augmented and virtual reality, and spatial computing. While recent AI-based methods have introduced new approaches for reconstruction, systematic comparisons with classical Structure-from-Motion (SfM) pipelines remain limited. This work benchmarks three state-of-the-art AI-based methods (VGGT, Pi3, and Depth Anything 3) and the classical COLMAP pipeline. The analysis uses a custom indoor dataset made up of five scenes captured with an iPhone 15 Pro and a Gemini 335Lg stereo vision camera. Stereo depth point clouds serve as ground truth for quantitative evaluation. Reconstructions are assessed using Chamfer Distance, Hausdorff Distance, accuracy, completeness, F-score, and reconstruction speed after scale alignment and ICP registration. Results show that no single method consistently outperforms the others across all scenes and metrics. COLMAP achieves the highest overall F-score due to more complete reconstructions, while Depth Anything 3 produces high accuracy but limited completeness. AI-based methods demonstrate promising performance but remain sensitive to scene characteristics such as texture and geometric complexity. These findings highlight trade-offs between classical and modern approaches and motivate further evaluation on larger and more diverse datasets.

## Results

<img width="1506" height="1406" alt="3D Reconstruction highlights" src="https://github.com/user-attachments/assets/c122f6da-6845-4151-9cae-cbfd7e1e371a" />

