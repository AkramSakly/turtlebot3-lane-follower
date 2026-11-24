# evaluate_all.py results (labelled dataset turtlebot3_autorace_dataset_to_fill.zip, 264 label files)
Run on 2026-11-24 on the robot PC with the ONNX models in ros2_package/tb3_lane_center/models.

## Segmentation (pixel level, threshold folded into the ONNX model)

| Network | Split | Images | IoU | Dice/F1 | Precision | Recall | Specificity | Pixel acc. | mIoU (line+bg) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yellow | train | 183 | 0.837 | 0.911 | 0.906 | 0.917 | 0.998 | 0.997 | 0.917 |
| yellow | validation | 61 | 0.718 | 0.836 | 0.829 | 0.842 | 0.997 | 0.994 | 0.856 |
| yellow | test | 14 | 0.828 | 0.906 | 0.893 | 0.919 | 0.998 | 0.997 | 0.912 |
| white | train | 183 | 0.773 | 0.872 | 0.898 | 0.848 | 0.999 | 0.997 | 0.885 |
| white | validation | 61 | 0.632 | 0.775 | 0.855 | 0.708 | 0.998 | 0.994 | 0.813 |
| white | test | 14 | 0.764 | 0.866 | 0.872 | 0.861 | 0.998 | 0.997 | 0.881 |

## Confusion matrices (pixels)

| Network | Split | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- |
| yellow | train | 148441 | 15481 | 13389 | 8817505 |
| yellow | validation | 45467 | 9364 | 8501 | 2934940 |
| yellow | test | 11365 | 1367 | 998 | 674398 |
| white | train | 102116 | 11654 | 18334 | 8862712 |
| white | validation | 31871 | 5414 | 13133 | 2947854 |
| white | test | 7142 | 1048 | 1153 | 678785 |

## Lane centre (near band, 256 px wide image)

| Split | Frames | Label pairs | Predicted pairs | Matched | Missed | Predicted only | MAE px | RMSE px | Median px | Max px | MAE % width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 183 | 167 | 161 | 161 | 6 | 0 | 0.767 | 2.961 | 0.250 | 31.000 | 0.300 |
| validation | 61 | 57 | 56 | 56 | 1 | 0 | 0.663 | 1.489 | 0.250 | 8.875 | 0.259 |
| test | 14 | 13 | 13 | 13 | 0 | 0 | 5.154 | 16.794 | 0.625 | 60.500 | 2.013 |

Test MAE without the single 60.5 px outlier: 0.54 px (12 images).

## Measured on all 334 images (ONNX Runtime, 2 CPU threads)
| | Yellow | White |
| --- | --- | --- |
| Parameters | 13,665 | 13,665 |
| Inference mean / p95 | 16.2 / 25.6 ms | 16.1 / 25.3 ms |
| Images with a steering command | 162/334 (48.5%) in v0.1.0, 302/334 (90.4%) from v0.2.0 | |
