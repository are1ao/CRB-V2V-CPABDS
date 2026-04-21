# 可视化效果图导出教程

本文档用于快速生成不同设置下的效果图，覆盖两类场景：

1. **数据层效果图**：点云 + GT 框
2. **推理层效果图**：点云 + 预测框 + GT 框

以下命令默认使用你的环境 `torch118`。如果环境名不同，请自行替换。

补充说明：

- 本文中的导出路径会自动创建，无需手动 `mkdir`。
- 在当前这类无图形界面的终端/服务器环境下，脚本会自动回退到可保存图片的备用导出方式；需要 Open3D 的 3D 视角时，请在带桌面的环境执行。
- `inference.py` 相关命令依赖模型权重；本文先保留用法，不在本次可执行性检查范围内。
- 若你在只读 home 目录的环境里看到 `matplotlib` 或 `mesa` 的缓存告警，这不影响出图；若想安静一些，可先执行 `mkdir -p /tmp/matplotlib && export MPLCONFIGDIR=/tmp/matplotlib`。

## 1. 控制点云着色

可选值：

- `constant`：统一颜色
- `intensity`：按反射强度着色
- `z-value`：按高度着色

示例：

```bash
conda run -n torch118 python opencood/visualization/vis_data_sequence.py \
  --fusion_method early \
  --color_mode intensity \
  --save_path logs/early_intensity.png
```

## 2. 控制融合策略(对于数据的效果图展示没有影响，它影响的是模型的融合策略)

可选值：

- `early`
- `late`
- `intermediate`

示例：

```bash
conda run -n torch118 python opencood/visualization/vis_data_sequence.py \
  --fusion_method late \
  --color_mode constant \
  --save_path logs/late_constant.png
```

## 3. 控制分辨率与细节

可调参数：

- `--width`
- `--height`
- `--point_size`
- `--background {dark,black,white}`

示例：

```bash
conda run -n torch118 python opencood/visualization/vis_data_sequence.py \
  --fusion_method early \
  --color_mode intensity \
  --width 2560 \
  --height 1440 \
  --point_size 1.5 \
  --background black \
  --save_path logs/early_intensity_4k.png
```

## 4. 指定某一帧或最后一帧

- `--frame_index 10`：第 10 帧
- `--frame_index -1`：最后一帧

示例：

```bash
conda run -n torch118 python opencood/visualization/vis_data_sequence.py \
  --fusion_method early \
  --color_mode constant \
  --frame_index -1 \
  --save_path logs/early_endFrame.png
```

## 5. 批量导出多帧

从某一帧开始连续导出：

```bash
conda run -n torch118 python opencood/visualization/vis_data_sequence.py \
  --fusion_method early \
  --color_mode intensity \
  --frame_index 20 \
  --num_frames 5 \
  --save_dir logs/early_batch
```

如果要从最后一段开始导出，建议先确认总帧数，再手动设置起始索引。

## 6. 导出推理结果效果图

需要提供训练好的 `model_dir`。

导出单帧：

```bash
conda run -n torch118 python opencood/tools/inference.py \
  --model_dir opencood/logs/<你的模型目录> \
  --fusion_method intermediate \
  --color_mode intensity \
  --frame_index 30 \
  --save_vis_dir logs/infer_one
```

导出最后一帧：

```bash
conda run -n torch118 python opencood/tools/inference.py \
  --model_dir opencood/logs/<你的模型目录> \
  --fusion_method intermediate \
  --color_mode z-value \
  --frame_index -1 \
  --save_vis_dir logs/infer_last
```

批量导出：

```bash
conda run -n torch118 python opencood/tools/inference.py \
  --model_dir opencood/logs/<你的模型目录> \
  --fusion_method early \
  --color_mode constant \
  --frame_index 50 \
  --num_frames 10 \
  --width 1920 \
  --height 1080 \
  --point_size 1.2 \
  --save_vis_dir logs/infer_batch
```

只看预测框或只看 GT：

```bash
conda run -n torch118 python opencood/tools/inference.py \
  --model_dir opencood/logs/<你的模型目录> \
  --fusion_method intermediate \
  --frame_index 30 \
  --pred_only \
  --save_vis_dir logs/pred_only
```

```bash
conda run -n torch118 python opencood/tools/inference.py \
  --model_dir opencood/logs/<你的模型目录> \
  --fusion_method intermediate \
  --frame_index 30 \
  --gt_only \
  --save_vis_dir logs/gt_only
```

## 7. 快速对比建议

如果你想快速测试不同想法，建议固定同一帧，然后只改一个变量：

1. 固定 `--frame_index`
2. 只改 `--color_mode`
3. 再只改 `--fusion_method`
4. 最后再调 `--width --height --point_size`

这样最容易看出不同设置对效果图的真实影响。
