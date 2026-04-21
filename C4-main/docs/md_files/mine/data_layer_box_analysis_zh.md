# 数据层效果图中的框来源分析

本文档只做代码框架分析，不涉及任何代码修改。目标是回答两个问题：

1. 为什么“数据层效果图”里也会出现框
2. 点云、预测框、模型、融合策略、GT 在 OpenCOOD 中分别是什么，以及它们如何互相影响

## 1. 直接结论

如果你运行的是 `opencood/visualization/vis_data_sequence.py`，那么图里的框通常不是预测框，而是 **GT 框**。

原因很直接：

- `vis_data_sequence.py` 在导出数据层效果图时，调用的是 `visualize_single_sample_dataloader(...)`，只读取样本中的 `origin_lidar`、`object_bbx_center` 和 `object_bbx_mask` 来画图，没有模型前向，也没有 `pred_box_tensor`。见 `opencood/visualization/vis_data_sequence.py:97-134` 和 `opencood/visualization/vis_utils.py:490-570`。
- 真正会同时画“预测框 + GT”的函数是 `visualize_single_sample_output_gt(...)`。它接收 `pred_tensor` 和 `gt_tensor`，其中预测框默认是红色，GT 默认是绿色。见 `opencood/visualization/vis_utils.py:349-408`。
- 这条“预测框 + GT”链路只在 `opencood/tools/inference.py` 中被调用。见 `opencood/tools/inference.py:199-225`。

因此，**数据层图里有框，不代表模型已经做了预测；更常见的含义是：这帧数据自带标注，脚本把 GT 一起画出来了。**

## 2. 两条可视化链路的区别

### 2.1 数据层效果图

入口：

- `opencood/visualization/vis_data_sequence.py`

主链路：

1. 读取 `visualization.yaml`
2. 根据 `fusion_method` 构建 dataset
3. 取一帧样本，得到 `sample_batched['ego']`
4. 调用 `visualize_single_sample_dataloader(...)`
5. 画出点云和 `object_bbx_center/object_bbx_mask` 对应的框

这一链路的特点是：

- 不加载模型
- 不做推理
- 不生成 `pred_box_tensor`
- 画出来的框来自 dataset 样本本身

### 2.2 推理层效果图

入口：

- `opencood/tools/inference.py`

主链路：

1. 从 `model_dir/config.yaml` 读取配置
2. 构建 dataset 和 model
3. 调用 `inference_late_fusion(...)` / `inference_early_fusion(...)` / `inference_intermediate_fusion(...)`
4. 在 dataset 的 `post_process(...)` 中得到 `pred_box_tensor` 和 `gt_box_tensor`
5. 调用 `visualize_result(...)` 或 `visualize_single_sample_output_gt(...)`

这一链路的特点是：

- 有模型前向
- 有后处理
- 有预测框
- 通常会把预测框和 GT 一起画出来

## 3. 为什么数据层图中会有框

从 dataset 角度看，OpenCOOD 在样本构建阶段就会把标注目标组织成框。

以 `EarlyFusionDataset` 为例，流程如下：

1. 把各个协同车辆的点云投影到 ego 坐标系
2. 把对应目标框整理成 `object_bbx_center`
3. 用 `mask` 标记有效框
4. 经过增强、裁剪和去重后，连同点云一起放进 `processed_data_dict['ego']`

见 `opencood/data_utils/datasets/early_fusion_dataset.py:72-145`。

也就是说，在“数据层”阶段，样本本身已经包含：

- `origin_lidar`：用于显示的点云
- `object_bbx_center`：目标框参数
- `object_bbx_mask`：哪些框有效

而 `visualize_single_sample_dataloader(...)` 正是直接取这三个字段来画图：

- 点云来自 `batch_data['origin_lidar']`
- 框来自 `batch_data['object_bbx_center']`
- 有效性来自 `batch_data['object_bbx_mask']`

见 `opencood/visualization/vis_utils.py:533-556`。

因此，**数据层图中出现框，是因为 dataset 样本里本来就包含 GT 标注框。**

## 4. GT、预测框、模型三者的代码关系

下面这三类变量最容易混淆。

| 名称 | 来自哪里 | 语义 | 出现在哪条链路 |
| --- | --- | --- | --- |
| `object_bbx_center` | dataset 样本构建阶段 | 标注框的中心参数表示，属于 GT 的中间形式 | 数据层、训练、推理 |
| `gt_box_tensor` | `post_processor.generate_gt_bbx(data_dict)` | 转成角点后的 GT 框，用于评估和展示 | 推理层 |
| `pred_box_tensor` | `post_processor.post_process(data_dict, output_dict)` | 模型输出经后处理得到的预测框 | 推理层 |

三种主流 dataset 都遵循同样的推理后处理接口：

- `EarlyFusionDataset.post_process(...)` 返回 `pred_box_tensor, pred_score, gt_box_tensor`，见 `opencood/data_utils/datasets/early_fusion_dataset.py:280-284`
- `LateFusionDataset.post_process(...)` 返回同样三项，见 `opencood/data_utils/datasets/late_fusion_dataset.py:269-273`
- `IntermediateFusionDataset.post_process(...)` 返回同样三项，见 `opencood/data_utils/datasets/intermediate_fusion_dataset.py:406-410`

GT 的生成逻辑本身也能从后处理器看清楚：`generate_gt_bbx(...)` 会把各车的 `object_bbx_center` 投影到 ego 坐标系、去重并合成 GT 框。见 `opencood/data_utils/post_processor/base_postprocessor.py:63-89`。

所以可以把关系理解成：

- `object_bbx_center` 是数据集中的 GT 描述
- `gt_box_tensor` 是更适合评估/可视化的 GT 角点框
- `pred_box_tensor` 才是模型真正“预测出来”的结果

## 5. 五个核心概念的工作原理

### 5.1 点云

点云是激光雷达采样得到的三维点集合，每个点通常带有坐标和反射强度。  
在这个仓库里，点云既是可视化对象，也是模型输入源。

在协同感知场景中，不同车辆各自采集点云，然后通过变换矩阵投影到 ego 坐标系，形成统一空间下的表示。这个过程会受到融合策略影响。

它影响：

- 模型能“看到”多大范围的环境
- 目标是否足够密集、清晰
- 最终预测框的质量

### 5.2 GT

GT 是 Ground Truth，表示数据集中的真实标注结果。  
在 OpenCOOD 中，GT 不是只在评估阶段出现，而是在 dataset 构建阶段就进入样本中。

GT 的作用有两层：

- 训练时作为监督信号，告诉模型“正确答案是什么”
- 推理评估时作为对照，衡量预测框是否准确

### 5.3 模型

模型负责把输入表示映射成目标检测结果。  
它并不直接输出最终可视化的框，而是先输出中间检测结果，再经过 `post_process(...)` 转成 `pred_box_tensor`。

所以在代码层面，只有发生了下面这件事，才可以说“出现了预测框”：

1. 执行模型前向
2. 得到 `output_dict`
3. 调用 `post_process(...)`
4. 生成 `pred_box_tensor`

如果这条链没有发生，那么图上的框就不是预测框。

### 5.4 融合策略

融合策略定义的是：多车信息在协同感知流程的哪个阶段融合。

- `early`：较早融合，通常在输入或输入前处理阶段就把多车点云/表示合起来
- `intermediate`：在中间特征阶段融合
- `late`：每车先独立处理，最后在结果层融合

在本仓库中，融合策略会同时影响：

- dataset 如何组织样本
- 点云如何投影和堆叠
- 模型前向走哪条分支
- 后处理怎样解释模型输出

它不会改变“GT 是 GT”这件事，但会改变模型如何利用点云去逼近 GT。

### 5.5 预测框

预测框是模型输出经后处理得到的检测结果。  
它往往伴随分数、NMS、阈值筛选等步骤，因此预测框是“模型判断后的结果”，而不是数据集中直接带来的内容。

在 OpenCOOD 中，预测框主要出现在推理链路中：

- `inference_utils.py` 负责调不同 fusion 的推理函数
- dataset 的 `post_process(...)` 负责把模型输出转成 `pred_box_tensor`

见 `opencood/tools/inference_utils.py:15-90`。

## 6. 它们之间如何互相影响

可以把整条链路理解成：

**点云 -> 融合策略 -> 模型 -> 预测框 -> 与 GT 对比**

更具体地说：

1. **点云** 决定感知输入的原始信息量
2. **融合策略** 决定多车信息在什么阶段合并
3. **模型** 根据融合后的表示做目标检测
4. **预测框** 是模型的输出结果
5. **GT** 是训练监督和评估参照

它们之间的影响关系如下：

- 点云越完整、越准确，模型越容易预测出接近 GT 的框
- 融合策略改变了模型看到的信息形式，因此也会影响预测框
- GT 不参与“生成预测框”，但决定模型训练目标，也决定评估标准
- 数据层可视化直接展示点云和 GT，因此用于检查“数据组织是否合理”
- 推理层可视化同时展示点云、预测框、GT，因此用于检查“模型表现是否合理”

## 7. 建议的排查思路

如果你以后再次遇到“图中框到底是不是预测框”的疑问，建议按下面顺序排查。

### 第一步：先看入口脚本

- 如果是 `vis_data_sequence.py`，优先判断为 GT
- 如果是 `inference.py`，再继续看是否生成了 `pred_box_tensor`

### 第二步：看调用的可视化函数

- 调到 `visualize_single_sample_dataloader(...)`：数据层，框来自 dataset
- 调到 `visualize_single_sample_output_gt(...)`：推理层，可同时包含预测框和 GT

### 第三步：看变量名

- `object_bbx_center` / `object_bbx_mask`：GT
- `gt_box_tensor`：GT
- `pred_box_tensor`：预测框

### 第四步：看是否经过模型与后处理

只要代码里没有这类步骤：

- `model(...)`
- `dataset.post_process(...)`
- `pred_box_tensor`

那这张图就不是模型预测图。

### 第五步：结合颜色判断

在当前默认实现里：

- 数据层框通过 `bbx2linset(...)` 绘制，默认颜色是绿色，见 `opencood/visualization/vis_utils.py:122-169`
- 推理层导出图中，预测框默认是红色，GT 默认是绿色，见 `opencood/visualization/vis_utils.py:401-404`

因此，如果图中只有一类绿色框，而你又运行的是数据层脚本，那么它基本就是 GT。

## 8. 最后结论

你这次遇到的问题，本质上是“数据层图里也画了框”，但这并不奇怪，因为 OpenCOOD 的 dataset 样本本来就包含 GT 框；数据层可视化正是用它来检查样本组织是否正确。

只有当代码真正走过：

- 模型前向
- 后处理
- `pred_box_tensor`

时，图里的框才应被理解为预测框。

换句话说：

- **数据层图中的框，主要是 GT**
- **推理层图中的框，才可能同时包含预测框和 GT**

把这两条链路分清后，再看图就不会混淆了。
