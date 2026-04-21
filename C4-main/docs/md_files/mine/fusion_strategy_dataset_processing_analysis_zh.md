# OpenCOOD 不同融合策略下的数据集处理与推理链路分析

本文档结合 `opencood/data_utils/datasets` 目录下的实现，以及 `opencood/tools/inference.py`、`opencood/tools/inference_utils.py` 的推理入口，系统解释 OpenCOOD 中不同融合策略会如何影响：

1. **数据层**：一帧样本怎样从原始数据被组织成 `batch_data`
2. **推理层**：模型收到什么输入、输出如何被后处理成预测框和 GT

本文重点分析当前 CLI 直接支持的三种策略：

- `early`
- `intermediate`
- `late`

## 1. 先说结论

在 OpenCOOD 中，“融合策略”不是只影响模型结构，它会从更前面的 **dataset 组样本方式** 就开始生效。

也就是说：

- 数据层可视化时，即使没有模型参与，不同 fusion dataset 也会生成不同的 `origin_lidar`、`object_bbx_center`、`processed_lidar`
- 推理时，不同 fusion dataset 又会进一步决定模型看到的数据格式，以及 `post_process(...)` 时如何解释输出

所以同一个场景、同一个时间戳，在 `early`、`intermediate`、`late` 下看到的效果图、样本结构、模型输入都可能不同。这是框架设计本身决定的，不是偶然现象。

---

## 2. 三种融合策略共享的公共底座

三种 dataset 都继承自 `BaseDataset`，共同复用原始数据装载、场景索引、ego 选择、时延/噪声注入等逻辑。

### 2.1 `BaseDataset` 负责什么

`BaseDataset` 的核心职责有四个：

1. 初始化 `scenario_database`
2. 把全局 `idx` 映射到具体 `scenario + timestamp`
3. 读取各 CAV 的 YAML / PCD
4. 给每个 CAV 补齐和 ego 相关的变换参数

关键点如下：

- 训练集读 `root_dir`，验证/测试集读 `validate_dir`
- 每个 scenario 下，最小的非负 `cav_id` 被认为是 ego
- 如果有 `-1` 这类 RSU，会被移到最后，避免当成 ego
- `retrieve_base_data(...)` 会返回每个 CAV 的：
  - `lidar_np`
  - `params`
  - `ego`
  - `time_delay`

### 2.2 `reform_param(...)` 在融合前做了什么

`BaseDataset.reform_param(...)` 会把每个 CAV 的当前帧 / 延迟帧参数整理成一套统一字段，其中最关键的是：

- `transformation_matrix`
- `gt_transformation_matrix`
- `spatial_correction_matrix`
- `vehicles`

这里有两个特别重要的设计：

1. **点云变换和 GT 变换的参考系不一定完全相同**
2. **中间融合可以控制是“先投点云再提特征”还是“先提特征再投特征”**

这些字段就是后面三种 fusion dataset 分化的基础。

---

## 3. 数据层角度：三种 fusion dataset 如何组织样本

## 3.1 Early Fusion：先把多车原始点云都投到 ego，再统一处理

对应类：

- `opencood/data_utils/datasets/early_fusion_dataset.py`

### 3.1.1 `__getitem__()` 的核心流程

`EarlyFusionDataset.__getitem__()` 做的事情可以概括成：

1. 通过 `retrieve_base_data(idx)` 取到所有 CAV 的原始数据
2. 找到唯一 ego 及其 `ego_lidar_pose`
3. 遍历所有通信范围内 CAV
4. 调 `get_item_single_car(...)`：
   - 将该 CAV 点云投到 ego 坐标系
   - 将该 CAV 的标注目标也投到 ego 坐标系
5. 把所有 CAV 的点云直接堆叠成一个 `projected_lidar_stack`
6. 把所有 CAV 的目标框直接合并去重成一个 `object_bbx_center`
7. 再统一做：
   - 数据增强
   - 点云范围裁剪
   - GT 范围裁剪
   - 体素/BEV 预处理
   - anchor / label 生成

### 3.1.2 early fusion 的数据层语义

在 early fusion 下，dataset 输出的 `ego` 样本已经是 **联合样本**：

- `origin_lidar`：多车原始点云投到 ego 后直接拼接
- `object_bbx_center`：多车目标框投到 ego 后直接合并
- `processed_lidar`：对联合点云做预处理后的结果

所以从数据层角度看，early fusion 的核心思想是：

> **先在输入级把多车原始观测融合成一份统一点云，再往后走。**

### 3.1.3 early fusion 对可视化的影响

这意味着数据层可视化中：

- 点云是联合点云
- GT 是联合 GT
- 效果图天然带有“协同后”的空间结构

所以即使没有模型，early fusion 的“融合”已经发生了。

---

## 3.2 Intermediate Fusion：每车先提特征，再在 ego 侧融合特征

对应类：

- `opencood/data_utils/datasets/intermediate_fusion_dataset.py`

### 3.2.1 `__init__()` 的两个关键开关

中间融合多了两个会直接影响数据层组织的参数：

- `proj_first`
- `cur_ego_pose_flag`

它们的含义是：

- `proj_first=True`：先把点云投到 ego，再提特征
- `proj_first=False`：各车先在自己坐标系提特征，再把特征变换到 ego
- `cur_ego_pose_flag`：在存在时延时，是按当前 ego 位姿还是延迟 ego 位姿组织数据

所以 intermediate 的数据层并不是固定的一种形态，而是会随配置改变。

### 3.2.2 `__getitem__()` 的核心流程

`IntermediateFusionDataset.__getitem__()` 主要做：

1. 读取 `base_data_dict`
2. 找 ego
3. 计算 `pairwise_t_matrix`
4. 遍历通信范围内 CAV
5. 对每个 CAV 调 `get_item_single_car(...)`
6. 收集：
   - `processed_features`
   - `object_bbx_center`
   - `velocity`
   - `time_delay`
   - `infra`
   - `spatial_correction_matrix`
7. 将多车特征合并成 `merged_feature_dict`
8. 生成联合 GT、label、prior encoding 等

### 3.2.3 `get_item_single_car(...)` 和 early 的根本差异

`IntermediateFusionDataset.get_item_single_car(...)` 和 early 最大区别在于：

- 它的主要产物不是联合原始点云，而是 **每车自己的 processed feature**
- 点云是否先投到 ego，受 `proj_first` 控制

也就是说，中间融合的 dataset 目标是：

- 保留每个 CAV 的特征身份
- 同时给模型提供多车配对关系、时延信息、基础先验

### 3.2.4 intermediate fusion 的数据层语义

中间融合输出的 `ego` 样本中，最关键的字段包括：

- `processed_lidar`
- `record_len`
- `pairwise_t_matrix`
- `prior_encoding`
- `spatial_correction_matrix`
- `object_bbx_center`

它表示的不是“单一联合点云”，而是：

> **一组多车特征 + 多车关系矩阵 + 联合 GT。**

### 3.2.5 intermediate 对可视化的影响

虽然 intermediate 主要是特征融合，但当 `visualize=True` 时，它仍然会额外保存 `projected_lidar_stack` 到 `origin_lidar`。  
因此数据层可视化里你仍能看到联合点云，但这只是为了展示方便，不代表模型真正吃进去的是一份“联合点云 tensor”。

这也是 intermediate 和 early 最容易混淆的地方：

- **可视化层面** 看起来都可能是多车联合点云
- **模型输入层面** 一个是联合输入，一个是多车特征集合

---

## 3.3 Late Fusion：各车独立处理，到结果层才融合

对应类：

- `opencood/data_utils/datasets/late_fusion_dataset.py`

### 3.3.1 `get_item_single_car(...)` 的核心思想

late fusion 的 `get_item_single_car(...)` 与另外两种最不同：

- 点云先在 **各自 CAV 坐标系** 下处理
- GT 也先在 **各自 CAV 坐标系** 下生成
- 每个 CAV 自己完成：
  - 点云裁剪
  - 自车点移除
  - GT 生成
  - 数据增强
  - 预处理
  - label 生成

所以 late fusion 的单车样本从一开始就是“车自己的完整检测样本”。

### 3.3.2 `get_item_test(...)` 的样本结构

测试时，late fusion 不会像 early 那样先把所有车揉成一个 `ego` 样本，而是返回：

- `ego`
- `其他 cav_id`

每个 key 对应一辆车自己的样本。

区别只在于，额外给每个非 ego CAV 补上了：

- `transformation_matrix`：当前 CAV -> ego 的变换

### 3.3.3 `collate_batch_test(...)` 为什么又会出现联合点云

late fusion 在 `collate_batch_test(...)` 里做了一件很容易误解的事：

- 为了可视化方便，它把各车点云临时再投到 ego 并堆成 `output_dict['ego']['origin_lidar']`

但这只是为了画图，不代表 late fusion 在模型输入上变成了 early fusion。

late fusion 的本质依然是：

> **每辆车单独跑检测，最后在结果层融合。**

### 3.3.4 late fusion 的数据层语义

因此 late fusion 的 dataset 输出本质是：

- 每车一个独立输入样本
- 每车一个独立 GT
- 每车一个独立 `processed_lidar`
- 只是在可视化时额外拼了一份“ego 视角联合点云”

这也是为什么 late fusion 的数据层图，看起来常常和 early / intermediate 不完全一样。

---

## 4. 推理层角度：不同 fusion strategy 如何影响模型输入与后处理

## 4.1 推理入口如何切换 fusion

推理入口在：

- `opencood/tools/inference.py`

其核心流程是：

1. 读取 `model_dir/config.yaml`
2. `build_dataset(hypes, visualize=True, train=False)`
3. 构建 dataloader
4. 创建 model
5. 根据 `--fusion_method` 选择不同 inference 函数

也就是说，推理时 fusion strategy 的切换有两层：

1. **dataset 切换**
2. **inference 调度切换**

这两层缺一不可。

---

## 4.2 Early Fusion 的推理层

`inference_early_fusion(...)` 的逻辑很简单：

1. 直接取 `batch_data['ego']`
2. 执行 `model(cav_content)`
3. `dataset.post_process(batch_data, output_dict)`

这里说明：

- 模型收到的是 early dataset 组织好的联合样本
- 模型只前向一次
- GT 来自 `dataset.post_process(...)` 中的 `generate_gt_bbx(...)`

所以 early 的推理层语义是：

> **联合输入 -> 单次前向 -> 联合预测框**

---

## 4.3 Intermediate Fusion 的推理层

`inference_intermediate_fusion(...)` 当前直接复用了 `inference_early_fusion(...)`。  
这并不表示它和 early 是同一件事，而是表示：

- 模型调用接口一样，都是 `model(batch_data['ego'])`
- 真正的差异已经被 dataset 输出的 `batch_data['ego']` 结构吸收掉了

换句话说，intermediate 的不同不在 inference 函数这一层，而在：

- `processed_lidar` 的结构
- `record_len`
- `pairwise_t_matrix`
- `prior_encoding`
- `spatial_correction_matrix`

模型在前向时读取这些字段，完成中间特征融合。

所以 intermediate 的推理层语义是：

> **多车特征集合 -> 单次前向 -> 联合预测框**

只是这次“联合”发生在网络中间，而不是输入级。

---

## 4.4 Late Fusion 的推理层

`inference_late_fusion(...)` 和另外两种明显不同：

1. 遍历 `batch_data` 中每个 CAV
2. 对每个 `cav_content` 都单独执行一次 `model(cav_content)`
3. 把每车输出收集到 `output_dict`
4. 再调用 `dataset.post_process(batch_data, output_dict)`

这说明 late fusion 的推理层语义是：

> **每车独立前向 -> 每车独立结果 -> 结果层融合**

所以 late 和 early/intermediate 在推理层最大的区别不是 `post_process(...)` 接口，而是：

- 前两者：一次前向
- late：多次前向

---

## 5. 三种 fusion 在 `post_process(...)` 上有什么共同点和不同点

三种 dataset 最终都会实现自己的 `post_process(...)`，并返回：

- `pred_box_tensor`
- `pred_score`
- `gt_box_tensor`

共同点是：

- 都会调用 `self.post_processor.post_process(...)`
- 都会调用 `self.post_processor.generate_gt_bbx(data_dict)`

也就是说，**统一输出接口是一样的**。

但真正差别在于：

- `data_dict` 的结构不同
- `output_dict` 的结构不同

于是：

- early / intermediate 的 `data_dict` 更像“ego 视角联合样本”
- late 的 `data_dict` 是“多车样本字典”

因此虽然函数名相同，后处理阶段看到的上下文并不相同。

---

## 6. 从数据层到推理层，三种 fusion 的本质差异

下面把三种策略压缩成一张表：

| 策略 | 数据层怎么组织 | 模型输入是什么 | 推理时前向次数 | 结果在哪一层融合 |
| --- | --- | --- | ---: | --- |
| Early | 多车点云先投到 ego 并直接拼接 | 联合点云样本 | 1 | 输入层/输入前 |
| Intermediate | 每车先形成特征，并保留多车关系矩阵 | 多车特征集合 + 关系信息 | 1 | 中间特征层 |
| Late | 每车单独形成完整检测样本 | 单车样本，逐车输入 | 多次 | 检测结果层 |

这张表解释了两个常见现象：

### 6.1 为什么数据层图也会变

因为不同 fusion dataset 在 `__getitem__()` 阶段就已经把样本组织成不同形态了。

### 6.2 为什么推理入口里有些 fusion 看起来共用函数

因为 intermediate 和 early 的区别主要已经体现在 dataset 输出结构里，而不是一定要在 `inference_utils.py` 里再写一套完全不同的调度代码。

---

## 7. 为什么“没模型参与”的数据层可视化也会受融合策略影响

这是最容易误解的点。

数据层可视化本身只画：

- `origin_lidar`
- `object_bbx_center`
- `object_bbx_mask`

但这三个字段本身就由 fusion dataset 决定。

所以：

- `early` 画的是联合输入级样本
- `intermediate` 画的是为中间特征融合准备的样本，同时附带一份可视化点云
- `late` 画的是单车样本体系下，为了展示方便临时堆起来的 ego 视角点云

因此它们虽然都调用同一个可视化函数，但输入进去的 sample 已经不同了，输出当然也会不同。

---

## 8. 阅读代码时的推荐视角

如果你之后还想继续深挖不同 fusion 的差异，建议按下面顺序读：

1. `BaseDataset`
   - 看原始数据怎么变成 `base_data_dict`
2. 三个 fusion dataset 的 `__getitem__()`
   - 看样本在数据层怎样分化
3. 三个 fusion dataset 的 `collate_batch_test()`
   - 看 dataloader 输出长什么样
4. `inference_utils.py`
   - 看推理时模型怎么被调用
5. 各 dataset 的 `post_process(...)`
   - 看预测和 GT 怎么被统一成最终框

如果只盯模型文件，很容易误以为 fusion 只发生在网络内部；而 OpenCOOD 的真实设计是：

> **fusion strategy 从 dataset 组样本时就已经开始生效。**

---

## 9. 一句话总结

在 OpenCOOD 中：

- **数据层**决定“这一帧样本长什么样”
- **推理层**决定“模型怎么吃这帧样本、怎么把结果变回框”

而不同融合策略的真正差异，正是沿着这两层同时展开的：

- early：先融合原始输入
- intermediate：先保留多车特征，再在网络中部融合
- late：每车先独立检测，最后融合检测结果

因此，想理解 fusion strategy，不能只看模型，也必须把 `opencood/data_utils/datasets` 这层一起看进去。
