# IntermediateFusionDataset 开发说明

## 1. 文档目的

本文档用于说明 OpenCOOD 中 `IntermediateFusionDataset` 的代码职责、主流程、输入输出结构，以及算法组在迁移新的 intermediate 融合算法时，通常应当修改哪些位置。

对应源码文件：

- `opencood/data_utils/datasets/intermediate_fusion_dataset.py`

适用范围：

- 以 **intermediate fusion** 为核心范式的方法
- 需要多车特征协同、但不采用 FPVRCNN 那种两阶段专用数据组织的方法

不直接适用：

- `IntermediateFusionDatasetV2`
- 明确依赖 `stage1 / stage2` 两套标签和 proposal-refine 流程的两阶段方法

## 2. 这个 Dataset 在整个链路中的作用

`IntermediateFusionDataset` 的作用不是定义融合网络本身，而是把一个场景中多车的原始数据整理成模型可以直接消费的中间融合输入。

它主要负责四件事：

1. 从底层 `BaseDataset` 取出一个场景内各个 CAV 的原始信息。
2. 以 ego 车为参考系，筛选通信范围内的协同车辆，并完成点云预处理。
3. 构造训练所需的目标框、anchor label，以及多车协同所需的先验信息。
4. 在 batch 维度上把样本重新拼接为模型前向所需的张量结构。

换句话说，算法组如果做的是新的 intermediate 融合模块，通常**模型主要改 `opencood/models/`，数据接口默认沿用本文件即可**；只有当输入字段、对齐方式、监督形式发生变化时，才需要改这里。

## 3. 类初始化做了什么

`__init__` 主要完成三件事：

### 3.1 继承 `BaseDataset`

父类负责更底层的数据读取、样本索引管理、场景组织、数据增强等通用能力。当前文件是在父类产出的基础数据之上，进一步组织 intermediate fusion 所需输入。

### 3.2 读取融合相关配置

这里有两个关键开关：

- `proj_first`
  - 默认 `True`
  - 含义：是否先把各个协同车的点云投影到 ego 坐标系，再进入预处理
  - 若设为 `False`，则表示后续更倾向于在特征层完成对齐

- `cur_ego_pose_flag`
  - 默认 `True`
  - 含义：在考虑通信延迟时，底层取数是否使用当前 ego pose

### 3.3 构建预处理器和后处理器

- `self.pre_processor = build_preprocessor(...)`
- `self.post_processor = build_postprocessor(...)`

它们分别负责：

- 点云转 voxel / BEV 等特征表达
- anchor 生成、监督标签生成、预测框后处理

因此，`IntermediateFusionDataset` 本身并不关心你使用的是 PointPillar、VoxelNet 还是其他 backbone，它只负责把数据整理成统一接口。

## 4. 单样本处理主流程：`__getitem__`

`__getitem__(idx)` 是本文件最核心的方法。它的输出是一个形如 `processed_data_dict['ego'] = {...}` 的字典，后续会进入 `collate_batch_train` / `collate_batch_test`。

### 4.1 从底层读取场景原始数据

调用：

```python
base_data_dict = self.retrieve_base_data(
    idx,
    cur_ego_pose_flag=self.cur_ego_pose_flag
)
```

`base_data_dict` 可以理解为：

- 一个场景样本
- 内含多个 CAV
- 每个 CAV 包含：
  - 当前是否为 ego
  - `lidar_np`
  - `params`
  - 目标框相关信息
  - 时间延迟信息等

### 4.2 确定 ego 车辆

代码会遍历 `base_data_dict`，找到 `cav_content['ego'] == True` 的车辆，并取出：

- `ego_id`
- `ego_lidar_pose`

同时代码假设：

- `OrderedDict` 的第一个元素必须是 ego

这说明下游很多逻辑都默认以 ego 为场景主车。

### 4.3 构造车间两两变换矩阵

调用：

```python
pairwise_t_matrix = self.get_pairwise_transformation(
    base_data_dict,
    self.max_cav
)
```

该矩阵的作用：

- 给需要显式几何对齐的融合模块提供任意两车之间的坐标变换关系
- 常见于 V2VNet、Where2comm、CoAlign 等方法

若 `proj_first=True`，则代码直接把所有 pairwise 变换视作单位阵，因为点云已经被投到 ego 坐标系，不再需要显式跨车对齐。

### 4.4 遍历所有 CAV，筛选可通信车辆

对每个 CAV，会先计算它和 ego 的平面距离：

```python
distance = sqrt((x_cav - x_ego)^2 + (y_cav - y_ego)^2)
```

如果超出：

```python
opencood.data_utils.datasets.COM_RANGE
```

则直接跳过。

这一步的意义是：

- 模拟有限通信半径
- 控制单个样本参与融合的车数
- 保证 `cav_num` 和后续 `record_len` 符合真实协同设定

### 4.5 处理单车数据：`get_item_single_car`

每个有效 CAV 都会调用：

```python
selected_cav_processed = self.get_item_single_car(
    selected_cav_base,
    ego_lidar_pose
)
```

这个函数完成单车级别的数据规整，见后文第 5 节。

### 4.6 汇总单车结果

对每个有效 CAV，会累积以下内容：

- `object_stack`
  - 该车视角下、已经转到 ego 坐标系的目标框中心
- `object_id_stack`
  - 目标 ID，用于多车去重
- `processed_features`
  - 单车点云经过预处理器后的特征
- `velocity`
  - 单车速度先验
- `time_delay`
  - 时间延迟先验
- `infra`
  - 是否为路侧单元的类型标记
- `spatial_correction_matrix`
  - 时延补偿矩阵

若开启可视化，还会累积：

- `projected_lidar_stack`

### 4.7 多车目标框去重

不同车辆可能看到同一物体，因此代码用 `object_id_stack` 去重：

```python
unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
```

然后得到统一的 `object_stack`。

注意：这里保留的是每个目标 ID 第一次出现的位置，不是更复杂的多视角融合策略。如果后续算法希望做更精细的 GT 选择逻辑，可以从这里扩展。

### 4.8 构造定长 GT 容器

为了让 batch 内样本维度一致，代码把目标框 pad 到：

```python
(max_num, 7)
```

对应输出：

- `object_bbx_center`
- `object_bbx_mask`

其中：

- 前者存目标框参数
- 后者标记哪些位置是真实框，哪些是补零

### 4.9 合并多车特征

调用：

```python
merged_feature_dict = self.merge_features_to_dict(processed_features)
```

其本质是把多个 CAV 的预处理结果拼成统一字典，例如：

```python
{
    'voxel_features': [...],
    'voxel_coords': [...],
    'voxel_num_points': [...]
}
```

后面再交给 `pre_processor.collate_batch(...)` 转成真正的 batch tensor。

### 4.10 生成 anchor 和监督标签

调用：

```python
anchor_box = self.post_processor.generate_anchor_box()
label_dict = self.post_processor.generate_label(
    gt_box_center=object_bbx_center,
    anchors=anchor_box,
    mask=mask
)
```

这里产出的是常规单阶段检测监督：

- 分类目标
- 回归目标
- 与 anchor 对齐后的正负样本标记

对多数 intermediate fusion 方法而言，这就是训练监督的主要来源。

### 4.11 补齐先验信息到 `max_cav`

为了让每个样本维度统一，代码会把：

- `velocity`
- `time_delay`
- `infra`
- `spatial_correction_matrix`

补齐到 `self.max_cav`。

这样做的原因是下游模型通常按固定最大协同车数建图，例如：

- `prior_encoding` 需要固定形状
- transformer / message passing 模块需要固定 agent 维

### 4.12 组织输出字典

最终单样本输出主要包含以下字段：

| 字段名 | 含义 |
| --- | --- |
| `object_bbx_center` | pad 后的 GT 框，形状约为 `(max_num, 7)` |
| `object_bbx_mask` | GT 框有效位 |
| `object_ids` | 去重后的目标 ID |
| `anchor_box` | anchor 定义 |
| `processed_lidar` | 多车预处理后的特征字典 |
| `label_dict` | 监督标签 |
| `cav_num` | 当前样本实际参与融合的车辆数 |
| `velocity` | 各车速度先验，已 pad |
| `time_delay` | 各车时延先验，已 pad |
| `infra` | 各车是否为路侧单元，已 pad |
| `spatial_correction_matrix` | 时延校正矩阵，已 pad |
| `pairwise_t_matrix` | 车间两两变换矩阵 |
| `origin_lidar` | 可视化时保存的原始/投影点云 |

## 5. 单车处理逻辑：`get_item_single_car`

这个函数负责把“某一辆车的原始观测”变成“可供协同融合的单车输入”。

主流程如下：

### 5.1 读取该车到 ego 的变换

```python
transformation_matrix = selected_cav_base['params']['transformation_matrix']
```

这个矩阵通常由底层数据读取阶段准备好，表示当前 CAV 到 ego 的坐标变换关系。

### 5.2 生成 ego 坐标系下的目标框

调用：

```python
self.post_processor.generate_object_center([selected_cav_base], ego_pose)
```

输出：

- `object_bbx_center`
- `object_bbx_mask`
- `object_ids`

这一步已经把监督信号统一到了 ego 坐标系。

### 5.3 点云清洗

对 `lidar_np` 依次执行：

1. `shuffle_points`
2. `mask_ego_points`
3. 若 `proj_first=True`，投影到 ego 坐标系
4. `mask_points_by_range`

这里最关键的是第 3 步：

- `proj_first=True` 时，后续网络拿到的是已经对齐到 ego 的输入
- `proj_first=False` 时，后续网络需要自己处理跨车几何关系

### 5.4 交给预处理器编码

```python
processed_lidar = self.pre_processor.preprocess(lidar_np)
```

这一步会根据配置生成：

- voxel 表达
- 或其他预处理结果

因此这里是点云原始输入和网络输入特征之间的边界。

### 5.5 生成单车输出

返回字段包括：

- `object_bbx_center`
- `object_ids`
- `projected_lidar`
- `processed_features`
- `velocity`

其中 `velocity` 会做一个简单归一化：

```python
velocity = ego_speed / 30
```

这是给下游模型提供的弱先验，不是严格物理建模。

## 6. 特征合并逻辑：`merge_features_to_dict`

这个函数很简单，但很关键。

它负责把多车 `processed_features` 合成一个大字典，规则是：

- 如果某个字段值是 `list`，则直接拼接列表
- 否则追加到同名字段列表里

这样做的结果是：

- 所有 CAV 的 voxel / point 特征会先在样本内部拼起来
- 后续 `collate_batch_train` 再处理 batch 维

它是“单车预处理结果”和“批量张量整理”之间的中间层。

## 7. 训练拼 batch：`collate_batch_train`

`__getitem__` 输出的是单样本字典，真正进入模型前还要通过 `collate_batch_train`。

它的职责是把一个 batch 内多个场景样本拼成统一张量。

### 7.1 聚合样本级字段

逐个样本收集：

- `object_bbx_center`
- `object_bbx_mask`
- `object_ids`
- `processed_lidar`
- `record_len`
- `label_dict`
- `pairwise_t_matrix`
- `velocity`
- `time_delay`
- `infra`
- `spatial_correction_matrix`

其中：

- `record_len` 表示每个样本实际有多少个有效 CAV
- 它是后续很多模型进行 agent regroup 的关键索引

### 7.2 处理多车特征

再次调用：

```python
merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
processed_lidar_torch_dict = self.pre_processor.collate_batch(
    merged_feature_dict
)
```

这一步会把样本内部已拼好的多车特征，进一步变成可直接送入网络的 torch 张量。

### 7.3 构造 `prior_encoding`

代码将：

- `velocity`
- `time_delay`
- `infra`

按最后一维堆叠，得到：

```python
prior_encoding.shape = (B, max_cav, 3)
```

该字段常见用途：

- 给 transformer / attention 模块作为附加先验
- 告知网络当前 agent 的速度、延迟、类型

### 7.4 组织训练输出

训练阶段最终常用字段如下：

| 字段名 | 说明 |
| --- | --- |
| `processed_lidar` | 已转为 torch 的输入特征 |
| `record_len` | 每个样本有效 CAV 数 |
| `label_dict` | 训练监督 |
| `object_bbx_center` | GT 框 |
| `object_bbx_mask` | GT mask |
| `prior_encoding` | 速度/时延/基础设施类型先验 |
| `spatial_correction_matrix` | 时延空间校正 |
| `pairwise_t_matrix` | 多车两两几何变换 |

### 7.5 可视化数据

如果 `visualize=True`，会对原始点云做最小下采样并转成 tensor，方便后续可视化接口使用。

## 8. 测试拼 batch：`collate_batch_test`

测试阶段要求：

```python
assert len(batch) <= 1
```

也就是默认只支持 batch size 为 1 的测试。

在训练版输出基础上，测试阶段额外加入：

- `anchor_box`
- `transformation_matrix = identity(4)`

这里的 `transformation_matrix` 表示 ego 到 ego 的单位变换，主要是为了统一后处理接口。

## 9. 后处理：`post_process`

该函数不是数据预处理，而是模型输出后的解码接口。

它会调用：

```python
self.post_processor.post_process(data_dict, output_dict)
```

得到：

- `pred_box_tensor`
- `pred_score`

同时通过：

```python
self.post_processor.generate_gt_bbx(data_dict)
```

得到：

- `gt_box_tensor`

因此它是评价和可视化阶段的重要统一出口。

## 10. 两两变换矩阵：`get_pairwise_transformation`

这个函数给显式空间对齐方法提供几何关系。

输出形状：

```python
(max_cav, max_cav, 4, 4)
```

语义是：

- 第 `i` 个 agent 到第 `j` 个 agent 的变换矩阵

逻辑分两种情况：

### 10.1 `proj_first=True`

若各车点云已经先投到 ego 坐标系，那么 pairwise 变换直接视为单位阵。

优点：

- 简化后续对齐逻辑
- 适合很多“先对齐再融合”的方法

### 10.2 `proj_first=False`

此时保留各车原始参考系，函数会根据每个 CAV 的 `transformation_matrix` 计算任意两车之间的变换：

```python
T_i_to_j = inv(T_j) @ T_i
```

这类设置更适合：

- 希望在特征层自己做 warping / sampling / graph message passing 的方法

## 11. 算法组最关心的输入输出接口

如果你们的工作是“新增一个 intermediate 融合模型”，最常用到的字段通常是：

### 11.1 模型输入侧

- `processed_lidar`
  - 点云预处理后的主输入
- `record_len`
  - 每个 batch 样本中有效 agent 数
- `pairwise_t_matrix`
  - 几何对齐关系
- `prior_encoding`
  - 速度 / 时延 / RSU 类型先验
- `spatial_correction_matrix`
  - 时延校正

### 11.2 训练监督侧

- `label_dict`
  - 分类和框回归监督
- `object_bbx_center`
  - GT 框
- `object_bbx_mask`
  - GT 有效位

## 12. 通常应该改哪里

### 12.1 只改融合网络，不改数据接口

如果你的方法只是替换融合模块，例如：

- 新的 attention 融合
- 新的图神经网络融合
- 新的消息传递策略
- 新的可学习对齐方式

通常只需要改：

- `opencood/models/`
- `opencood/models/fuse_modules/`
- 对应的 yaml 配置

`IntermediateFusionDataset` 可以直接复用。

### 12.2 需要改数据接口的典型情况

以下情况才建议修改本文件：

1. 需要新增模型输入字段
   - 例如新增置信度先验、通信质量、agent 类别 embedding 等
2. 需要改变对齐策略
   - 例如不再使用当前的 `proj_first` 逻辑
3. 需要改变监督形式
   - 例如多阶段监督、agent-specific supervision、occupancy supervision
4. 需要改变有效 CAV 筛选机制
   - 例如基于带宽、视野重叠、动态拓扑而不是纯通信半径

### 12.3 优先修改的位置

如果确实要改，优先关注下面几个函数：

- `__getitem__`
  - 改样本级字段组织方式
- `get_item_single_car`
  - 改单车点云处理和单车输出字段
- `collate_batch_train`
  - 改 batch 维拼接逻辑
- `get_pairwise_transformation`
  - 改车间几何关系定义

## 13. 当前实现的几个隐含假设

算法组开发时需要特别注意这些前提：

### 13.1 ego 必须是 `OrderedDict` 第一个元素

代码中有显式断言。如果底层数据组织方式变化，这里会直接报错。

### 13.2 超出通信半径的 CAV 被直接丢弃

这意味着：

- 某些远距离但视觉上仍有补充价值的 agent 不会参与融合
- `cav_num` 是动态的，不一定等于场景内总车数

### 13.3 GT 去重基于 `object_id`

这是一种较简单的多车 GT 合并策略。若数据源里 `object_id` 质量不稳定，可能影响标签质量。

### 13.4 `proj_first=True` 时，pairwise 变换退化为单位阵

如果你设计的方法仍想显式利用跨车相对位姿，不能直接依赖当前默认行为。

### 13.5 `prior_encoding` 只有 3 个通道

当前只编码：

- 速度
- 时延
- 是否为基础设施

如果方法对更多元信息敏感，需要从这里扩展。

## 14. 与 `IntermediateFusionDatasetV2` 的区别

为了避免算法组混淆，这里单独说明：

- `IntermediateFusionDataset`
  - 是当前仓库中大多数 intermediate fusion 方法的通用数据接口
- `IntermediateFusionDatasetV2`
  - 是给两阶段方法准备的数据接口
  - 典型特征是会构造 `stage1 / stage2` 两套标签

因此，对于“新的 intermediate 融合算法”开发基线，默认应以 `IntermediateFusionDataset` 为统一参考。

## 15. 建议的开发协作方式

为了让算法组后续修改成本最低，建议按下面方式协作：

1. 保持 `IntermediateFusionDataset` 的主输出字段不变。
2. 如果新增字段，尽量以增量方式添加，不要破坏现有字段名。
3. 新模型先复用已有 `label_dict` 和 `processed_lidar`。
4. 只有当监督目标或 agent 组织方式发生根本变化时，再新建 dataset 变体。

一个比较稳妥的策略是：

- 先在现有 `IntermediateFusionDataset` 上加最少量字段
- 等方法定型后，再判断是否值得抽出 `IntermediateFusionDatasetXXX`

## 16. 一句话总结

`IntermediateFusionDataset` 是 OpenCOOD 中 **通用 intermediate fusion 方法的数据组织主干**：它负责把多车原始观测整理成以 ego 为中心、可直接送入协同融合模型训练和推理的统一输入接口。
