# OpenCOOD 可视化中 GT、Ego 与数据集链路分析

本文档基于当前仓库代码与本地 `v2xset/validate` 样本做静态分析，不涉及任何代码修改。目标是回答以下三个问题：

1. 为什么数据层效果图中的 GT 框看起来没有严格贴合到车辆实体上
2. 当前可视化使用的数据集是如何输入 OpenCOOD 的，如何按场景/帧精确查看效果图，以及自建数据集至少需要满足什么格式
3. 如何在效果图中识别 ego，ego 是否唯一，以及图中的 GT 框到底属于谁

## 1. 直接结论

### 1.1 你当前导出的图片来自哪个数据集

如果你运行的是：

```bash
python opencood/visualization/vis_data_sequence.py --color_mode intensity --save_path logs/early_intensity.png
python opencood/visualization/vis_data_sequence.py --color_mode z-value --save_path logs/early_z-value.png
python opencood/visualization/vis_data_sequence.py --color_mode constant --save_path logs/early_constant.png
```

那么默认使用的是：

- 脚本：`opencood/visualization/vis_data_sequence.py`
- 配置：`opencood/hypes_yaml/visualization.yaml`
- 数据目录：`v2xset/validate`

原因是 `vis_data_sequence.py` 通过 `build_dataset(params, visualize=True, train=False)` 构建数据集，而 `BaseDataset` 在 `train=False` 时读取的是 `validate_dir`，不是 `root_dir`。

因此，你这次图片并不是从 `opv2v_data_dumping/train` 导出的，而是从 `v2xset/validate` 导出的。

### 1.2 GT 框“没有贴住车”更像什么问题

主结论：这更像是 **当前可视化语义 + 点云处理方式 + 数据标注定义共同造成的视觉现象**，不像一个明显的 OpenCOOD 框架坐标变换 bug。

更具体地说：

- `color_mode` 只改变点云颜色，不改变几何位置。
- 当前脚本走的是 early fusion 视角，图中显示的是“多车点云投到单一 ego 坐标系后的叠加结果”。
- 图中的 GT 不是“ego 单车能看到的目标框”，而是“协同车辆标注目标的合并结果”。
- 代码会显式删除 ego 自车附近的点，因此 ego 自车本身最容易出现“框还在，但点很少”的现象。

所以如果你感觉某些 GT 框没有紧贴车辆实体，首先不要把它理解成“框必然投错了”，而要先确认这个框是否本来就是：

- ego 自车框
- 被其他协同车标到、但当前叠加点云采样很少的目标
- 由于 3D 视角、遮挡和点云稀疏导致看起来不够贴合的目标

## 2. 当前可视化链路是怎么工作的

## 2.1 入口脚本做了什么

`opencood/visualization/vis_data_sequence.py` 的主流程很直接：

1. 解析命令行参数
2. 读取 `visualization.yaml`
3. 根据 `fusion_method` 选择数据集类
4. 构建 dataset
5. 取一帧数据并调用 `visualize_single_sample_dataloader(...)`
6. 用 `origin_lidar`、`object_bbx_center`、`object_bbx_mask` 出图

这条链路的特征是：

- 不加载模型
- 不做前向推理
- 不产生 `pred_box_tensor`
- 图中的框来自 dataset 样本本身

因此，你当前导出的图片本质上是 **数据层效果图**，不是推理结果图。

## 2.2 当前图片里的点云和 GT 是怎么来的

在 early fusion 下，数据集会做下面几件事：

1. 读取当前全局索引对应的场景、时间戳和所有 CAV 原始数据
2. 找到当前样本的唯一 ego
3. 把每个 CAV 的点云投到 ego 坐标系
4. 把各车的 `vehicles` 标注投到 ego 坐标系
5. 合并、去重、裁剪范围
6. 用 `origin_lidar` 和 `object_bbx_center` 出图

换句话说，当前图里看到的不是“某一辆车自己的原始点云 + 它自己的 GT”，而是：

**所有参与协同的车的点云叠加到一个 ego 坐标系后，再配上联合 GT。**

## 3. 为什么 GT 框看起来没有严格贴住车辆

## 3.1 不是颜色模式问题

`intensity`、`z-value`、`constant` 三种模式只影响点云着色方式，不改变点和框的几何位置。  
所以如果你三张图里“框和点不贴合”的感觉一致，这很正常，因为几何根本没变。

## 3.2 当前图里的 GT 不是“ego 单车可见 GT”

当前 early fusion 数据层图里的 GT 来自所有协同车 `vehicles` 的并集，经过去重后统一投到唯一 ego 坐标系下。

因此，图中的某个 GT 框可能满足下面任一情况：

- ego 自己看得到
- ego 自己看不清，但其他协同车标到了
- 多车都看到了，但当前叠加后的点云对该目标只保留了局部表面

这意味着：

**GT 框附近点很少，并不自动等于框错了。**

## 3.3 ego 自车最容易看起来“不贴框”

OpenCOOD 在处理每辆车的点云时，会调用 `mask_ego_points(...)` 删除自车附近的点云区域。  
这样做的目的，是避免把传感器所在车体自身点云当成环境目标。

但 ego 自车仍然可能作为别的车视角中的一个标注目标出现在联合 GT 里。于是就会出现：

- ego 自车 GT 框还在
- ego 自车附近点云被大量挖空
- 视觉上像“框没框住车”

我对你当前默认第一帧做了实查，结果如下：

- 这帧是 `v2xset/validate/2021_08_21_17_30_41` 的 `000069`
- 参与协同的车有 3 辆：`4279`、`4288`、`4297`
- 融合后的点云有 `87886` 个点
- 最终有 `27` 个 GT 框
- 大多数框内点数正常，但 ego 自车 `4279` 对应的框里只剩 `4` 个点

所以如果你是盯着原点附近那辆车看，最容易误判成“GT 框偏了”。

## 3.4 数据标注定义也会造成视觉偏差

当前 GT 不是按“车身外观轮廓点最密集处”来定义的，而是由 YAML 中的：

- `location`
- `center`
- `extent`
- `angle`

组合得到的三维框。

这意味着：

- 框的中心是 3D 几何意义上的标注中心
- 点云只是车表面被激光扫到的离散点
- 遮挡、稀疏采样、地形起伏和可见性都会让“点云中心”和“标注框中心”看起来不完全重合

因此，轻微不贴合更应优先解释为数据表达差异，而不是先认定框架坐标有 bug。

## 4. 当前 frame_index 如何映射到数据集

## 4.1 `frame_index` 是全局索引，不是场景内局部索引

`BaseDataset` 会把所有 scenario 按顺序串起来，再把每个 scenario 的时间戳累计成全局索引。  
所以：

- `--frame_index 0` 是整个 dataset 的第一帧
- 不是某个场景内部的第 0 帧

## 4.2 当前 `v2xset/validate` 的全局索引分段

基于当前本地数据目录，索引范围如下：

| Scenario | Ego CAV | 帧数 | 全局索引范围 |
| --- | --- | ---: | --- |
| `2021_08_21_17_30_41` | `4279` | 168 | `0..167` |
| `2021_08_22_22_01_17` | `6169` | 107 | `168..274` |
| `2021_08_23_10_51_24` | `6754` | 64 | `275..338` |
| `2021_08_23_13_17_21` | `7863` | 48 | `339..386` |
| `2021_08_23_19_42_07` | `133` | 68 | `387..454` |
| `2021_09_09_22_21_11` | `15553` | 154 | `455..608` |

## 4.3 如何按你想看的场景和帧导图

如果你想看某个 scenario 的首帧：

```bash
python opencood/visualization/vis_data_sequence.py \
  --frame_index <该场景全局起点> \
  --num_frames 1 \
  --save_path logs/target_frame.png
```

如果你想看该场景内相对第 `k` 帧：

```text
全局 frame_index = 该场景起点 + k
```

例如，查看 `2021_08_23_19_42_07` 的首帧：

```bash
python opencood/visualization/vis_data_sequence.py \
  --frame_index 387 \
  --num_frames 1 \
  --save_path logs/frame_387.png
```

导出整个场景：

```bash
python opencood/visualization/vis_data_sequence.py \
  --frame_index 387 \
  --num_frames 68 \
  --save_dir logs/scene_2021_08_23_19_42_07
```

## 5. 如何在效果图中识别 ego

## 5.1 一张图里 ego 只有一个

在当前实现下，一个 sample 只会选一个 ego。  
ego 的规则不是“看起来最中心的车”，而是：

- 每个 scenario 下按 `cav_id` 排序
- 如果存在负 id，如 `-1` 的 RSU，会被移到最后
- 排序后的第一个非负 CAV 被认作 ego

所以在当前样本里：

- `4279` 是 ego
- `4288` 和 `4297` 不是 ego

## 5.2 图里不一定能看到一个完整的 ego 车体

想在图里找 ego，不要简单靠“哪辆车最完整”来判断。原因是：

- ego 自车附近点会被删除
- ego 往往位于坐标原点附近
- 原点附近反而最可能出现一个点云空洞

因此更可靠的识别方式是：

1. 看坐标原点附近
2. 找自车点云被裁掉的区域
3. 把该区域附近的 GT 框理解为 ego 相关目标，而不是期待看到完整车壳

## 6. 图中的 GT 框到底属于谁

当前 early fusion 可视化中的 GT 框，不是“所有 ego 都能看到的框”，因为一张图里只有一个 ego。

更准确地说，图里的 GT 是：

- 当前通信范围内所有 CAV 的 `vehicles`
- 合并后去重
- 统一投到唯一 ego 坐标系下
- 再做范围裁剪

它表达的是：

**协同感知样本在该 ego 参考系下的联合监督目标。**

而不是：

- ego 单车可见目标集
- 只属于某一辆非 ego 车的目标集
- 每辆车各自单独成图后的 GT

所以会出现下面这种情况：

- 某个 GT 框在图上很清楚
- 但框附近点不多
- 甚至 ego 本车看起来几乎没有完整实体

这在当前语义下是允许的，不必然代表代码错误。

## 7. 自建数据集至少要怎么组织

如果你想构建自己的数据集，最小落盘结构建议满足：

```text
dataset_root/
  scenario_x/
    cav_a/
      000001.yaml
      000001.pcd
      000001_camera0.png
      000001_camera1.png
      ...
    cav_b/
      000001.yaml
      000001.pcd
      ...
```

其中要注意：

- `scenario` 目录表示一个完整场景
- `cav_id` 目录名应使用数值字符串
- 如果有 RSU，可使用负 id，例如 `-1`
- 同一 scenario 下，各 CAV 应共享同一组 timestamp

每帧 YAML 至少要能提供：

- `lidar_pose`
- `vehicles`

而每个 `vehicle` 至少要包含：

- `location`
- `center`
- `extent`
- `angle`

当前框架就是依赖这些字段把世界坐标中的目标投到 ego 坐标系，并生成 GT 框。

## 8. 自建数据集的建议验收顺序

建议不要一上来就训练，先做四步最小验证：

1. 先准备一个只有单场景、少量帧的小数据子集
2. 用 `vis_data_sequence.py` 看单帧图，确认框大体落在合理位置
3. 连续导出几帧，确认目标运动和框的连续性正常
4. 再验证多 CAV 协同叠加是否正确，最后再进入训练/推理

这样最容易把“数据组织问题”和“模型问题”拆开。

## 9. 本次分析对应的关键判断

针对你当前的问题，可以把最终回答压缩成三句：

1. 你现在看到的 GT 框偏差，更像是 early fusion 联合 GT、ego 自车点云裁剪、点云稀疏与标注定义共同造成的视觉现象，不像一个明显的框架投影 bug。
2. 你这次导图默认读取的是 `v2xset/validate`，`frame_index` 是全局索引；想看某个特定场景，应该先把场景映射到全局索引范围，再设置 `--frame_index` 和 `--num_frames`。
3. 一张图里只有一个 ego；图中的 GT 是所有协同车标注目标在该 ego 参考系下的联合结果，不是“所有 ego 都能看到的框”。

## 10. 推荐的继续排查方式

如果你还想更细地理解某一张图，建议继续做两件事：

### 10.1 固定同一场景，连续导出多帧

这样你可以区分：

- 单帧视觉错觉
- 还是某个目标在时间上一直偏

### 10.2 指定某个 object id 做逐项核对

例如挑一个你怀疑“不贴车”的目标：

1. 在对应 YAML 里找到它的 `location/center/extent/angle`
2. 对照图中框的相对位置
3. 再看该目标附近的点数是否本来就很少

这样通常可以把“标注定义造成的偏差”和“真正的投影错误”区分开。
