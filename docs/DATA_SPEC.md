# POPF 数据规范

状态：`2026 Reconstruction / Inventory and contract draft`

本文件记录论文中可以确认的数据事实，并定义后续 raw/processed 数据工程的约束。它不创建数据，也不把缺失资料补成假数据。

## 1. 当前数据状态

当前仓库没有 `data/` 目录，没有原始 JPG/PNG，没有标签表，没有数据 manifest，也没有数据库导出。唯一研究资料是 `docs/2019_paper.pdf`。

因此以下内容是：

- `2019 Original`：论文明确描述的历史数据事实。
- `2026 Reconstruction`：后续要执行的目录和校验规则。
- `Unknown / To verify`：论文没有提供的字段、文件或细节。

## 2. 2019 Original 数据事实

### 2.1 数量和来源

| 项目 | 论文描述 | 状态 |
|---|---|---|
| 带标签图片 | 272 张 | `2019 Original` |
| 训练数据类别数 | 16 类 | `2019 Original` |
| 额外测试图片 | 约 200 张，来自各种搜索引擎 | `2019 Original` |
| 总体规模 | 约 500 张，包含 272 张带标签图片 | `2019 Original` |
| 主要来源 | `http://www.paulnoll.com` | `2019 Original` |
| 每类样本数 | 未提供 | `Unknown` |
| 图片文件名和 URL | 未提供 | `Unknown` |
| 2019 原始图片包 | 当前仓库没有 | `Unknown / To verify` |

注意：论文先称 272 张为 frontal images，后面又写数据集包含 frontal 和 side face types。这个描述冲突必须原样保留，不能自行判断哪一句正确。

### 2.2 论文明确出现的字段

论文 Table 1 的典型记录包含：

| 字段 | 证据 | 备注 |
|---|---|---|
| `Role` | Table 1 | 角色名称 |
| `Repertoire` | Table 1 | 剧目 |
| `Type of POPF` | Table 1 | 脸谱类型 |
| `Introduction` | Table 1 | 介绍文本 |

网站文字还提到：

- 角色介绍。
- 出自哪部戏及剧情。
- 使用的颜色及其含义。
- 脸谱结构。

这些信息是否全部是独立数据库字段，论文未说明。MySQL schema、字段类型、关系和图片存储方式均未提供。

### 2.3 16 类分类体系

论文只明确说存在 16 类，并没有给出完整 16 类清单。

正文明确出现的分类/示例术语：

| 术语 | 出现语境 | 是否确认用于 MLP |
|---|---|---|
| `entire face` | 模块页示例 | 是 |
| `three-tile face` | 模块页示例、模型类别 | 是 |
| `colored three-tile face` | 模块页示例、模型类别 | 是 |
| `six-tenths face` | 模块页示例 | 未说明 |
| `fragmented pattern face` | 模型类别 | 是 |
| `cross-shaped pattern face` | 模型类别 | 是 |
| `pictographic face` | 模型类别 | 是 |
| `eunuch face` | 网站示例 | 未说明 |
| `goblin face` | 上传预测示例 | 未说明 |

论文明确说明实际构建识别模型的只有六类：

```text
three-tile face
fragmented pattern face
cross-shaped pattern face
pictographic face
entire face
colored three-tile face
```

因为只有这些类型达到至少 20 张训练样本的基线，另外 10 类无法被当前模型分类。[PAPER:P4]

剩余 10 类的正式名称、类别 ID、中文名、别名和类间关系均为 `Unknown`。不能用其他资料中的常见 16 类列表替代它们。

## 3. 2019 图像处理事实

论文明确描述的处理步骤：

1. 输入包含 JPG 和 PNG。
2. 原始图片尺寸不一致。
3. 论文描述数据包含正面和侧面脸谱，但前文又把 272 张称为 frontal images。
4. 图像转换为 `100 x 100` 灰度图。
5. 像素转为浮点。
6. 像素归一化到 `[0, 1]`。
7. 计算像素值的算术值和像素均值。
8. 在 KNIME Feature calculator 中选择图像特征。

论文未说明：

- 是否先裁剪脸谱区域。
- 缩放插值方法。
- 是否保持纵横比。
- 灰度转换公式。
- 100x100 图像如何变成特征向量。
- “mean”是单个全局均值还是其他统计特征。
- 训练/测试图片是否经过同一套处理。
- 文件损坏、重复或缺失时如何处理。

## 4. 2026 Reconstruction 数据目录

以下目录是拟定结构，不代表当前已经创建或存在数据：

```text
data/
  raw/                  # 原始文件，只读，不改写
  raw_metadata/         # 原始来源、许可、人工备注
  manifests/            # raw 文件清单、大小、hash、来源
  processed/            # 从 raw 派生的图像和标签
  processed_manifests/  # 处理参数和输入输出血缘
```

`raw/` 中的文件必须保持原样。任何重命名、转码、裁剪、灰度化、缩放、去重和标签映射都生成新文件到 `processed/`，并保留输入文件标识。

## 5. 建议的最小 manifest 字段

这些字段是 2026 的数据工程建议，不是 2019 schema：

| 字段 | 用途 | 初始值 |
|---|---|---|
| `record_id` | 稳定记录标识 | 待生成 |
| `raw_path` | raw 文件路径 | 待登记 |
| `raw_sha256` | 原始文件完整性 | 待计算 |
| `source_url` | 原始来源 URL | 待核验 |
| `source_label` | 2019 或 2026 来源标签 | 待核验 |
| `role` | 角色 | 待核验 |
| `repertoire` | 剧目 | 待核验 |
| `type_of_popf` | 类别文本 | 待核验 |
| `class_id` | 稳定类别 ID | 待建立 |
| `split` | train/test/unknown | 待核验 |
| `license_status` | 使用权限状态 | 待核验 |
| `notes` | 人工核验备注 | 可选 |

任何无法从原始资料确认的字段应为空或 `unknown`，不能批量猜填。

## 6. 版权和来源

论文说部分数据来自 `paulnoll.com`，并称网站所有者允许在非商业情况下免费使用材料。[PAPER:P6]

这不是对当前 2026 重建数据权利的自动授权。原始图片恢复后仍需核验：

- 每个文件的真实来源。
- 当前许可是否仍然适用。
- 是否允许重新分发、训练模型和部署网站。
- 用户上传图片的隐私与保留策略。

## 7. 数据质量门槛

在进入模型前，必须能够回答：

- 每张图片对应哪个 raw 文件和 hash。
- 标签来自哪里，是否人工复核。
- 16 类完整清单是否已经被证据确认。
- 272 张与约 200 张测试图片如何区分。
- 是否存在重复或同源近重复图片。
- 2019 划分协议到底如何解释。

这些问题没有答案时，只能继续做资料补全，不能训练。
