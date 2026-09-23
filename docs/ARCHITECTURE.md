# POPF 系统架构说明

状态：`2026 Reconstruction / Design only`

本文件描述目标系统的资料边界、数据血缘和未来模块。当前仅完成设计，尚未实现模型、服务或前端。

## 1. 架构原则

- raw 是不可修改的事实层。
- processed 是从 raw 生成的派生层，每个产物必须能回溯到输入文件和处理参数。
- 模型、识别结果、知识关联和反馈都必须带版本和来源。
- `2019 Original` 的流程和 `2026 Reconstruction` 的工程实现分别记录，不能混写。
- 缺少原始证据的地方使用 `Unknown / To verify`，而不是默认填入行业惯例。

## 2. 目标数据流

```text
                       2019 Original evidence
             +----------------+------------------+
             | paper          | unknown archives |
             | screenshots    | images / DB /    |
             | reported result| KNIME / website  |
             +--------+-------+---------+--------+
                      |                 |
                      +--------+--------+
                               v
                    [source inventory + provenance]
                               |
                               v
                      [data/raw - immutable]
                               |
                         manifest / checksums
                               |
                               v
                   [data/processed - generated]
                      /          |          \
                     v           v           v
           [2019 MLP baseline] [CNN] [evaluation sets]
                     \           |           /
                      +----------+----------+
                                 v
                         [inference service]
                                 |
                +----------------+----------------+
                v                                 v
       [knowledge association]              [user feedback]
                |                                 |
                +----------------+----------------+
                                 v
                         [Bad Case analysis]
```

图中下半部分是 `2026 Reconstruction` 的目标架构，不表示 2019 已经实现了这些工程模块。

## 3. 模块边界

### 3.1 Source inventory

职责：

- 登记每个文件的路径、类型、大小、来源、归属标签和校验和。
- 记录论文、截图、原始图片、数据库导出、KNIME 工作流和网站材料之间的证据关系。
- 输出盘点报告，不修改源文件。

当前状态：已完成最小文件盘点；仅有论文 PDF和 `.DS_Store`。

### 3.2 Raw data layer

拟定目录：

```text
data/raw/
data/raw_metadata/
data/manifests/
```

该目录尚不存在。建立后，原始图片和原始导出文件只能追加，不允许覆盖。任何去重、格式转换、重采样、裁剪和标签修复都必须在 processed 层进行。

### 3.3 Processed data layer

拟定目录：

```text
data/processed/
data/processed_manifests/
```

处理过程必须保存：

- 输入 raw 文件标识。
- 处理脚本版本。
- 处理参数。
- 输出文件标识。
- 标签映射版本。
- 处理时间。

当前不执行任何图像处理，因为 raw 数据尚未到位。

### 3.4 2019 MLP baseline

职责是复现论文中可证实的最小流程：

- 100x100 灰度输入。
- 浮点像素和 0-1 归一化。
- 论文提及的图像特征计算/选择。
- MLP 分类。
- 10-fold cross-validation 或经证据确认的评估协议。

当前状态：只完成规格记录，未训练。

### 3.5 Modern CNN

这是 `2026 Reconstruction` 新增模块。模型架构、增强、训练协议、指标和部署方式都必须在实际实现后记录，不得回写成 2019 功能。

当前状态：未开始。

### 3.6 Inference

目标是把图片上传、预处理、模型推理、类别概率和知识关联组成可审计的识别流程。

必须保留：

- 输入图片版本或哈希。
- 使用的模型版本。
- 预处理版本。
- top-k 概率。
- 是否为模型支持的类别范围。

当前状态：未实现。

### 3.7 Knowledge association

2019 论文明确提到图片与角色、剧目、剧情、颜色含义和脸谱结构等关联信息。2026 重建将把这些内容组织成带来源和审核状态的关联层。

该层不能从模型预测直接推导文化事实。知识条目必须有来源；来源缺失时显示为待补证。

当前状态：未实现，且知识资料尚未提供。

### 3.8 Feedback and Bad Case

论文明确描述了识别结果后的 `yes` / `no` 反馈入口。系统化 Bad Case 分析是 2026 重建新增的工程能力，计划分析：

- 用户反馈与模型预测的对应关系。
- 真实标签或人工复核结果。
- 错误类别、输入质量和数据来源。
- 低置信度与分布外样本。

当前状态：未实现，不能声称 2019 已有完整分析机制。

## 4. 关键接口约束

以下是 2026 的设计约束，不是 2019 的已知 schema：

| 接口 | 最低输入 | 最低输出 | 状态 |
|---|---|---|---|
| inventory -> raw | 原始文件及来源 | immutable 文件登记 | 待实现 |
| raw -> processed | raw 文件、处理配置 | processed 文件和 manifest | 待实现 |
| processed -> baseline | 标签、类别映射、划分协议 | 训练/评估结果 | 待实现 |
| model -> inference | 图片、模型版本 | 类别概率、top-k | 待实现 |
| inference -> knowledge | 已验证类别 ID | 带来源的知识条目 | 待设计 |
| inference -> feedback | 预测、用户反馈 | 反馈记录 | 待设计 |
| feedback -> bad case | 反馈和复核信息 | 可分析的错误案例 | 待设计 |

## 5. 版本与可追溯性

每个数据、模型和结果产物都应至少记录：

- `source_label`: `2019 Original` 或 `2026 Reconstruction`
- `dataset_version`
- `preprocess_version`
- `label_schema_version`
- `model_version`
- `evaluation_protocol`
- `created_at`
- `source_files`

在资料不足时，字段可以为 `unknown`，但不能用猜测值填充。
