# POPF Modern Dataset Report

状态：`2026 Reconstruction / modern_v1`

本报告只描述从 272 份历史图片和历史元数据生成的现代训练前数据集。它不训练模型，不复现 KNIME。

## 1. 输入和映射边界

- 图片来源：`data/raw/脸谱/整理工作/爬虫脸谱图集.zip`
- 历史元数据来源：`data/raw/脸谱新一步/272数据张丹阳.xls`
- 候选核验依据：`data/processed/272_candidate/`
- 映射策略：`001 -> 序号 1` 到 `272 -> 序号 272`
- `mapping_status`：`user_confirmed_sequence_mapping`
- 映射证据：Project owner confirmed on 2026-09-20 that Excel 序号 and the image filename number are one-to-one for the current POPF dataset.

该映射由项目负责人确认后，在当前现代数据集构建中作为有效的一一对应关系使用。

## 2. 数据概览

| 指标 | 数量 |
|---|---:|
| 历史图片总数 | 272 |
| 可读取图片 | 272 |
| 有效非空标签记录 | 256 |
| 缺失标签记录 | 16 |
| 非空 raw_face_type 类别数 | 20 |
| modern_v1 入选图片 | 222 |
| modern_v1 入选类别 | 7 |
| modern_v1 排除图片 | 50 |

## 3. 当前实际标签分布

本阶段按当前 XLS 的 `谱式` 原始文本分析标签，不再尝试与论文标签体系对齐。

| raw_label | sample_count | percentage | sufficient_for_training | notes |
| --- | --- | --- | --- | --- |
| 三块瓦脸 | 73 | 26.84% | yes | Included in modern_v1 by the 2026 minimum sample policy using the current XLS raw label. |
| 碎脸 | 45 | 16.54% | yes | Included in modern_v1 by the 2026 minimum sample policy using the current XLS raw label. |
| 象形脸 | 35 | 12.87% | yes | Included in modern_v1 by the 2026 minimum sample policy using the current XLS raw label. |
| 花三块瓦脸 | 27 | 9.93% | yes | Included in modern_v1 by the 2026 minimum sample policy using the current XLS raw label. |
| 整脸 | 17 | 6.25% | yes | Included in modern_v1 by the 2026 minimum sample policy using the current XLS raw label. |
| (missing) | 16 | 5.88% | no | Missing 谱式 in source metadata; excluded from modern_v1. |
| 十字门脸 | 15 | 5.51% | yes | Included in modern_v1 by the 2026 minimum sample policy using the current XLS raw label. |
| 六分脸 | 10 | 3.68% | yes | Included in modern_v1 by the 2026 minimum sample policy using the current XLS raw label. |
| 僧道脸 | 8 | 2.94% | no | Excluded from modern_v1 because sample_count < 10; preserved in modern_all_candidates.csv. |
| 元宝脸 | 6 | 2.21% | no | Excluded from modern_v1 because sample_count < 10; preserved in modern_all_candidates.csv. |
| 歪脸 | 5 | 1.84% | no | Excluded from modern_v1 because sample_count < 10; preserved in modern_all_candidates.csv. |
| 神仙脸 | 5 | 1.84% | no | Excluded from modern_v1 because sample_count < 10; preserved in modern_all_candidates.csv. |
| 太监脸 | 2 | 0.74% | no | Excluded from modern_v1 because sample_count < 10; preserved in modern_all_candidates.csv. |
| 元宝类 | 1 | 0.37% | no | Excluded from modern_v1 because sample_count < 10; preserved in modern_all_candidates.csv. |
| 小妖脸 | 1 | 0.37% | no | Excluded from modern_v1 because sample_count < 10; preserved in modern_all_candidates.csv. |
| 枣核脸 | 1 | 0.37% | no | Excluded from modern_v1 because sample_count < 10; preserved in modern_all_candidates.csv. |
| 白粉脸 | 1 | 0.37% | no | Excluded from modern_v1 because sample_count < 10; preserved in modern_all_candidates.csv. |
| 窝头脸 | 1 | 0.37% | no | Excluded from modern_v1 because sample_count < 10; preserved in modern_all_candidates.csv. |
| 筝形脸 | 1 | 0.37% | no | Excluded from modern_v1 because sample_count < 10; preserved in modern_all_candidates.csv. |
| 腰子脸 | 1 | 0.37% | no | Excluded from modern_v1 because sample_count < 10; preserved in modern_all_candidates.csv. |
| 角筝形脸 | 1 | 0.37% | no | Excluded from modern_v1 because sample_count < 10; preserved in modern_all_candidates.csv. |

## 4. modern_v1 训练类别

本阶段采用的 2026 Reconstruction 策略是：只保留非空 `raw_face_type`，并要求每类至少 10 张图片。该门槛服务于现代模型的基础 train/validation/test 划分。

| raw_label | total | train | validation | test |
| --- | --- | --- | --- | --- |
| 三块瓦脸 | 73 | 51 | 11 | 11 |
| 碎脸 | 45 | 31 | 7 | 7 |
| 象形脸 | 35 | 25 | 5 | 5 |
| 花三块瓦脸 | 27 | 19 | 4 | 4 |
| 整脸 | 17 | 11 | 3 | 3 |
| 十字门脸 | 15 | 11 | 2 | 2 |
| 六分脸 | 10 | 6 | 2 | 2 |

## 5. 暂时排除的类别和原因

| raw_label | sample_count | reason |
| --- | --- | --- |
| (missing) | 16 | missing_raw_face_type |
| 僧道脸 | 8 | class_below_min_samples_per_class_10 |
| 元宝脸 | 6 | class_below_min_samples_per_class_10 |
| 歪脸 | 5 | class_below_min_samples_per_class_10 |
| 神仙脸 | 5 | class_below_min_samples_per_class_10 |
| 太监脸 | 2 | class_below_min_samples_per_class_10 |
| 元宝类 | 1 | class_below_min_samples_per_class_10 |
| 小妖脸 | 1 | class_below_min_samples_per_class_10 |
| 枣核脸 | 1 | class_below_min_samples_per_class_10 |
| 白粉脸 | 1 | class_below_min_samples_per_class_10 |
| 窝头脸 | 1 | class_below_min_samples_per_class_10 |
| 筝形脸 | 1 | class_below_min_samples_per_class_10 |
| 腰子脸 | 1 | class_below_min_samples_per_class_10 |
| 角筝形脸 | 1 | class_below_min_samples_per_class_10 |

排除原因汇总：

| reason | records |
| --- | --- |
| class_below_min_samples_per_class_10 | 34 |
| missing_raw_face_type | 16 |

## 6. 数据划分

| split | records |
|---|---:|
| train | 154 |
| validation | 34 |
| test | 34 |

划分配置：

- random seed：`42`
- 目标比例：70% train, 15% validation, 15% test
- 实际比例：train 69.37%, validation 15.32%, test 15.32%
- 方法：按 `raw_face_type` 分层，每类独立排序、固定 seed 洗牌，再按四舍五入后的 15%/15% 分配 validation/test，剩余为 train。

## 7. 数据质量检查

| 检查项 | 结果 |
|---|---|
| 图片是否全部可读取 | 272/272 readable |
| ZIP 内 SHA-256 是否唯一 | yes |
| 重复 SHA-256 group 数 | 0 |
| 涉及重复 SHA-256 的图片数 | 0 |
| train/validation/test 是否存在相同 SHA-256 跨 split | no |
| 标签缺失 | 16 records |
| 类别极度不平衡 | yes, largest class 73 and smallest nonempty class 1 |

图片尺寸分布：

| width_height | count |
| --- | --- |
| 123x150 | 163 |
| 90x110 | 42 |
| 89x110 | 22 |
| 88x110 | 16 |
| 91x110 | 8 |
| 87x110 | 5 |
| 86x110 | 4 |
| 92x110 | 3 |
| 123x152 | 2 |
| 123x149 | 1 |
| 94x110 | 1 |
| 85x110 | 1 |
| 93x110 | 1 |
| 84x110 | 1 |
| 83x110 | 1 |
| 123x151 | 1 |

## 8. 泄漏检查

未发现相同 SHA-256 跨 train/validation/test。当前 272 张候选图内 SHA-256 全部唯一。

## 9. 标签问题

- 16 条记录缺少 `谱式`，不能进入第一版监督训练。
- 非空 `谱式` 有 20 种文本值，本阶段按照当前实际标签处理，不与论文标签体系做对应分析。
- 部分类别样本很少，1 到 8 张的类别暂不进入 modern_v1，但全部保留在 `modern_all_candidates.csv`。

## 10. 是否足够进入现代视觉模型训练

可以进入探索性现代视觉模型训练，数据集状态为 `2026 Reconstruction / user_confirmed_sequence_mapping`。主要限制是样本量小、类别不平衡，以及部分当前实际标签的样本数过低。

## 11. 下一步建议

1. 固化一个 `legacy_dataset` 和 `modern_dataset` 分离的目录规范，避免现代训练选择反向污染 2019 复现。
2. 编写只读取 ZIP 成员的 PyTorch Dataset/DataLoader，保持 `image_path` 的 archive-member 语义。
3. 第一版模型建议使用 ImageNet 预训练的轻量 CNN，例如 ResNet-18 或 MobileNetV3，并先冻结大部分 backbone。数据只有 222 张，直接从零训练不稳。
4. 训练前再次人工复核 modern_v1 的当前实际标签，尤其是低样本类别和容易混淆的谱式。
