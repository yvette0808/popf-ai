# 2019 MLP 基线规范

状态：`2019 Original evidence + 2026 Reconstruction plan`

本文件把论文中关于识别模型的内容拆成“明确证据”和“未知参数”。它是未来复现的约束，不代表当前已经完成训练。

## 1. 目标

目标是先复现论文明确描述的 2019 KNIME + MLP 流程，再与现代模型分开评估。任何 2026 的重建结果都必须标记为 `2026 Reconstruction`，不能写成论文的原始结果。

## 2. 论文明确写出的流程

```text
带标签图片
  -> 图像导入
  -> 100x100 灰度化
  -> 像素转浮点
  -> 0-1 归一化
  -> 计算/选择图像特征
  -> MLP Learning
  -> 计算 16 类概率
  -> 最高概率类别作为结果
```

论文还描述或展示了以下评估相关节点：

- `Partitioning`
- `Equal Size Sampling`
- `10-fold Cross Validation`
- `Learning`
- `Prediction`
- `Data Presentation`
- `SimpleMLP`

节点名称来自 Figure 2，节点内部参数和实际连接方式没有完整文字说明。[FIGURE][UNKNOWN]

## 3. 训练类别范围

论文明确说，实际用于构建识别模型的只有六类：

1. `three-tile face`
2. `fragmented pattern face`
3. `cross-shaped pattern face`
4. `pictographic face`
5. `entire face`
6. `colored three-tile face`

原因是只有这些类型的样本数量达到至少 20 张的训练基线。剩余 10 类因训练样本不足，当前识别模型不能分类。[PAPER:P4]

因此，论文同时说模型会计算 16 类概率，和后文只使用六类训练样本之间存在范围上的张力。2026 复现必须先恢复原始类别/样本分布，不能直接假设输出维度就是 6 或 16。

## 4. 数据划分证据与冲突

论文给出了两段描述：

### 4.1 4:1 描述

图像处理部分称，272 张带标签图片被划分为训练数据集和测试数据集，比例为 4:1。[PAPER:P3]

### 4.2 随机 80/20 描述

评估部分称，Partitioning node 随机使用 80% 图片作为训练集，其余作为测试集。[PAPER:P4]

这两个比例数值相同，但论文没有说清楚：

- 是否只是同一个划分的两种表达。
- 是否先做一次 4:1 外层划分，再对学习集做 80/20。
- 10-fold cross-validation 作用于全部 272 张、某个训练集，还是另一个学习集。
- 是否使用分层采样。

2026 Reconstruction 必须把划分协议显式写在实验配置中，并在没有原始 KNIME 工作流时同时保留“按单次 80/20 划分”和“按 10-fold CV”两种解释的证据记录，直到原始资料确认。

## 5. 论文报告的结果

- 论文报告识别模型在 10-fold cross-validation 后准确率约为 70%。[PAPER:P4]
- 这不是本仓库运行得到的指标。
- 论文没有提供精确准确率、每 fold 指标、混淆矩阵、分类别 precision/recall/F1、测试文件列表或评估脚本。

## 6. 论文未说明的模型细节

以下内容均为 `Unknown / To verify`：

- KNIME 版本。
- `SimpleMLP` 的具体算法实现。
- 输入特征数量和顺序。
- MLP 层数、每层宽度和偏置。
- 激活函数。
- 损失函数。
- 优化器和学习率。
- 批大小。
- 最大迭代次数。
- 正则化和早停。
- 类别权重。
- 随机种子。
- 10-fold 的分层方式和汇总方式。
- 预测概率的校准方式。
- 6 类之外的节点行为。

这些内容在复现时可以作为 2026 实验配置，但必须标记为重建设计或待补证，不得写成 2019 实现。

## 7. 复现前置条件

进入训练前，至少需要：

- 272 张带标签原始图片或可核验的等价归档。
- 完整类别名称与类别映射。
- 额外测试图片及其来源/标签，或明确的测试集边界。
- 原始 KNIME workflow（优先）或更完整的节点截图与参数。
- 原始 Feature calculator 配置或特征导出。
- KNIME 版本或运行环境线索。
- 原始结果日志、报告或可复核的评估输出。

缺少这些资料时，可以实现“论文描述的 2026 近似基线”，但必须命名为 approximation，并且不能宣称复现了 2019 原始模型。

## 8. 2026 评估记录要求

每次真正运行时，结果记录至少包括：

```text
dataset_version
raw_manifest_hash
processed_manifest_hash
label_schema_version
preprocess_config
split_protocol
cross_validation_protocol
random_seed
model_config
metrics
confusion_matrix
run_timestamp
```

当前阶段不执行这些运行，也不生成任何指标。
