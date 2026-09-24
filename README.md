# 京剧脸谱 AI 识别与知识探索

POPF（京剧脸谱数字传承）是一个面向公开展示的 AI 产品项目。用户上传一张京剧脸谱图片后，系统使用图像分类模型返回 Top-3 候选类别，并从已整理的历史元数据中关联角色、剧目、谱式和其他可用字段，帮助用户继续探索。

当前仓库是 `2026 Reconstruction` 产品原型，不是对 `2019 Original` 网站、KNIME 工作流或 MLP 的完整复现。项目文档始终区分论文和历史资料明确证明的内容、当前重建实现以及仍待补证的内容。

## 项目简介

产品流程：

```text
上传图片
  → AI 识别
  → Top-3 候选
  → 类别知识摘录
  → 相关脸谱图片
  → 用户反馈
```

Web MVP 提供：

- JPG / JPEG / PNG 图片上传、预览和拖拽上传
- 当前模型的 Top-1 和 Top-3 概率
- 基于现有历史元数据的资料摘录
- 从历史图片 ZIP member 读取的相关脸谱
- 匿名反馈写入 `artifacts/feedback/feedback.jsonl`
- `GET /health`、`POST /api/predict`、`POST /api/feedback` 等 API

## AI 能力

当前最佳模型：

```text
ResNet-18 + ImageNet pretrained weights
```

推理预处理与训练记录保持一致：

```text
RGB → Resize 224×224 → Tensor → ImageNet Normalize
```

当前模型覆盖 `modern_v1` 的 7 个实际标签：

```text
三块瓦脸
碎脸
象形脸
花三块瓦脸
整脸
十字门脸
六分脸
```

这里的 7 类是当前数据可支持的现代训练范围，不表示京剧脸谱官方分类只有 7 类。官方 16 种谱式作为后续数据建设和知识分类目标保留；当前项目没有把现有 raw 标签强行映射成 16 类，也没有声称已经复现 2019 年的 16 类模型。

## 当前模型结果

以下指标来自 `modern_v1` 的 held-out test set，属于 `2026 Reconstruction` 实验结果，不是生产环境准确率，也不是每次上传图片的即时准确率：

```text
Test Accuracy: 61.76%
Test Macro-F1: 59.11%
Test Macro Precision: 70.13%
Test Macro Recall: 54.35%
Test Top-3 Accuracy: 88.24%
```

当前限制：

- 当前模型只支持 7 类。
- `modern_v1` 包含 222 张图片，固定划分为 `train=154 / validation=34 / test=34`。
- `花三块瓦脸` 的 Test F1 当前为 `0`。
- `碎脸`、`象形脸`、`花三块瓦脸` 等类别仍存在明显混淆。
- 小样本类别的测试指标不稳定。
- Web Demo 用于产品原型验证，不代表生产级识别能力。

## 数据集

现代数据集来自已核验的历史候选资料：

- 272 张历史图片候选
- 272 条结构化元数据记录
- 256 条非空当前 `谱式` 标签
- `modern_v1` 按当前实际标签和最低样本门槛保留 7 类、222 张图片
- 三个 split 之间的 SHA-256 重复为 `0`

数据准备和证据边界见：

- [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md)
- [`docs/DATA_SPEC.md`](docs/DATA_SPEC.md)
- [`docs/LEGACY_DATASET_RECONCILIATION.md`](docs/LEGACY_DATASET_RECONCILIATION.md)
- [`data/processed/modern_dataset/README.md`](data/processed/modern_dataset/README.md)

仓库只保留 Web MVP 运行所需的一个小型历史图片 ZIP。完整 `data/raw/` 历史资料库没有作为公开发布包上传。公开发布前应确认该历史图片及元数据的再分发权利；如果没有相应权利，应改为由使用者在本地挂载或自行提供原始资产。

## 模型实验

当前已完成的现代视觉实验均为 `2026 Reconstruction`：

1. Frozen ResNet-18 baseline：冻结 backbone，仅训练分类头。
2. Layer4 fine-tuning：解冻 `layer4 + fc`。
3. Conservative fine-tuning：保持结构不变，降低学习率并使用 validation macro-F1 early stopping。

实验报告和训练曲线见 [`reports/`](reports/)：

- [`reports/resnet18_baseline_report.md`](reports/resnet18_baseline_report.md)
- [`reports/resnet18_layer4_finetune_report.md`](reports/resnet18_layer4_finetune_report.md)
- [`reports/resnet18_conservative_finetune_report.md`](reports/resnet18_conservative_finetune_report.md)
- [`reports/resnet18_finetuning_schedule_comparison.md`](reports/resnet18_finetuning_schedule_comparison.md)

当前 Web MVP 使用：

```text
artifacts/models/resnet18_conservative_finetune/best_val_macro_f1.pt
```

该 checkpoint 约 44.8MB，低于 GitHub 单文件 100MB 限制。后续如果加入更大的模型或多个权重，建议使用 Git LFS 或外部模型存储。

## 本地运行

建议使用 Python 3.11 或更高版本，并先安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

启动 FastAPI：

```bash
python3 -m uvicorn app.main:app --reload
```

浏览器访问：

```text
http://127.0.0.1:8000/
```

服务启动时会加载现有 checkpoint，不会训练模型。上传图片只在内存中完成推理；运行时反馈和日志不会写入 Git。

## Docker 运行

构建镜像：

```bash
docker build -t popf-web .
```

启动：

```bash
docker run --rm -p 8000:8000 popf-web
```

然后访问 <http://127.0.0.1:8000/>。

Docker 镜像只复制 MVP 所需的代码、processed 元数据、label mapping、一个历史图片 ZIP 和当前最佳 checkpoint。

## 线上部署

仓库包含 [`render.yaml`](render.yaml)，可用于从公开 GitHub 仓库创建 Render Docker Web Service。Render Blueprint 会：

1. 从 `main` 分支读取当前 `Dockerfile`。
2. 构建包含 Web MVP、历史图片 ZIP、processed 元数据和当前 checkpoint 的镜像。
3. 使用 `/health` 作为 HTTP health check。
4. 为服务提供一个公开的 HTTPS `onrender.com` 地址。

部署步骤：

1. 登录 [Render Dashboard](https://dashboard.render.com/)。
2. 选择 **New → Blueprint**，连接 `yvette0808/popf-ai`。
3. 选择 `main` 分支并确认 `render.yaml`。
4. 创建服务，等待镜像构建和 `/health` 通过。
5. 使用 Render 分配的公开 URL 访问 Web MVP。

当前配置使用 Render `free` 计划，服务在一段时间无访问后可能休眠，首次访问会有冷启动延迟。模型使用 CPU 推理；如果免费实例在构建或启动时出现内存不足，需要在 Render 中升级实例规格，不需要修改模型或数据集。

线上部署边界：

- 这是 `2026 Reconstruction` Web MVP，不是生产级识别系统。
- `artifacts/feedback/feedback.jsonl` 写入容器本地文件系统；免费实例重启或重新部署后，反馈记录可能丢失。
- 不要把用户原始图片、个人信息、API 密钥或其他敏感数据写入日志或反馈。
- 线上服务只支持当前 `modern_v1` 的 7 类标签。

## API

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

返回模型加载状态、模型名称、类别数量和预处理配置。

### `POST /api/predict`

```bash
curl -X POST \
  -F "file=@path/to/face.jpg" \
  http://127.0.0.1:8000/api/predict
```

返回当前模型实际计算的 Top-3 概率、Top-1 资料摘录和相关图片 URL。概率是模型置信度，不应表述为识别准确率；Top-3 概率不会被重新归一化。

### `POST /api/feedback`

前端会提交当前上传图片的匿名 SHA-256、Top-3 预测、用户是否认可 Top-1，以及修正类别（如果用户选择“不准确”）。反馈保存到：

```text
artifacts/feedback/feedback.jsonl
```

该运行时文件被 `.gitignore` 排除，不应上传真实用户身份、原始上传图片或其他个人信息。

### `GET /api/reference-image/{image_id}`

从保留的历史 ZIP member 读取相关图片并返回，不解压、不修改 raw 文件。

## 仓库结构

```text
app/                         FastAPI Web MVP、推理、知识和反馈
src/                         Dataset、ResNet-18 和训练基础模块
scripts/                     数据审计、数据集检查和历史实验脚本
data/processed/              可追溯的 processed 数据和现代 split
data/raw/                    仅保留 Web MVP 所需的小型图片 ZIP
artifacts/models/             当前公开使用的最佳 checkpoint
docs/                        项目、数据、产品和历史证据说明
reports/                     已真实运行的实验报告和图表
```

## 证据边界

- `2019 Original`：论文或历史资料明确记载、展示或实际提供的内容。
- `2026 Reconstruction`：当前仓库新增的数据处理、模型、API、Web MVP、知识关联和反馈实现。
- `Unknown / To verify`：资料不足以确认的内容，不用 2026 实现反推 2019 细节。

项目不虚构缺失图片、标签、文化知识或模型结果。新增知识内容只从当前已有元数据字段读取，不在 Web MVP 中凭常识补写文化解释。

## 公开发布前检查

- 不上传 `.env`、密钥、密码、个人账号信息或本地日志。
- 不上传完整 `data/raw/` 历史资料库。
- 不上传其他训练 checkpoint、torch cache 或运行时反馈。
- 确认保留的历史图片 ZIP、processed 元数据和模型权重具有公开再分发许可。
- 继续保持 raw 文件只读，任何后续数据处理都从 raw 生成 processed。
