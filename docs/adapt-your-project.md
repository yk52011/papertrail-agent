# 接入自己的 NLP 项目

PaperTrail 不要求项目采用某个训练框架，但 v0.2 的恢复动作限定为配置 JSON。
先在独立环境确认代码来源和依赖；不要直接在本机执行不可信仓库。

## 1. 将运行参数放入配置 JSON

例如 `config.json`：

```json
{"train_path":"data/train.json","test_path":"data/test.json","text_column":"text","min_df":1}
```

训练入口应读取该文件。无需使用本项目的分类器，可以换成你自己的模型。

## 2. 定义实验与参考标签

`experiment.json`：

```json
{
  "command": ["python", "train.py"],
  "timeout_seconds": 120,
  "artifact": "results/metrics.json",
  "ground_truth": "data/reference.json",
  "metric": "macro_f1",
  "minimum": 0.75,
  "container_image": "your-nlp-environment:local"
}
```

这是示例阈值，应根据任务预先确定，不要在见到测试结果后调整。
参考标签格式为 `[{"id":"sample-1","label":1},{"id":"sample-2","label":0}]`。
v0.2 独立核验仅支持单标签分类的 Accuracy 和 Macro F1，参考标签文件不超过 500 KB、5000 个样本。
对于其他任务，可不设置 `ground_truth`，但此时核验只检查程序报告的指标，不能声称独立重算。

预测产物须包含：

```json
{
  "accuracy": 1.0,
  "macro_f1": 1.0,
  "predictions": [{"id":"sample-1","prediction":1},{"id":"sample-2","prediction":0}]
}
```

每个参考 ID 必须恰好出现一次。预测标签类型必须与参考标签一致；当前支持整数或字符串标签。

## 3. 定义允许的恢复动作

`task.json`：

```json
{
  "title": "My NLP experiment",
  "manifest": "experiment.json",
  "config": "config.json",
  "allowed_updates": {
    "train_path": ["data/train.json"],
    "text_column": ["text"],
    "min_df": [1,2]
  }
}
```

只声明愿意自动修改的配置键和值。系统拒绝修改命令、实验清单和核验阈值。
规则策略只识别演示中的少量错误类型；不同项目优先使用模型策略，或实现专用规则适配器。

## 4. 准备项目资料与环境

README 中应写清数据、字段、环境、运行方式及恢复建议。
为 Docker 构建包含依赖的镜像，确保当前机器上已存在该镜像；任务运行时禁网，不自动安装依赖。
默认单次执行上限 120 秒，内存 512 MB，源仓库快照上限 50 MB，因此大型模型训练不在当前默认范围内。

## 5. 运行

```bash
uv run papertrail reproduce /path/to/project --policy llm --max-attempts 3
```

模型配置放在启动目录的 `.env`。无需将自己的 API key 写入项目代码或提交 GitHub。
首次可加 `--step`，逐步查看检索依据与配置修改。

JSON Schema 见仓库 `schemas/` 目录，可用于编辑器提示或外部配置验证。
