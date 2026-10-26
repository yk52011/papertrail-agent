# 相关工作与定位

调研日期：2026-10-26。以下描述依据官方项目与论文；不构成完整文献综述。

| 工作 | 主要范围 | 与 PaperTrail 的关系 |
|---|---|---|
| [PaperCoder / Paper2Code](https://github.com/going-doer/Paper2Code) | 从论文规划、分析并生成代码仓库 | 更偏 paper-to-code；本项目第一版检查已有仓库 |
| [CORE-Bench](https://github.com/siegelz/core-bench) | 科研代码的计算复现，包含环境安装、运行与结果理解 | 与本项目设想高度重合，应作为后续评测与实现对照 |
| [ReproAgent](https://github.com/kernel-14/ReproAgent) | 以实现要求与参考证据组织规划、生成、修复 | 已涉及证据检索及修复，不能把“带证据”单独视作创新 |
| [PaperBench](https://openai.com/index/paperbench/) | 从头复现机器学习论文的评测基准 | 可学习任务分解与评分方式，本项目没有跑该基准 |

PaperTrail v0.2 的定位是工程学习项目：使用 LangChain / LangGraph，将实际执行、失败检索、受约束的配置恢复、独立指标核验和持久化检查点连接起来；通过人工故障注入测试展示可复现的控制流程。
“低预算 NLP 复现诊断”是应用范围，不是已证明的研究贡献。

实现参考官方文档：

- [LangChain 模型与工具接口](https://docs.langchain.com/oss/python/langchain/models)
- [LangGraph StateGraph、节点与条件边](https://docs.langchain.com/oss/python/langgraph/graph-api)
