# 加性异质性共生报告

## 研究问题

单情境基准回答的是：能否把一个焦点企业特征对应的 CATE 关系恢复为准确的函数
形态和关键量。本模块进一步追问：当多个企业特征共同产生处理效应异质性时，能否
在同一个数据生成过程中分别重构并确认它们的加性贡献？

本模块是单情境流程的扩展，而不是另一种因果森林估计器。每个 replication 仍只用
全部 12 个候选企业特征训练一棵 `CausalForestDML`。森林估计联合 CATE；随后只在
TRAIN 中对预先指定的活动成分做加性重构，untouched TEST 则在同时控制其他成分后
确认已经固定的成分。

## 方法贡献

该模块把现有流程向前推进了三步：

1. 从恢复一条焦点 CATE 曲线，扩展到剥离同一数据生成过程中共存的两条或三条曲线；
2. 每个成分仍执行完整的单情境形态和关键量标准，不以总体 CATE 拟合良好替代逐成分
   准确性；
3. 增加匹配的全 null 实验，只要任一候选成分被假确认，就记录为一次 family-wise
   错误。

两组件和三组件结果分开保存。它们共用代码和阈值，但使用独立配置、报告、图片、
表格和证据 manifest。

## 冻结设计与验收

- 真实 Compustat 派生 L=3 历史提供企业特征的联合分布；处理变量和因果结果由程序
  模拟；
- 每次 replication 使用 `n=3000`、一次 80/20 TRAIN/untouched-TEST 划分和一棵
  1,000 棵树的 honest 多变量 `CausalForestDML`；
- 活动成分集合为本次能力实验预先指定；TEST 结果和 oracle 真值不参与变量、形态或
  参数选择；
- 两个实验分别运行 20 次 fresh active replications 和 20 次匹配全 null
  replications；
- 一个成分只有同时通过实践证据、准确形态、曲线 Spearman 不低于 `.95`、标准化
  RMSE 不高于 `.20` 以及全部预设关键量，才记为成功；
- 每个成分成功率都不低于 90%、总体 CATE fidelity 不低于 90%、全 null 下任一假
  确认率不高于 10%，实验才通过。

## 两组件确认

两个成分分别是固定资产强度的倒 U 效应和无形资产强度的 U 型效应。

| 成分 | 形态 | 完整恢复率 |
|---|---|---:|
| 固定资产强度 | 倒 U | 18/20 = 90% |
| 无形资产强度 | U 型 | 20/20 = 100% |

总体 CATE fidelity 为 100%，总体 CATE Spearman 中位数为 `.997`，标准化 RMSE
中位数为 `.025`，全 null 下任一假确认率为 0%。

![两组件恢复与全 null 控制](two_component/evidence/FIGURE_1_RECOVERY_AND_NULL_CONTROL.svg)

这说明在该加性设计中，两条组成关系共存时，函数形态和要求的关键量仍可分别恢复；
总体 CATE 和全 null 检验也同时通过，因此逐成分结果并非以牺牲总体准确性或错误
控制为代价。

## 三组件确认

三个成分分别是现金持有变化的符号反转、无形资产强度的倒 U 和杠杆变化的 U 型。

| 成分 | 形态 | 完整恢复率 |
|---|---|---:|
| 现金持有变化 | 符号反转 | 19/20 = 95% |
| 无形资产强度 | 倒 U | 18/20 = 90% |
| 杠杆变化 | U 型 | 20/20 = 100% |

总体 CATE fidelity 为 100%，总体 CATE Spearman 中位数为 `.993`，标准化 RMSE
中位数为 `.025`，全 null 下任一假确认率为 5%。

![三组件恢复与全 null 控制](three_component/evidence/FIGURE_1_RECOVERY_AND_NULL_CONTROL.svg)

这说明三种不同函数形态可以在同一套逐成分标准下被剥离。5% 的全 null 结果表示：
20 次匹配 null replication 中有 1 次至少假确认了一个成分，仍低于预设的 10%
family-wise 上限。

## 研究解释与边界

两项确认共同建立了单情境结果之外的一项有限但实质性的扩展：对于信号清晰、活动
成分预先指定、加性且无交互的异质性，两个和三个企业特征的贡献可以被分别剥离，
同时保持逐成分准确性、总体 CATE fidelity 和全 null family-wise 错误控制。

该结果不等于可以始终自动发现活动变量，也不覆盖交互效应、任意变量与形态组合或
不受限制的组件数量。这些属于后续独立的能力边界问题，不改变本模块已经验证的结论。

## 复现与审计

在仓库根目录分别运行：

```powershell
python additive_coexistence/src/run.py --config additive_coexistence/two_component/config.json
python additive_coexistence/src/run.py --config additive_coexistence/three_component/config.json
python additive_coexistence/verify.py
```

验证程序检查共享方法合同、两份独立配置、验收结果、规定文件和 SHA256。详细信息见
[METHOD.md](METHOD.md)、[两组件报告](two_component/evidence/REPORT.zh-CN.md)和
[三组件报告](three_component/evidence/REPORT.zh-CN.md)。
