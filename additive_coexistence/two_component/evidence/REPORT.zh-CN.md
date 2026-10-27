# 2 成分加性共生示例

## 决策

按照冻结的逐成分统一标准，结果为 **通过**。

该示例使用真实 L=3 企业协变量骨架、`n=3000`、一次 80/20 TRAIN/untouched
TEST 划分，以及一棵同时纳入 12 个候选企业特征的 1,000-tree
`CausalForestDML`。活动变量集合在能力实验中预先指定，不使用 oracle 真值或 TEST
结果选择。

## 各成分结果

| 企业特征 | 形态 | 成功次数 | 完整恢复率 |
|---|---|---:|---:|
| Intangible-asset intensity | u-shaped | 20/20 | 100% |
| Fixed-asset intensity | inverted-u | 18/20 | 90% |

完整恢复要求实践侧证据、准确形态、曲线 Spearman 不低于 `.95`、标准化 RMSE
不高于 `.20`，并且全部预设关键量通过不变的单情境容差。

## 总体与全 null 结果

- 总 CATE fidelity：100%
- 重构 CATE Spearman 中位数：0.997
- 标准化 RMSE 中位数：0.025
- 匹配全 null 下任一假确认率：0%

`FIGURE_1_RECOVERY_AND_NULL_CONTROL.svg` 分别展示每个成分、总体 CATE 和全
null 错误率；`FIGURE_2_COMPONENT_CURVES.svg` 展示 20 次 fresh replication 的
真实曲线、估计中位曲线及 10%-90% 区间。

该结果支持这个信号清晰、预先指定的加性示例；它不代表自动发现活跃变量、交互
效应或任意组合的普遍恢复能力。

## 研究含义

两组件结果是单特征基准之外的第一步扩展。它表明两个成分共同构成同一个 CATE 面
后，仍可以分别维持完整的形态和关键量标准。总体 CATE 与全 null 检验同时通过，
说明这种剥离不是以牺牲总体准确性或错误控制为代价。
