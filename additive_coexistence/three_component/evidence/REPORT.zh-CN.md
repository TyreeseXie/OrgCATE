# 3 成分加性共生示例

## 决策

按照冻结的逐成分统一标准，结果为 **通过**。

该示例使用真实 L=3 企业协变量骨架、`n=3000`、一次 80/20 TRAIN/untouched
TEST 划分，以及一棵同时纳入 12 个候选企业特征的 1,000-tree
`CausalForestDML`。活动变量集合在能力实验中预先指定，不使用 oracle 真值或 TEST
结果选择。

## 各成分结果

| 企业特征 | 形态 | 成功次数 | 完整恢复率 |
|---|---|---:|---:|
| Change in cash holdings | sign-reversal | 19/20 | 95% |
| Change in leverage | u-shaped | 20/20 | 100% |
| Intangible-asset intensity | inverted-u | 18/20 | 90% |

完整恢复要求实践侧证据、准确形态、曲线 Spearman 不低于 `.95`、标准化 RMSE
不高于 `.20`，并且全部预设关键量通过不变的单情境容差。

## 总体与全 null 结果

- 总 CATE fidelity：100%
- 重构 CATE Spearman 中位数：0.993
- 标准化 RMSE 中位数：0.025
- 匹配全 null 下任一假确认率：5%

`FIGURE_1_RECOVERY_AND_NULL_CONTROL.svg` 分别展示每个成分、总体 CATE 和全
null 错误率；`FIGURE_2_COMPONENT_CURVES.svg` 展示 20 次 fresh replication 的
真实曲线、估计中位曲线及 10%-90% 区间。

该结果支持这个信号清晰、预先指定的加性示例；它不代表自动发现活跃变量、交互
效应或任意组合的普遍恢复能力。

## 研究含义

三组件结果把已经验证的边界推进到同一 CATE 面中的三种不同函数形态。各成分仍
满足单特征完整恢复标准，总体 CATE fidelity 保持 100%，family-wise 假确认率也
低于预设上限。因此，本结果支持的是冻结加性设计下的逐成分剥离，而不只是对合并
效应的准确预测。
