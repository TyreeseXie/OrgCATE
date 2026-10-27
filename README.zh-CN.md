# OrgCATE

[English](README.md)

OrgCATE 是一个用于研究组织情境中处理效应异质性的 Python 项目。程序首先用
cross-fitted DML/AIPW 估计平均处理效应，再用一棵包含全部候选企业特征的
`CausalForestDML` 估计 CATE，并沿焦点企业特征重构效应函数。最终结果包括函数
形态及其关键量，例如斜率、转折点、符号转换点和阈值位置。

研究流程如下：

```text
研究设计与 overlap 检查
  -> cross-fitted DML/AIPW ATE
  -> ATE-CATE 一致性检查
  -> 一棵多变量 CausalForestDML
  -> held-out 数据上的 CATE 属性检验
  -> 函数形态分类与参数估计
  -> 由研究者完成理论解释记录
```

模型估计、held-out 确认和模拟评价使用不同的数据对象。模拟真值只用于评价方法，
不会进入模型估计或形态分类。

## 本仓库建立的两项方法结果

本仓库包含两项相互衔接的结果。第一，针对单个焦点企业特征，恢复其 CATE 函数
形态及决定该形态的关键量。第二，把同一套可审计标准扩展到有限的加性共生情境：
当两个或三个预先指定的企业特征共同构成同一个 CATE 面时，分别剥离并评价每一条
异质性成分，而不是只判断总体 CATE 是否拟合良好。

共生模块没有降低单情境标准。每个活动成分都必须独立通过形态、曲线和关键量验收，
重构后的总体 CATE 也必须通过，并使用匹配的全 null 情境控制“任一成分被假确认”
的概率。

## 安装与基本检查

参考环境为 Python 3.13 和 EconML 0.17.0，完整版本记录在
`requirements-lock.txt`。

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps

orgcate doctor
orgcate verify
orgcate contract
orgcate evidence
```

`doctor` 检查运行环境和随仓库提供的 backbone；`verify` 核对代码、配置与结果
哈希；`contract` 检查方法和发布条件；`evidence` 输出九种情境的汇总结果。

## 合成数据示例

下面的命令使用模拟协变量、处理变量和结果变量运行完整流程。示例采用
`n=3000` 和与正式实验相同的 1,000 棵树配置。

```powershell
orgcate demo --scenario sign-reversal --output artifacts/public_demo
```

输出分为 `analyst/`、`practical/`、`oracle/` 和 `evaluator/`。前两个目录保存
估计输入与最终报告，后两个目录保存模拟真值和准确度指标。程序还会生成一份空白
解释记录；研究者填写后可以检查其结构：

```powershell
orgcate theory-audit --input artifacts/public_demo/practical/researcher_theory_audit.json
```

## 函数形态

| 形态 | 主要输出 |
|---|---|
| homogeneous | 恒定效应估计 |
| linear-monotone | 方向与斜率 |
| nonlinear-monotone | 方向、中心、幅度、转变宽度、有效梯度和局部斜率 |
| steep-continuous | 方向、中心、较窄的转变宽度、有效梯度和局部斜率 |
| inverted-u | 转折点、峰值和两侧差异 |
| u-shaped | 转折点、谷值和两侧差异 |
| kinked | 折点位置及左右斜率 |
| threshold-step | 边界位置及边界两侧效应 |
| sign-reversal | 符号转换点、端点效应和斜率 |

多个有序阈值作为 `threshold-step` 的子型报告，不另设一个高层类别。

## 验证标准与结果

九种单情境使用同一套联合标准。一个异质性 replication 只有同时满足以下条件才算
成功：盲态 TRAIN/TEST 证据支持固定候选形态、操作性形态名称正确、曲线 Spearman
相关不低于 `.95`、标准化曲线 RMSE 不高于 `.20`，并且该形态要求的所有关键量均
准确。位置和宽度在冻结的 0--1 ECDF（经验分位秩）尺度上的误差不得超过 `.10`；
效应水平、斜率、曲率、对比度和幅度在观测支持区间内换算后的误差，不得超过真实
CATE 极差的 20%。10 次冻结重复中至少 9 次整体成功，该情境才通过。实践程序固定
输出后，评价器才读取模拟真值。

下表报告 `n=3000` 下的统一恢复率。

| 函数形态 | 统一恢复率 |
|---|---:|
| homogeneous | 10/10 |
| linear-monotone | 10/10 |
| nonlinear-monotone | 9/10 |
| steep-continuous | 10/10 |
| inverted-u | 10/10 |
| u-shaped | 10/10 |
| kinked | 10/10 |
| threshold-step | 10/10 |
| sign-reversal | 10/10 |

`nonlinear-monotone` 一行来自操作性转变规则冻结后的独立专项确认；`kinked` 一行
来自折点专项 v2 的独立确认。完整定义及各形态的必需关键量见
`docs/SINGLE_SCENARIO_ACCEPTANCE_STANDARD_V1.md`。

![九种单情境统一恢复结果](reports/single_scenario_release_candidate_v1/FIGURE_SINGLE_SCENARIO_UNIFIED_RECOVERY.svg)

## 加性异质性共生

第二个研究模块考察：一棵联合因果森林估计完整 CATE 后，能否剥离多个预先指定的
企业特征贡献。它把研究从“每次恢复一条形态”推进到有限的多成分共生，同时保持
逐成分验收标准不变。两组件和三组件共用一套方法，但分别配置、分别运行、分别
汇报。

| 示例 | 各成分完整恢复率 | 总 CATE fidelity | 全 null 下任一假确认率 |
|---|---:|---:|---:|
| 两组件 | 90%、100% | 100% | 0% |
| 三组件 | 95%、90%、100% | 100% | 5% |

这些 `n=3000` 结果建立了一个可审计的能力边界：在信号清晰、活动成分预先指定、
加性且无交互时，程序可以剥离两个和三个共生异质性成分，同时保持总体 CATE
恢复和全 null 错误控制。该结论不等于自动发现活动变量，也不代表任意组合都能
恢复。两个实验各自的配置、报告、汇总表、图和 manifest 见
[`additive_coexistence/`](additive_coexistence/)，综合解释见
[模块报告](additive_coexistence/ADDITIVE_COEXISTENCE_REPORT.zh-CN.md)。

![两组件恢复与全 null 控制](additive_coexistence/two_component/evidence/FIGURE_1_RECOVERY_AND_NULL_CONTROL.svg)

![三组件恢复与全 null 控制](additive_coexistence/three_component/evidence/FIGURE_1_RECOVERY_AND_NULL_CONTROL.svg)

## 半合成复现

仓库包含实验所用、经研究团队授权的 Compustat 派生 L=3 处理前企业面板骨架：

`execution/phase2_real_backbone_v1/headline_one_anchor_per_firm_L3.pkl.gz`

该文件只提供企业协变量。处理变量、潜在结果、观测结果、真实 CATE 和形态标签均
由版本化代码模拟。完整实验通过以下命令运行：

```powershell
orgcate reproduce --target morphology-product --phase confirmation
orgcate reproduce --target morphology-product --phase report
orgcate reproduce --target kink-resolution --phase confirmation
orgcate reproduce --target kink-resolution --phase report
python additive_coexistence/src/run.py --config additive_coexistence/two_component/config.json
python additive_coexistence/src/run.py --config additive_coexistence/three_component/config.json
python additive_coexistence/verify.py
```

程序会保留已经完成的 replication，并记录运行失败。`--freeze-only` 仅用于文档
规定的“先开发、后确认”流程。

## 可下载的数据到结果验收

CI 还会使用仓库中的 L=3 backbone 对九种预设情境分别做一次固定种子的全新
分析，而不是调用合成协变量生成器：

```powershell
orgcate acceptance --config configs/data_to_results_acceptance_v1.json --output artifacts/ci_acceptance
python tools/verify_acceptance_artifacts.py artifacts/ci_acceptance
```

上传的真实数据提供 8,648 行处理前企业协变量。程序针对每种情境从中抽取 3,000
家企业，重新模拟处理变量与因果结果，重新拟合 DML ATE 和一棵联合因果森林。
下载包包含九情境汇总表、九宫格总图、每种情境各两张 SVG 图片、诊断信息、分析
报告和 SHA256 manifest，但不包含企业级数据行。

每次 GitHub Actions 成功运行后，打开该次运行的 summary 页面，在 **Artifacts**
区域下载 `orgcate-data-to-results-<run_id>`。

同一次运行还会提供 `orgcate-additive-coexistence-<run_id>`，其中包含两份独立确认
报告、图片、汇总表、配置和证据 manifest，不包含企业级数据行。

## 文档与结果

- `docs/METHOD_CONTRACT_AND_STOP_RULES.md`：估计顺序与停止规则；
- `docs/TECHNICAL_APPENDIX.md`：估计量、样本划分、形态重构、验证标准和复现细节；
- `docs/SINGLE_SCENARIO_ACCEPTANCE_STANDARD_V1.md`：冻结的单情境成功标准；
- `additive_coexistence/`：共生方法、两组件与三组件独立确认及综合模块报告；
- `docs/REPOSITORY_VERIFICATION_RECORD.md`：代码、证据、数据和端到端分析的最终
  验证记录；
- `docs/OUTPUT_SCHEMA.md`：结果字段与状态定义；
- `docs/SHAPE_OUTPUT_REFERENCE.md`：各形态报告的参数；
- `reports/morphology_product_v1/`：主实验结果与图；
- `reports/kink_resolution_v2/`：连续折点专项实验；
- `DATA_NOTICE.md`：L=3 backbone 的内容与授权范围；
- `PUBLIC_RELEASE_MANIFEST.json`：文件清单与 SHA256。

仓库只包含 `DATA_NOTICE.md` 指定的 backbone。原始下载、其他处理后面板、访问记录、
凭证及模拟生成的企业级结果不在发布范围内。

## 开发检查

```powershell
.venv\Scripts\ruff.exe check src tests execution tools additive_coexistence
.venv\Scripts\python.exe -m compileall -q src tests execution tools additive_coexistence
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
python tools/verify_single_scenario_evidence.py
python additive_coexistence/verify.py
python tools/audit_public_release.py --allow-metadata-pending
orgcate acceptance --config configs/data_to_results_acceptance_v1.json --output artifacts/ci_acceptance
python tools/verify_acceptance_artifacts.py artifacts/ci_acceptance
```

加入 `LICENSE` 和 `CITATION.cff` 后，最终审计不再使用 metadata 例外。
