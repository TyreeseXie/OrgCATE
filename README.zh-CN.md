# OrgCATE

[English](README.md)

OrgCATE 是一套由研究者主导的处理效应异质性理论发展方法。它从可信的因果设计
出发，将交叉拟合的 ATE 估计、多变量因果森林、留出样本确认和可解释的 CATE
形态重构连接为一条完整路径。

```text
可信的因果设计
  -> 处理变量与 overlap 合格性检查
  -> cross-fitted DML/AIPW ATE 锚点
  -> 常规 OLS 与辅助面板固定效应对比
  -> 一棵包含全部候选变量的 CausalForestDML
  -> DML-ATE/CATE gap failure test
  -> held-out 验证与可支持的 CATE 属性
  -> 预先指定的焦点企业特征 + 全部变量排名信息
  -> 三层分类 + 易读的函数形态
  -> 支持候选 / 证据不足 / 不作主张 / 设计不合格
  -> 在结果得到支持后可选用大模型辅助理论发展
  -> 研究者主导的理论审计或克制结论
```

因果设计负责效应识别，因果森林负责估计异质性，留出样本负责检验候选模式，
研究者负责提出并评估理论解释。

## 一页读懂方法

本项目的估计对象是条件平均处理效应
`tau(x) = E[Y(1) - Y(0) | X=x]`。基准实验采用半合成架构：真实
Compustat 派生的处理前企业历史提供协变量联合分布，处理变量、潜在结果、观测
结果和 oracle CATE 由程序模拟。Oracle 字段不会进入实践工作流。

| 环节 | 估计或检查内容 | 在工作流中的作用 |
|---|---|---|
| 设计合格性 | 二元处理、overlap、支持范围和必要变量 | 因果设计不合格时停止分析 |
| ATE 锚点 | 五折 cross-fitted DML/AIPW | 独立估计总体平均处理效应 |
| 常规估计对照 | `Y ~ 1 + T + X` 中的恒定处理系数，并报告 HC3 不确定性 | 比较常规同质效应估计与 DML-ATE |
| 面板固定效应对比 | 吸收企业和时期固定效应后估计 `T_i x Post_t`，并使用企业聚类不确定性 | 提供辅助性的重复结果对比，不改变 CATE 主流程 |
| HTE 估计 | 一棵联合纳入 12 个企业特征的 `CausalForestDML` | 在多变量条件下估计 CATE，不为每个变量单独训练森林 |
| DML-ATE/CATE 检验 | `abs(CATE_TRAIN_TARGET - DML-ATE_TRAIN) / SD(Y_train) <= 0.10` | 在同一组目标企业上比较两种估计；不一致时停止后续解释并输出 `INCONCLUSIVE` |
| 形态重构 | 仅在 TRAIN 上进行正交化形态拟合 | 把噪声较大的森林曲面转为斜率、转折点、零点、折点或边界等明确参数 |
| 独立确认 | untouched TEST 上的正交损失与组件消融 | 把模型构造与证据确认分开 |
| 理论审计 | 研究者填写机制、文献、排除理由和最终决定 | 理论发展由研究者负责，并允许克制结论 |
| 可选大模型辅助 | 只把得到支持的估计结果和关键形态参数输入可复用 prompt | 生成候选机制和命题，交由研究者复核 |

对观测 `i`，AIPW score 为：

```text
psi_i = m1(X_i) - m0(X_i)
        + T_i * (Y_i - m1(X_i)) / e(X_i)
        - (1 - T_i) * (Y_i - m0(X_i)) / (1 - e(X_i)).
```

所有 nuisance function 都在样本外拟合，score 中的倾向得分截断在
`[0.05, 0.95]`。主因果森林使用外部 80/20 TRAIN/TEST 划分，`cv=2`、
`honest=True`、1,000 棵树、`min_samples_leaf=5`、`max_samples=0.45`。
每条合格曲线都进入同一个常数、线性、二次、tanh、hinge 与分段常数候选库；
形态族和候选参数只在 TRAIN 上通过统一分类路径选择。TEST 只用于
确认，oracle 真值只用于模拟评价。完整公式推导、固定效应对比、形态恢复结果及可选理论发展示例见
[`docs/TECHNICAL_APPENDIX.docx`](docs/TECHNICAL_APPENDIX.docx)。

## 快速开始

当前验证环境为 Python 3.12+ 和 EconML 0.17.0，精确版本记录在
`requirements-lock.txt`。

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps

orgcate doctor
orgcate verify
orgcate evidence
```

运行一个完全公开的合成演示。演示采用冻结的 `n=3000`、1,000 棵树和正式
工作流配置。

```powershell
orgcate demo --scenario sign-reversal --output artifacts/public_demo
```

输出按 `analyst/`、`oracle/`、`practical/` 和 `evaluator/` 物理分离，并附带
文件哈希 manifest。`--scenario` 可以选择下列九种情境中的任意一种。

`practical/` 还会生成空白的研究者 Step 4 审计表。研究者完成机制候选、文献依据、
排除理由与采纳/克制决定后，可检查其结构：

```powershell
orgcate theory-audit --input artifacts/public_demo/practical/researcher_theory_audit.json
```

## 编辑与审稿人完整复现

下列命令重新计算并整合全部公开研究模块：

```powershell
orgcate acceptance --output artifacts/reviewer_nine_situations
orgcate ate-comparison --output artifacts/reviewer_ate_comparison
orgcate fixed-effects-comparison --output artifacts/reviewer_fixed_effects
python tools/verify_acceptance_artifacts.py artifacts/reviewer_nine_situations
python tools/verify_average_effect_artifacts.py --ate artifacts/reviewer_ate_comparison --fixed-effects artifacts/reviewer_fixed_effects
python tools/assemble_additive_coexistence_artifact.py artifacts/reviewer_additive
python tools/verify_additive_coexistence_artifact.py artifacts/reviewer_additive
python tools/assemble_complete_analysis_artifact.py --acceptance artifacts/reviewer_nine_situations --ate artifacts/reviewer_ate_comparison --fixed-effects artifacts/reviewer_fixed_effects --additive artifacts/reviewer_additive --output artifacts/reviewer_complete
python tools/verify_complete_analysis_artifact.py artifacts/reviewer_complete
```

首先打开 `artifacts/reviewer_complete/README.md`。完整成果包包括：

| 模块 | 内容 |
|---|---|
| 九情境分析 | 重新计算的统计量、诊断、CATE 曲线、变量排名、图片和报告 |
| 平均效应比较 | 重新计算的常规 OLS 与 cross-fitted DML-ATE 结果 |
| 面板固定效应比较 | 重新计算的企业和时期固定效应，以及 True ATE、DML-ATE 与 CATE 对照 |
| 异质性共生 | 哈希核验的两组件和三组件结果 |
| 可选理论发展 | 可复用英文 prompt 与示范性大模型输出 |

成果包首页直接链接九情境总图、两张平均效应比较图、异质性共生报告、
完整 Prompt、示例输出和技术附录，并在首页概括 DML-ATE/CATE 检验及
形态恢复结果。编辑或读者无需逐层搜索目录即可找到主要证据。

GitHub Actions 执行相同流程。进入仓库的 **Actions** 页面，打开成功的
`public-artifact-ci`，在 **Artifacts** 下载
`orgcate-complete-analysis-<run_id>`。ZIP 中含完整目录说明和逐文件哈希
manifest；同一页面还提供单独的九情境分析包与共生模块包。

## 九情境形态层

| 情境 | 关键输出 |
|---|---|
| homogeneous | 不形成异质性理论主张 |
| linear-monotone | 方向与斜率 |
| nonlinear-monotone | 效应排序与非线性强度 |
| steep-continuous | 排序与剧烈连续转变位置 |
| inverted-u | 转折点、峰值与两侧差异 |
| u-shaped | 转折点、谷值与两侧差异 |
| kinked | 折点位置及左右斜率 |
| threshold-step | 候选边界及边界两侧效应差异 |
| sign-reversal | 符号转换点、两端效应与符号、全局及转换点两侧斜率 |

多个有序边界作为 `threshold-step` 的多阶梯子型报告，不另设第十种高层情境。

## 冻结结果

下表结果由同一个九类盲态分类器生成。每种情境使用相同运行路径和相同验收标准。

| 情境 | Fresh 统一恢复 |
|---|---:|
| homogeneous | 9/10 |
| linear-monotone | 10/10 |
| nonlinear-monotone | 10/10 |
| steep-continuous | 10/10 |
| inverted-u | 9/10 |
| u-shaped | 10/10 |
| kinked | 9/10 |
| threshold-step | 10/10 |
| sign-reversal | 10/10 |

`orgcate evidence --format json` 输出机器可读结果，`orgcate verify` 复核冻结
代码、配置和结果哈希；`orgcate contract` 检查 proposal 层面的停止规则。九行均由
同一分类器生成，只在分类完成后的关键量读取方式上有所不同。

下列命令可在九种一维 HTE 情形中比较常规平均效应估计与 DML-ATE，且不训练
因果森林：

```powershell
orgcate ate-comparison --output artifacts/ate_estimator_comparison
```

输出逐情形报告常规恒定处理系数 OLS、cross-fitted DML-ATE、二者偏离比例，以及
仅在模拟评价端计算的 bias、RMSE 和置信区间覆盖率。每家企业贡献一个处理—结果
观测，因此常规基准采用控制处理前协变量并报告 HC3 不确定性的 OLS。

附加的面板固定效应比较把同一批已经冻结的企业层 replication 扩展为六个相对
时期，并在吸收企业和时期固定效应后估计 `T_i x Post_t`，标准误按企业聚类：

```powershell
orgcate fixed-effects-comparison --output artifacts/fixed_effects_comparison
```

图中同时呈现 True ATE、固定效应系数、DML-ATE 和 CATE。该命令直接复用已
冻结的 DML 与 CATE 估计，不重新训练因果森林。结果见
[`reports/fixed_effects_comparison_v1/`](reports/fixed_effects_comparison_v1/)。

## 多个异质性成分共存

单形态基准回答一个焦点变量的形态能否恢复；独立的能力边界 pilot 则进一步
考察：当多个预先指定的企业特征共同以加性方式影响 CATE 时，同一棵联合森林和
正交重构能否把各自贡献剥离出来。这组实验用于刻画加性共存条件下的初步能力边界。

目前已有两个按照单情境同等成分标准通过的仓库示例。在 `n=3000`、`R=20` 下，
两成分示例分别以 90% 和 100% 完整恢复固定资产强度的倒 U 与无形资产强度的
U 型；三成分示例分别以 95%、90% 和 100% 完整恢复现金持有变化的符号反转、
无形资产强度的倒 U 和杠杆变化的 U 型。两种示例的总 CATE fidelity 均为
100%，匹配全 null 下任一假确认率分别为 0% 和 5%。复现命令、图表、报告和哈希见
[`additive_coexistence/`](additive_coexistence/)。当前证据覆盖这两个预先指定的两成分
和三成分加性设计；活跃变量集合发现、交互效应和更多成分是后续能力研究的方向。

## 完整半合成复现

GitHub 上传候选已经包含研究团队明确授权用于复现的 Compustat 派生 L=3
处理前企业面板骨架：
`execution/phase2_real_backbone_v1/headline_one_anchor_per_firm_L3.pkl.gz`。
该文件只提供真实企业协变量；处理变量、潜在结果、观测结果、真实 CATE 和全部
oracle 标签均由程序重新模拟。先运行 `orgcate doctor` 验证数据与环境，再通过
统一入口运行：

```powershell
orgcate reproduce --target unified-morphology --phase confirmation
orgcate reproduce --target unified-morphology --phase report
orgcate reproduce --target unified-morphology --phase acceptance
```

runner 会保存已完成的 cell、保留失败记录，并从最近的完成位置继续运行。
`--freeze-only` 用于“先开发、再冻结、后确认”的协议。

## 输出与审计

- `schemas/orgcate-result-v1.schema.json`：实践输出 schema；
- `docs/OUTPUT_SCHEMA.md`：字段含义和四种实践状态；
- `docs/SHAPE_OUTPUT_REFERENCE.md`：九类形态的英文关键量字段，包括转折点、
  曲率、折点、候选边界、斜率、分段效应、跳跃幅度与零点；
- `docs/UNIFIED_MORPHOLOGY_V2_METHOD.md`：共同候选库、判定顺序和关键量读取规则；
- `reports/unified_morphology_v2/`：统一 fresh 证据、混淆矩阵与图；
- `reports/ate_estimator_comparison_v1/`：九种形态下常规 OLS 与 DML 的偏差、覆盖率和图；
- `reports/fixed_effects_comparison_v1/`：企业与时期固定效应、True ATE、DML-ATE 和 CATE 的附加比较；
- `docs/REPOSITORY_VERIFICATION_RECORD.md`：公开仓库验证记录。

Oracle 真值不得进入实践侧变量排名、调参、形态选择、边界选择或理论状态。

## 可选理论发展提示框架

仓库只收录英文版
[`optional_theory_development/PROMPT.md`](optional_theory_development/PROMPT.md)，
用于将估计得到的 CATE 形态转化为组织机制、理论命题和后续证据设计。英文示例结果见
[`AI_Strategic_Decision_Making_Theory_Development.txt`](optional_theory_development/AI_Strategic_Decision_Making_Theory_Development.txt)。
提示词输入来自 DML、因果森林、形态重构和留出样本确认的估计结果。

## 数据与发布安全

研究团队仅明确授权发布公共候选中 `DATA_NOTICE.md` 记录并经哈希固定的 L=3
处理后 backbone。不得再加入其他原始下载、转换后的企业面板、访问记录、凭证、
机器专属路径或生成的 analyst/oracle 行级结果。

第一次公开发布前，研究团队仍需批准软件许可证、引用信息、作者顺序以及允许纳入
仓库的汇总结果。proposal 中可执行的方法要求已经接入；权威状态记录于
`docs/METHOD_CONTRACT_AND_STOP_RULES.md` 和
`configs/github_release_method_contract_v1.json`。详见 `docs/RELEASE_CHECKLIST.md`。
`orgcate contract` 检查这些发布要求；在 `LICENSE` 和 `CITATION.cff` 获得批准前，
它会返回阻断状态。该状态与分析是否成功是两项不同检查。

## 验证命令

```powershell
.venv\Scripts\ruff.exe check src tests execution tools additive_coexistence
.venv\Scripts\python.exe -m compileall -q src tests execution tools additive_coexistence
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp reports/.pytest_release
orgcate verify
python tools/audit_public_release.py --allow-metadata-pending
```
