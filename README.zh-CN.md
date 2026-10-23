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
| nonlinear-monotone | 效应排序与非线性强度 |
| steep-continuous | 排序与剧烈连续转变的位置 |
| inverted-u | 转折点、峰值和两侧差异 |
| u-shaped | 转折点、谷值和两侧差异 |
| kinked | 折点位置及左右斜率 |
| threshold-step | 边界位置及边界两侧效应 |
| sign-reversal | 符号转换点、端点效应和斜率 |

多个有序阈值作为 `threshold-step` 的子型报告，不另设一个高层类别。

## 验证结果

下表合并 Morphology Product v1 与单独开展的 kink-resolution v2。每个比例基于
`n=3000` 下的 10 次确认实验。

| 函数形态 | 精确恢复 |
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

`kinked` 一行报告的是折点专项 v2 独立确认后的最终结果。

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
```

程序会保留已经完成的 replication，并记录运行失败。`--freeze-only` 仅用于文档
规定的“先开发、后确认”流程。

## 文档与结果

- `docs/METHOD_CONTRACT_AND_STOP_RULES.md`：估计顺序与停止规则；
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
.venv\Scripts\ruff.exe check src tests execution tools
.venv\Scripts\python.exe -m compileall -q src tests execution tools
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
python tools/audit_public_release.py --allow-metadata-pending
```

加入 `LICENSE` 和 `CITATION.cff` 后，最终审计不再使用 metadata 例外。
