# OrgCATE GitHub 首次发布操作教程

这份教程只针对经过白名单复制、数据哈希限制和内容审计的独立上传候选目录。
下面用 `<UPLOAD_DIR>` 表示该目录。不要在私有研究仓库中初始化或连接公开
GitHub 仓库。

## 一、先理解将要建立的三层验证

1. **本地内容审计**：确认没有多余数据、凭据、绝对路径或未批准文件。
2. **本地干净运行**：确认公开副本脱离私有研究仓库后仍能安装、测试和运行。
3. **GitHub Actions 验证**：确认 GitHub 的全新 Linux 环境能够重复上述检查。

只有三层都通过，GitHub 上的绿色勾才有实质含义。

## 二、完成人工元数据

公开前必须由研究团队确定：

- 软件许可证；
- 项目正式标题；
- 作者顺序与英文姓名；
- 作者单位；
- 通讯作者及邮箱；
- 首个版本号，建议 `0.1.0`；
- 推荐引用方式。

将决定写入根目录的 `LICENSE` 和 `CITATION.cff`，更新 README 的引用部分，
并把 `configs/github_release_method_contract_v1.json` 中
`public_license_and_citation` 的状态从 `HUMAN_DECISION` 改为 `PASS`，同时记录
这两个文件作为 evidence。

选择许可证时：MIT 最简洁、允许范围最宽；BSD-3-Clause 额外禁止使用作者或
机构名称为衍生产品背书，通常很适合学术软件。许可证只适用于代码；授权的 L=3
复现 backbone 的范围另由 `DATA_NOTICE.md` 界定。

## 三、本地最终检查

打开 PowerShell：

```powershell
Set-Location <UPLOAD_DIR>

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps

.\.venv\Scripts\python.exe tools\audit_public_release.py
.\.venv\Scripts\ruff.exe check src tests execution tools
.\.venv\Scripts\python.exe -m compileall -q src tests execution tools
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\orgcate.exe doctor --json
.\.venv\Scripts\orgcate.exe verify --json
.\.venv\Scripts\orgcate.exe contract --json
.\.venv\Scripts\orgcate.exe demo --scenario sign-reversal --output artifacts\local_release_demo
```

预期结果：内容审计 `PASS`；pytest、Ruff、编译通过；`verify` 为 `PASS`；
`contract` 为 `READY`；demo 正常完成。`PUBLIC_RELEASE_MANIFEST.json` 会由内容
审计重新生成，必须和本次待提交文件一起进入 Git。

再单独核对唯一授权数据：

```powershell
Get-FileHash execution\phase2_real_backbone_v1\headline_one_anchor_per_firm_L3.pkl.gz -Algorithm SHA256
```

结果必须为：

`91C963E02CBDC17A19DE85868BA44F5F4FD65791BEC83906980F307A6CFD3174`

## 四、初始化本地 Git 仓库

仍在上传目录中执行：

```powershell
git init -b main
git config user.name "你的 GitHub 显示名称"
git config user.email "你的 GitHub 邮箱或 GitHub noreply 邮箱"

git add -A
git status --short
git diff --cached --stat
git diff --cached --name-only
```

这里先暂存再检查，不等于已经上传。重点确认：

- `.venv/`、`artifacts/`、缓存和日志没有被暂存；
- 没有 ZIP、Word、PowerPoint、原始数据或私有 handoff；
- 只有一个 `.pkl.gz` 数据文件，即哈希固定的 L=3 backbone；
- `LICENSE`、`CITATION.cff`、`DATA_NOTICE.md` 和 manifest 都在暂存列表中。

确认后创建第一个提交：

```powershell
git commit -m "release: initial OrgCATE public research artifact"
```

## 五、在 GitHub 网站创建空仓库

1. 登录 GitHub，点击右上角 `+`，选择 `New repository`。
2. 建议仓库名使用 `OrgCATE`。
3. 选择 Public 或先用 Private 做一次演练。
4. 不要勾选自动创建 README、`.gitignore` 或 LICENSE，因为本地已经准备好。
5. 创建后复制 GitHub 显示的 HTTPS 地址，例如：
   `https://github.com/<你的账号>/OrgCATE.git`。

本机当前没有安装 GitHub CLI，因此直接使用普通 Git：

```powershell
git remote add origin https://github.com/<你的账号>/OrgCATE.git
git remote -v
git push -u origin main
```

Windows Git Credential Manager 通常会弹出浏览器让你登录 GitHub。不要把访问
token 写入代码、配置文件或命令历史。

## 六、检查 GitHub Actions

推送后进入 GitHub 仓库的 `Actions` 页面，打开 `public-artifact-ci`。它应依次：

1. 安装 Python 3.13 和冻结依赖；
2. 严格审计公开目录；
3. 运行 Ruff 和编译；
4. 运行全部公开测试；
5. 验证冻结证据哈希；
6. 验证环境和方法契约；
7. 运行一次 sign-reversal 端到端公开 demo。

任何一步失败都不要通过删除测试或放宽审计解决。先打开失败步骤的日志，定位是
环境、路径、依赖、哈希、元数据还是模型运行问题，再做最小修复。

## 七、从 GitHub 做真正的独立复核

Actions 通过后，在另一个空目录重新 clone：

```powershell
Set-Location <用于独立复核的父目录>
git clone https://github.com/<你的账号>/OrgCATE.git OrgCATE_GitHub_Clone_Check
Set-Location OrgCATE_GitHub_Clone_Check

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\orgcate.exe doctor --json
.\.venv\Scripts\orgcate.exe verify --json
.\.venv\Scripts\orgcate.exe contract --json
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

这一轮证明的是：别人获得的 GitHub 仓库，而不是你电脑上的原研究目录，能够独立
完成安装与验证。

## 八、设置保护并发布 v0.1.0

在 GitHub `Settings -> Branches` 中为 `main` 添加保护规则，至少要求 CI 通过后
才能合并。确认 clone 复核通过后：

```powershell
git tag -a v0.1.0 -m "OrgCATE v0.1.0"
git push origin v0.1.0
```

然后在 GitHub `Releases` 中基于 `v0.1.0` 创建首个 release，简要说明方法范围、
九种输出形态、审计/克制机制、数据授权边界和已知局限。

## 九、以后每次更新的固定节奏

```text
修改 -> 本地测试 -> 严格审计 -> 检查暂存文件
-> commit -> push -> GitHub Actions -> 独立 clone 验证 -> tag/release
```

不要直接把私有研究仓库的新文件复制进公开仓库。每次新增文件都要先回答：它是否
是运行依赖、公开文档、测试或非披露性证据；是否带有路径、凭据、企业行级信息或
oracle 泄漏；是否已经进入审计白名单。
