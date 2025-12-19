# GitHub Actions 部署指南

## 第一步：准备 API Keys

你需要准备以下 API Keys：

### 1. Anthropic API Key
- 访问 https://console.anthropic.com/
- 登录后在 API Keys 页面创建新 key
- 格式: `sk-ant-api03-xxxxx`

### 2. Resend API Key（邮件服务）
- 访问 https://resend.com/
- 免费账户每月 3000 封邮件
- 创建账户后在 API Keys 页面获取
- 格式: `re_xxxxx`

### 3. 验证发件域名（可选但推荐）
- 在 Resend 控制台添加你的域名
- 或者使用默认的 `onboarding@resend.dev`（仅限测试）

---

## 第二步：创建 GitHub 仓库

### 方式 A：新建仓库

```bash
# 1. 解压项目
unzip paper-assistant.zip
cd paper-assistant

# 2. 初始化 Git
git init
git add .
git commit -m "Initial commit"

# 3. 在 GitHub 创建新仓库（不要勾选 README）

# 4. 推送
git remote add origin https://github.com/YOUR_USERNAME/paper-assistant.git
git branch -M main
git push -u origin main
```

### 方式 B：Fork 现有仓库
如果我发布到 GitHub，你可以直接 fork。

---

## 第三步：配置 GitHub Secrets

1. 打开你的 GitHub 仓库
2. 点击 **Settings** → **Secrets and variables** → **Actions**
3. 点击 **New repository secret**
4. 添加以下 secrets：

| Name | Value |
|------|-------|
| `ANTHROPIC_API_KEY` | `sk-ant-api03-xxxxx` |
| `RESEND_API_KEY` | `re_xxxxx` |
| `EMAIL_TO` | `your-email@example.com` |

![GitHub Secrets 设置示意](https://docs.github.com/assets/cb-28270/images/help/repository/actions-secret-repository.png)

---

## 第四步：启用 GitHub Actions

1. 点击仓库的 **Actions** 标签
2. 如果看到提示，点击 **I understand my workflows, go ahead and enable them**
3. 你会看到 "Daily Paper Digest" workflow

---

## 第五步：手动测试

1. 在 Actions 页面，点击 **Daily Paper Digest**
2. 点击右侧的 **Run workflow** 按钮
3. 可以选择：
   - `days_back`: 往前看几天的论文（默认 1）
   - `dry_run`: 勾选则不发邮件，只生成报告

![手动触发](https://docs.github.com/assets/images/help/repository/actions-manually-run-workflow.png)

---

## 第六步：自定义定时

编辑 `.github/workflows/daily-digest.yml`：

```yaml
on:
  schedule:
    # 每天北京时间早上 8:00 (UTC 0:00)
    - cron: '0 0 * * *'
    
    # 或者每天 UTC 7:00
    - cron: '0 7 * * *'
```

Cron 格式: `分钟 小时 日 月 星期`

常用时间：
- 北京时间 8:00 = `0 0 * * *` (UTC 0:00)
- 北京时间 9:00 = `0 1 * * *` (UTC 1:00)
- 美西时间 7:00 = `0 15 * * *` (UTC 15:00)

---

## 第七步：自定义关键词

编辑 `config.yaml` 文件中的 `keywords` 列表：

```yaml
keywords:
  - diffusion model
  - your specific topic
  - another keyword
```

修改后 commit 并 push：
```bash
git add config.yaml
git commit -m "Update keywords"
git push
```

---

## 常见问题

### Q: 邮件没收到？
1. 检查垃圾邮件文件夹
2. 在 Actions 页面查看运行日志
3. 确认 Resend 账户已验证邮箱

### Q: 运行失败？
查看 Actions 页面的日志，常见原因：
- API Key 配置错误
- 网络超时（arXiv 有时较慢）

### Q: 如何查看运行历史？
Actions → Daily Paper Digest → 查看所有运行记录

### Q: 费用估算
- **Anthropic API**: 每次运行约 $0.01-0.05（取决于论文数量）
- **Resend**: 免费额度 3000 封/月
- **GitHub Actions**: 免费额度 2000 分钟/月（私有仓库）或无限（公开仓库）

---

## 本地开发

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 创建 .env 文件
cp .env.example .env
# 编辑 .env 填入你的 API keys

# 3. 创建配置
cp config.example.yaml config.yaml
# 编辑 config.yaml 自定义关键词

# 4. 测试运行
python main.py --dry-run

# 5. 正式运行
python main.py
```
