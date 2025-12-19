# 📚 Daily Paper Assistant

一个自动化的科研论文追踪助手，每天自动搜集、筛选、总结最新论文并发送到你的邮箱。

## ✨ 功能特性

- **多来源聚合**: arXiv、HuggingFace Daily Papers、手动添加
- **智能筛选**: 关键词匹配 + 可选 LLM 精筛
- **AI 总结**: Claude 生成论文摘要和研究洞察
- **自动推送**: 通过 GitHub Actions 定时发送邮件
- **可扩展**: 预留了作者筛选、单位筛选、Embedding 相似度等接口

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/paper-assistant.git
cd paper-assistant
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml，设置你的关键词和研究兴趣
```

### 4. 设置环境变量

```bash
export ANTHROPIC_API_KEY="your-api-key"
export RESEND_API_KEY="your-resend-key"
export EMAIL_TO="your-email@example.com"
```

### 5. 本地测试

```bash
# 预览模式（不发送邮件）
python main.py --dry-run

# 正常运行
python main.py

# 查看更多天的论文
python main.py --days 3
```

## 📧 部署到 GitHub Actions

1. Fork 这个仓库
2. 在仓库设置中添加 Secrets:
   - `ANTHROPIC_API_KEY`
   - `RESEND_API_KEY`
   - `EMAIL_TO`
3. 启用 GitHub Actions
4. 默认每天 UTC 7:00 运行（可在 `.github/workflows/daily-digest.yml` 中修改）

## 📁 项目结构

```
paper-assistant/
├── main.py              # 主入口
├── config.py            # 配置管理
├── models.py            # 数据模型
├── sources.py           # 论文来源（arXiv, HF, Manual）
├── filters.py           # 筛选器（关键词, LLM, 作者）
├── summarizer.py        # Claude 摘要生成
├── emailer.py           # 邮件发送
├── config.yaml          # 配置文件
├── manual_papers.json   # 手动添加的论文
└── .github/workflows/   # GitHub Actions
```

## 🔧 配置说明

### 关键词配置

```yaml
keywords:
  - diffusion model
  - chain of thought
  - ai safety
```

论文标题或摘要匹配任一关键词即被选中。

### 研究兴趣描述

用于 LLM 筛选和生成更相关的总结：

```yaml
research_interests: |
  我的研究方向包括：
  1. 扩散模型，特别是语言扩散模型
  2. LLM 推理，包括 Chain-of-Thought
  ...
```

### LLM 筛选（可选）

当论文太多时，可启用 LLM 二次筛选：

```yaml
llm_filter_enabled: true
llm_filter_threshold: 30  # 超过30篇时启用
```

## 📝 手动添加论文

编辑 `manual_papers.json`：

```json
{
  "papers": [
    {
      "url": "https://arxiv.org/abs/2401.00001",
      "notes": "导师推荐"
    }
  ]
}
```

也可以只添加 URL，系统会自动获取元数据。

## 🔮 未来计划

- [ ] Cloudflare D1 集成（支持 Chatbot 自动添加论文）
- [ ] Telegram Bot 交互
- [ ] Embedding 相似度筛选
- [ ] 作者/单位关注列表
- [ ] OpenReview 会议论文追踪
- [ ] 论文阅读进度追踪

## 📄 License

MIT
