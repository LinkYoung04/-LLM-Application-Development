# 分层上下文 Chat 系统

一个基于**四层分层上下文栈**的 LLM 对话系统，能够自动分类、压缩和管理对话历史，确保在 token 预算内保留最有价值的上下文信息。

## 核心特性

- **四层上下文架构**：固定锁层 / 核心任务层 / 临时对话层 / 无关归档层
- **自动消息分流**：从 JSON 历史文件导入时，自动按消息类型（任务内容 / 闲聊 / 空消息）分类
- **智能上下文压缩**：当 token 超限时，按优先级逐层清理（归档 → 早期任务历史 → 过期临时对话）
- **多会话管理**：支持创建、加载、切换多个独立会话
- **DeepSeek API 集成**：通过 OpenAI 兼容接口调用 DeepSeek 模型

## 上下文四层架构

| 层级 | 名称 | 说明 |
|------|------|------|
| L1 | 固定锁层 (Lock) | 角色提示词 + 全局硬性约束，永不删除 |
| L2 | 核心任务层 (Core Task) | 有效的任务相关对话，压缩时优先保留 |
| L3 | 临时对话层 (Temp Chat) | 最近 N 轮实时对话，超出限制时裁剪 |
| L4 | 归档层 (Archive) | 无关闲聊，默认不送入模型，超限时最先清空 |

## 环境配置

1. 复制 `.env` 文件并填入你的 API 密钥：

```env
LLM_API_KEY=your_deepseek_api_key
LLM_BASE_URL=https://api.deepseek.com/v1
MAX_TOKEN_LIMIT=4096
```

2. 安装依赖：

```bash
pip install openai python-dotenv
```

## 使用方式

```bash
python main.py
```

运行后提供两种模式：

- **模式 1**：从 `history_demo.json` 导入历史对话，自动分层清洗后开始对话
- **模式 2**：创建空白会话（CIFAR10 训练专属助手），可自定义角色和约束规则

## JSON 历史文件格式

```json
{
  "session_id": "demo_session_001",
  "meta": {
    "role_prompt": "系统角色提示词",
    "system_rules": ["硬性约束1", "硬性约束2"],
    "create_time": "2026-07-12 10:00:00",
    "tag": "标签"
  },
  "raw_messages": [
    {
      "msg_id": "m1",
      "role": "user",
      "content": "消息内容",
      "msg_type": "task_content | useless_chat | empty_msg",
      "timestamp": "2026-07-12 10:01",
      "valid_flag": true
    }
  ]
}
```

- `msg_type: "task_content"` → 归入核心任务层
- `msg_type: "useless_chat"` → 归入归档层
- `role: "ignore"` → 直接丢弃

## 项目结构

```
test3.0/
├── main.py              # 主程序：上下文栈 + 会话管理 + 交互入口
├── history_demo.json    # 示例历史对话 JSON（CIFAR10 训练场景）
├── .env                 # 环境变量配置
└── README.md
```