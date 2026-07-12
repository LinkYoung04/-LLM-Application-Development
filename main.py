import json
import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)
MAX_TOTAL_TOKEN = int(os.getenv("MAX_TOKEN_LIMIT", 4096))
RECENT_CHAT_LIMIT = 8


# LLM工具：统计token + 对话请求
def llm_count_token(messages: List[Dict]) -> int:
    """简易token统计，演示用；生产可用tiktoken"""
    total_len = 0
    for msg in messages:
        total_len += len(msg["content"])
    return total_len


def llm_chat(messages: List[Dict]) -> str:
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.7
    )
    return resp.choices[0].message.content


# 核心：四层分层上下文栈
class LayeredContextStack:
    def __init__(self):
        self.max_total_token = MAX_TOTAL_TOKEN
        self.recent_chat_limit = RECENT_CHAT_LIMIT
        # 四层存储
        self.lock_layer: List[Dict] = []       # 固定锁层：角色、全局规则（永不删除）
        self.core_task_layer: List[Dict] = []   # 核心任务有效对话
        self.temp_chat_layer: List[Dict] = []   # 最近N轮实时对话
        self.archive_layer: List[Dict] = []    # 无关闲聊归档（默认不进模型）

    # 从标准JSON导入历史对话
    def load_history_from_json(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            session_data = json.load(f)
        # 1. 加载固定锁层（角色+全局规则，永久生效）
        self.lock_layer.append({"role": "system", "content": session_data["meta"]["role_prompt"]})
        for rule in session_data["meta"]["system_rules"]:
            self.lock_layer.append({"role": "system", "content": f"硬性约束：{rule}"})
        # 2. 自动分流所有历史消息
        for msg in session_data["raw_messages"]:
            if msg["role"] == "ignore":
                continue  # 直接丢弃无效垃圾消息
            if msg["valid_flag"] and msg["msg_type"] == "task_content":
                self.core_task_layer.append({"role": msg["role"], "content": msg["content"]})
            elif msg["msg_type"] == "useless_chat":
                self.archive_layer.append({"role": msg["role"], "content": msg["content"]})
            else:
                self.temp_chat_layer.append({"role": msg["role"], "content": msg["content"]})
        # 3. 自动压缩、截断，控制总token不超限
        self.auto_compress_context()
        print("✅ JSON历史导入完成，已自动分层清洗上下文")

    def calc_all_token(self) -> int:
        # 仅计算送入模型的三层，归档闲聊不计入运行token
        all_valid = self.lock_layer + self.core_task_layer + self.temp_chat_layer
        return llm_count_token(all_valid)

    def auto_compress_context(self):
        total_tok = self.calc_all_token()
        if total_tok <= self.max_total_token:
            return
        print(f"⚠️ 上下文超限，开始自动清理，当前总字符：{total_tok}")
        # 清理优先级1：直接清空归档闲聊（最低价值）
        self.archive_layer.clear()
        if self.calc_all_token() <= self.max_total_token:
            print("✅ 已清空无关闲聊归档，上下文恢复正常")
            return
        # 清理优先级2：压缩核心任务最早历史（演示：简化为截断一半）
        if len(self.core_task_layer) > 4:
            cut_idx = len(self.core_task_layer) // 2
            self.core_task_layer = self.core_task_layer[cut_idx:]
        if self.calc_all_token() <= self.max_total_token:
            print("✅ 已压缩早期任务历史，上下文恢复正常")
            return
        # 清理优先级3：删减最近对话（最后手段）
        while len(self.temp_chat_layer) > self.recent_chat_limit:
            self.temp_chat_layer.pop(0)
        print("✅ 已删减过久临时对话，上下文恢复正常")

    # 拼接给模型的完整消息列表
    def get_model_input(self, user_query: str) -> List[Dict]:
        input_msgs = self.lock_layer + self.core_task_layer + self.temp_chat_layer
        input_msgs.append({"role": "user", "content": user_query})
        return input_msgs

    # 单轮对话入口
    def chat(self, user_query: str) -> str:
        input_msg = self.get_model_input(user_query)
        reply = llm_chat(input_msg)
        # 新对话存入临时缓存
        self.temp_chat_layer.append({"role": "user", "content": user_query})
        self.temp_chat_layer.append({"role": "assistant", "content": reply})
        self.auto_compress_context()
        return reply


# 会话管理器：多会话CRUD
class SessionManager:
    def __init__(self):
        self.session_pool: Dict[str, LayeredContextStack] = {}

    def create_empty_session(self, sid: str, role_prompt: str, rules: List[str]) -> LayeredContextStack:
        stack = LayeredContextStack()
        stack.lock_layer.append({"role": "system", "content": role_prompt})
        for r in rules:
            stack.lock_layer.append({"role": "system", "content": f"硬性约束：{r}"})
        self.session_pool[sid] = stack
        return stack

    def load_session_from_json(self, sid: str, json_path: str) -> LayeredContextStack:
        stack = LayeredContextStack()
        stack.load_history_from_json(json_path)
        self.session_pool[sid] = stack
        return stack

    def get_session(self, sid: str) -> LayeredContextStack:
        return self.session_pool.get(sid, None)

    def list_all_session(self):
        return list(self.session_pool.keys())


# ---------------- 交互演示入口 ----------------
if __name__ == "__main__":
    sm = SessionManager()
    print("===== 分层上下文Chat系统演示 =====")
    print("1. 导入JSON历史新建会话  2. 新建空白会话  3. 退出")
    op = input("请选择操作：")
    if op == "1":
        # 从demo历史文件加载会话
        sid = "test_task_01"
        stack = sm.load_session_from_json(sid, "history_demo.json")
        print(f"当前会话ID：{sid}，开始对话，输入exit退出\n")
        while True:
            query = input("你：")
            if query.strip() == "exit":
                break
            res = stack.chat(query)
            print(f"助手：{res}\n")
    elif op == "2":
        sid = "new_empty_chat"
        role = "你是CIFAR10图像分类训练专属助手，只解答CNN模型训练相关问题"
        rules = [
            "回答必须附带训练指标说明",
            "禁止回答游戏、影视、爬虫等无关内容",
            "不得编造数据集、CUDA报错解决方案"
        ]
        stack = sm.create_empty_session(sid, role, rules)
        print(f"空白会话创建完成，输入exit退出\n")
        while True:
            query = input("你：")
            if query.strip() == "exit":
                break
            res = stack.chat(query)
            print(f"助手：{res}\n")
    else:
        print("程序退出")