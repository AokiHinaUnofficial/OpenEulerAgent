import os
import subprocess

import ragflow_sdk
from langchain.tools import tool

from config import RAGFLOW_API_BASE, RAGFLOW_API_KEY, RAGFLOW_DATASET_ID

# --- RAGFlow 客户端初始化 ---
RAGFLOW_CLASS = ragflow_sdk.RAGFlow
RAGFLOW_CLIENT = None
if RAGFLOW_CLASS is not None:
    try:
        RAGFLOW_CLIENT = RAGFLOW_CLASS(
            base_url=RAGFLOW_API_BASE,
            api_key=RAGFLOW_API_KEY
        )
    except Exception as e:
        print(f"RAGFlow 客户端初始化失败 (请检查 API BASE 和 KEY): {e}")


# --- 1. 基础 Shell 工具 ---
@tool
def execute_safe_shell(command: str) -> str:
    """
    安全地执行一条 openEuler命令，并返回其结果。
    仅用于执行低风险的查询和状态检查命令 (如 ls, df, free, uptime, cat /etc/os-release, ps)。
    禁止执行修改系统状态的命令 (如 rm, mv, reboot 等)。
    """
    print(f"\n[AGENT] 尝试执行命令: {command}")
    dangerous_keywords = ['rm', 'mv', 'reboot', 'shutdown', 'systemctl', 'useradd', 'passwd', 'mkfs', 'chown', 'chmod', 'dd']
    if any(keyword in command for keyword in dangerous_keywords):
        return "ERROR: 拒绝执行此命令。此工具仅允许执行查询或状态检查等安全命令。"

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception as ex:
        return f"命令执行过程中发生错误: {ex}"

# --- 2. C 语言教学演示工具 ---
@tool
def run_teaching_demo(program_name: str, arguments: str = "") -> str:
    """
    运行一个已经编译和链接好的 C 语言教学演示程序。
    流程：
    1. 根据 program_name 确定可执行文件的名称。
    2. 查找并确保该可执行文件存在。
    3. 运行该可执行文件并返回输出。
    参数:
    program_name: 可执行文件或源文件的名称（例如 'producer-consumer' 或 'producer-consumer.c'）。
    arguments: 传递给程序的运行时参数。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DEMO_PATH = os.path.abspath(os.path.join(current_dir, "../OpenEuler基础"))
    clean_name = program_name.lower().replace(".c", "").replace("-", "_")
    exe_name_kebab = clean_name.replace('_', '-')
    exe_name_snake = clean_name
    if os.path.exists(os.path.join(DEMO_PATH, exe_name_kebab)):
        exe_full_path = os.path.join(DEMO_PATH, exe_name_kebab)
        exe_name = exe_name_kebab
    elif os.path.exists(os.path.join(DEMO_PATH, exe_name_snake)):
        exe_full_path = os.path.join(DEMO_PATH, exe_name_snake)
        exe_name = exe_name_snake
    else:
        return (f"错误：在目录中找不到与 '{clean_name}' 匹配的已编译可执行文件。\n"
                f"请确保文件 '{exe_name_kebab}' 或 '{exe_name_snake}' 存在于 {DEMO_PATH} 中。")

    print(f"[AGENT] 确定可执行文件: '{exe_name}'")
    try:
        subprocess.run(
            ["chmod", "+x", exe_full_path],
            check=True,
            capture_output=True,
            text=True
        )
    except Exception as ex:
        return f"错误：无法设置可执行文件权限 ({exe_name})。请检查用户权限设置: {ex}"
    try:
        run_cmd = f"stdbuf -oL {exe_full_path} {arguments}"
        print(f"[AGENT] 准备执行：{run_cmd}")
        result = subprocess.run(
            run_cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        return f"程序运行成功。\n\n程序输出:\n{result.stdout.strip()}"
    except subprocess.TimeoutExpired as ex:
        stdout_output = ex.stdout.strip() if ex.stdout else "无输出内容"
        stderr_output = ex.stderr.strip() if ex.stderr else "无错误信息"
        return (f"程序运行完成（触发超时保护，这对无限循环程序是正常的）。\n"
                f"在 {ex.timeout} 秒内的运行输出如下:\n\n"
                f"--- 标准输出 (stdout) ---\n{stdout_output}\n\n"
                f"--- 标准错误 (stderr) ---\n{stderr_output}")
    except subprocess.CalledProcessError as ex:
        return (f"程序运行时崩溃 (非零退出)。\n"
                f"错误信息 (Stderr):\n{ex.stderr}")
    except Exception as ex:
        return f"运行阶段发生未知错误: {ex}"

# --- 3. RAGFlow 知识库检索工具 ---
@tool
def search_knowledge_base(question: str) -> str:
    """
    使用 RAGFlow 检索 openEuler 知识库。
    用于查询具体的系统配置、报错信息或文档细节。
    """
    if RAGFLOW_CLIENT is None:
        return "RAGFlow 客户端未成功初始化，无法执行检索。"
    try:
        response = RAGFLOW_CLIENT.retrieve(
            question=question,
            dataset_ids=[RAGFLOW_DATASET_ID],
            top_k=3,
            similarity_threshold=0.5
        )
        chunks_data = []
        if isinstance(response, list):
            chunks_data = response
        elif isinstance(response, dict):
            chunks_data = response.get('data', {}).get('chunks', []) # 字典里是 dicts 或 Chunk objects
        chunks = []
        if isinstance(chunks_data, list):
            for item in chunks_data:
                try:
                    content = item.content
                    if content:
                        chunks.append(content)
                except AttributeError:
                    if isinstance(item, dict) and item.get('content'):
                        chunks.append(item.get('content'))
        # === 结果处理 ===
        if chunks:
            result_text = "\n\n--- 检索到的参考资料 ---\n".join(chunks)
            return f"查询成功，参考资料如下:\n{result_text}"
        else:
            return "知识库未找到与该问题高度相关的参考资料。"
    except Exception as ex:
        return f"RAGFlow 调用失败: {ex}"

TOOLS = [execute_safe_shell, run_teaching_demo, search_knowledge_base]
