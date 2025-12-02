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


# --- 1. 基础 Shell 工具 (保持不变) ---
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
            cwd="/"
        )
        return result.stdout.strip()
    except Exception as ex:
        return f"命令执行过程中发生错误: {ex}"

@tool
def clear_system_cache() -> str:
    """
    执行清理 openEuler 系统缓存的操作，用于释放内存。
    这个命令具有轻微副作用（短暂延迟），但通常是安全的。
    """
    try:
        subprocess.run("sync", shell=True, check=True)
        return "系统缓存清理操作已尝试执行。建议检查内存使用情况。"
    except Exception as ex:
        return f"缓存清理工具执行失败: {ex}"

# --- 2. C 语言教学演示工具 (保持不变) ---
@tool
def run_teaching_demo(program_name: str, arguments: str = "") -> str:
    """
    运行预先编译好的 C 语言操作系统教学演示程序（例如分页算法模拟、调度算法演示）。
    参数:
    program_name: 教学程序的可执行文件名（例如 'paging_sim', 'scheduler_demo'）。
    arguments: 传递给程序的参数。
    """
    # 假设编译好的 C 程序都放在这个目录下
    DEMO_PATH = "/opt/teaching_demos/"
    full_command = f"{DEMO_PATH}{program_name} {arguments}".strip()

    print(f"\n[AGENT] 尝试运行 C 语言教学演示: {full_command}")
    try:
        # 检查文件是否存在 (模拟)
        check_cmd = f"ls {DEMO_PATH}{program_name}"
        subprocess.run(check_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 运行程序
        result = subprocess.run(
            full_command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        return f"C 语言演示程序 {program_name} 运行输出:\n{result.stdout.strip()}"
    except Exception as ex:
        return f"运行演示程序时发生错误: {ex}"

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

TOOLS = [execute_safe_shell, clear_system_cache, run_teaching_demo, search_knowledge_base]
