from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama as Ollama
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from config import OLLAMA_URL, MODEL_NAME, SYSTEM_PROMPT
from tools import TOOLS


# --- 状态定义 ---
class AgentState(BaseModel):
    input: str
    command: dict
    tool_calls: list
    output: str
    intermediate_steps: List[str]

# --- 初始化 LLM ---
try:
    #判断是否调用工具的LLM
    llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_URL, temperature=0.0)
    llm_with_tools = llm.bind_tools(TOOLS)

    #用于总结的LLM
    llm_synthesize = Ollama(model=MODEL_NAME, base_url=OLLAMA_URL, temperature=0.2)
except Exception as e:
    print(f"⚠️ 初始化失败：{e}")
    llm_with_tools = None
    llm_synthesize = None

# --- 节点定义 ---

#分析需求
def analyst_node(state_analyst: AgentState) -> AgentState:
    if llm_with_tools is None:
        return AgentState(
            input="",
            tool_calls=[],
            command={},
            output="LLM 初始化失败",
            intermediate_steps=[],
        )

    print("\n[AGENT] 正在分析用户请求...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", state_analyst.input)
    ])
    response = llm_with_tools.invoke(prompt.invoke({}))
    steps = state_analyst.intermediate_steps.copy()

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        steps.append(f"LLM 建议调用工具: {tool_call['name']}")
        return AgentState(
            input=state_analyst.input,
            tool_calls=response.tool_calls,
            command=dict(tool_call),
            output="",
            intermediate_steps=steps
        )
    else:
        return AgentState(
            input=state_analyst.input,
            tool_calls=[],
            command={},
            output=response.content,
            intermediate_steps=steps
        )

#调用工具
def call_tool_node(state_call_tool: AgentState) -> AgentState:
    tool_call = state_call_tool.command
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_func = next((t for t in TOOLS if t.name == tool_name), None)
    if not tool_func:
        return AgentState(
            input="",
            tool_calls=[],
            command={},
            output=f"错误：找不到工具 {tool_name}",
            intermediate_steps=[]
        )

    print(f"待执行的操作: {tool_name}，参数: {tool_args}")
    if input("是否确定执行此操作？(yes/no): ").strip().lower() == 'yes':
        try:
            result = tool_func.invoke(tool_args)
            print(f"操作 `{tool_name}` 执行成功，准备总结。")
            return AgentState(
                input=state_call_tool.input,
                tool_calls=state_call_tool.tool_calls,
                command=state_call_tool.command,
                output=result,
                intermediate_steps=state_call_tool.intermediate_steps
            )
        except Exception as ex:
            return AgentState(
                input=state_call_tool.input,
                tool_calls=state_call_tool.tool_calls,
                command=state_call_tool.command,
                output=f"工具执行出错: {ex}",
                intermediate_steps=state_call_tool.intermediate_steps
            )
    else:
        return AgentState(
            input=state_call_tool.input,
            tool_calls=state_call_tool.tool_calls,
            command=state_call_tool.command,
            output="操作被用户取消。",
            intermediate_steps=state_call_tool.intermediate_steps
        )

#综合总结
def synthesizer_node(state_synthesizer: AgentState) -> AgentState:
    if llm_synthesize is None:
        return AgentState(
            input="",
            tool_calls=[],
            command={},
            output="总结 LLM 初始化失败",
            intermediate_steps=[]
        )
    print("\n[AGENT] 正在总结信息并生成回答...")
    retrieval_result = state_synthesizer.output
    original_input = state_synthesizer.input
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", f"原始问题: {original_input}\n\n--- 参考材料 ---\n{retrieval_result}")
    ])
    response = llm_synthesize.invoke(synthesis_prompt.invoke({}))
    return AgentState(
        input=state_synthesizer.input,
        tool_calls=[],
        command={},
        output=response.content,
        intermediate_steps=state_synthesizer.intermediate_steps
    )

# --- 图构建 ---
def initialize_graph():
    if llm_with_tools is None: return None
    workflow = StateGraph(AgentState)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("tool_executor", call_tool_node)
    workflow.add_node("synthesizer", synthesizer_node) # 新增总结节点
    workflow.set_entry_point("analyst")

    workflow.add_conditional_edges(
        "analyst",
        lambda s: "tool_executor" if s.tool_calls else END,
        {"tool_executor": "tool_executor", END: END}
    )
    workflow.add_edge("tool_executor", "synthesizer")
    workflow.add_edge("synthesizer", END)
    return workflow.compile()

app = initialize_graph()
if __name__ == "__main__":
    print(f"--- openEuler 教学助理 ({MODEL_NAME}) ---")
    while True:
        q = input("\n[AGENT]请问你想问什么？ (exit退出): ")
        if q.lower() == 'exit': break
        initial_state=AgentState(
            input=q,
            tool_calls=[],
            command={},
            output="",
            intermediate_steps=[]
        )
        try:
            state = app.invoke(initial_state)
            if state and state.get('output'):
                print("\n========== 回答 ==========")
                print(state['output'])
        except Exception as e:
            print(f"\n出错:{e}")
