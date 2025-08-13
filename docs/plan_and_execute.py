"""
Plan and Execute 실행 예제
- 참고 : https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/plan-and-execute/plan-and-execute.ipynb
"""

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()


# 툴 정의
from langchain_community.tools.tavily_search import TavilySearchResults
tools = [TavilySearchResults(max_results=3)]

# llm 정의
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

# agent 정의
from langgraph.prebuilt import create_react_agent
agent_executor = create_react_agent(llm, tools, prompt="You are a helpful assistant that can answer questions and help with tasks.")

# 단일실행
# result = agent_executor.invoke({"messages": [("user", "부산 관광지 3개 소개해주세요.")]})
# print(result)

# State 정의
import operator
from typing import Annotated, List, Tuple
from typing_extensions import TypedDict

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

# Step 별 정의 (1) Planning Step
from pydantic import BaseModel # python에서 데이터 검증, 구조 정의를 쉽게 할 수 있도록 도와주는 클래스
from pydantic import Field # 각 필드의 기본값, 설명, 검증 조건등을 설정하는 도우미 함수

class Plan(BaseModel):
    """계획을 제안하는 모델"""
    steps: List[str] = Field(description="수행해야 하는 작업 단계들을 나열한 리스트, 정렬된 순서여야 함.")

from langchain_core.prompts import ChatPromptTemplate
planner_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """당신은 복잡한 작업을 체계적으로 수행하는 전문가입니다. 주어진 목표를 달성하기 위해 구체적이고 실행 가능한 단계별 계획을 세워야 합니다.

중요한 계획 수립 원칙:
1. 구체성: 각 단계는 명확하고 구체적이어야 하며, 모호한 표현을 피하세요
2. 연속성: 각 단계는 이전 단계의 결과를 활용하여 다음 단계로 진행되어야 합니다
3. 맥락 유지: 모든 단계에서 원래 질문의 맥락과 키워드를 일관되게 유지하세요
4. 식별자 유지: 질문에서 추출된 핵심 식별자(지역명, 시간, 주제, 대상 등)를 모든 단계에서 일관되게 유지하세요
    - 지역명: "서울", "부산", "일본", "미국" 등
    - 시간: "2025년", "최근", "올해" 등  
    - 주제: "관광지", "음식", "기술", "경제" 등
    - 대상: "레스토랑", "호텔", "회사", "제품" 등
    - 수량: "3개", "5곳", "상위 10개" 등
5. 실행 가능성: 각 단계는 주어진 도구들로 실제로 수행 가능해야 합니다
6. 완성도: 마지막 단계에서 사용자가 원하는 완전한 답변을 제공할 수 있어야 합니다

계획 작성 시 주의사항:
- 각 단계는 이전 단계에서 수집한 정보를 참조해야 합니다
- 단계 간 정보 전달이 명확해야 합니다
- 질문의 핵심 식별자(예: 특정 지역, 시간, 주제, 대상 등)를 각 단계에서 반드시 유지하세요
- 너무 많은 단계를 만들지 말고, 핵심적인 3-4단계로 구성하세요
- 마지막 단계는 반드시 최종 답변 생성을 위한 단계여야 합니다

단계별 예시:
나쁜 단계별 예시는 모호한 표현을 사용하는 것입니다. 예를들어, "[주제] 정보를 조사합니다."
좋은 단계별 예시는 구체적으로 작성하는 것입니다. 예를들어, "1단계에서 수집한 [지역명] [주제] 정보를 바탕으로, 각 [대상]의 [특정 속성]을 조사합니다."

이제 주어진 목표를 달성하기 위한 구체적이고 실행 가능한 단계별 계획을 세워주세요."""
    ),
    (
        "placeholder", # messages 라는 변수값에 저장되는 대화 리스트 전체가 들어가도록 하는 역할
        "{messages}"
    ),
])
planner = planner_prompt | llm.with_structured_output(Plan)
# result = planner.invoke({"messages": [("user", "2025년 가장 핫한 부산 관광지 3개 소개해주세요.")]})
# print(result)

# Step 별 정의 (2) Re-Plan Step
from typing import Union # 여러 타입중 하나 가질수있음.

class Response(BaseModel):
    """사용자에게 답변을 제공하는 모델"""
    response: str
    
class Act(BaseModel):
    """수행할 작업을 정의하는 모델"""
    action: Union[Response, Plan] = Field(
        description="수행할 작업입니다. 사용자에게 바로 응답하려면 Response를 사용하세요. 정답을 얻기 위해 도구를 추가로 사용해야 한다면 Plan을 사용하세요"
    )
    
replanner_prompt = ChatPromptTemplate.from_template(
    """현재까지의 진행 상황을 검토하고 다음 단계를 결정하세요.

당신의 목표는 다음과 같습니다:
{input}

원래 계획은 다음과 같습니다:
{plan}

현재까지 수행 완료한 단계들은 다음과 같습니다:
{past_steps}

남은 계획 단계들을 확인하세요. 만약 모든 단계가 완료되었거나 충분한 정보를 수집했다면, 
사용자에게 최종 답변을 제공하세요. 그렇지 않다면 다음 단계를 계속 진행하세요.

중요: 이미 완료된 단계는 다시 계획하지 마세요. 원래 계획을 그대로 유지하면서 
아직 완료되지 않은 단계만 남겨두세요."""
)

replanner = replanner_prompt | llm.with_structured_output(Act)

# 노드 정의
from typing import Literal
from langgraph.graph import END

async def execute_node(state: PlanExecute):
    """현재 계획의 첫 번째 단계를 실행합니다."""
    plan = state.get("plan", [])
    if not plan:
        return {"response": "모든 단계가 완료되었습니다."}
    
    # 현재 실행할 작업
    current_task = plan[0]
    past_steps = state.get("past_steps", [])
    original_input = state.get("input", "")
    
    # 이전 단계들의 결과를 정리
    context_info = ""
    if past_steps:
        context_info = "\n\n이전 단계에서 수집한 정보:\n"
        for i, (task, result) in enumerate(past_steps, 1):
            context_info += f"\n{i}. {task}\n결과: {result}\n"
    
    # 작업 실행을 위한 프롬프트 구성
    task_formatted = f"""원래 질문: {original_input}

현재 수행할 작업: {current_task}{context_info}

위의 정보를 참고하여 현재 작업을 수행해주세요. 
이전 단계에서 수집한 정보가 있다면 그것을 활용하여 더 구체적이고 유용한 결과를 제공해주세요.
특히 원래 질문의 핵심 식별자(지역명, 시간, 주제, 대상 등)를 유지하면서 작업을 수행해주세요."""

    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    
    # 완료된 작업을 plan에서 제거하고 past_steps에 추가
    remaining_plan = plan[1:] if len(plan) > 1 else []
    
    return {
        "past_steps": [(current_task, agent_response["messages"][-1].content)],
        "plan": remaining_plan
    }

async def plan_node(state: PlanExecute):
    """초기 계획을 생성합니다."""
    plan = await planner.ainvoke({"messages":[("user",state["input"])]})
    return {"plan": plan.steps}

async def replan_node(state: PlanExecute):
    """계획의 진행 상황을 확인하고 다음 단계를 결정합니다."""
    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])
    original_input = state.get("input", "")
    
    # 모든 단계가 완료되었으면 LLM을 사용해 최종 응답 생성
    if not plan:
        # 수집된 정보를 정리
        collected_info = "\n\n".join([f"단계 {i+1}: {task}\n결과: {result}" 
                                     for i, (task, result) in enumerate(past_steps)])
        
        # LLM에게 최종 응답 생성 요청
        final_prompt = f"""다음은 사용자의 질문에 대한 정보 수집 과정입니다:

사용자 질문: {original_input}

수집된 정보:
{collected_info}

위의 수집된 정보를 바탕으로, 사용자의 원래 질문에 대한 깔끔하고 유용한 최종 답변을 작성해주세요. 
수집된 정보를 종합하여 논리적이고 실용적인 답변을 제공해주세요."""

        final_response = await llm.ainvoke([("user", final_prompt)])
        return {"response": final_response.content}
    
    # 아직 남은 단계가 있으면 계속 진행
    return {"plan": plan}

def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "execute_node"
    
# 그래프 정의
from langgraph.graph import StateGraph, START

workflow = StateGraph(PlanExecute)
workflow.add_node("plan_node", plan_node)
workflow.add_node("replan_node", replan_node)
workflow.add_node("execute_node", execute_node)

workflow.add_edge(START, "plan_node")
workflow.add_edge("plan_node", "execute_node")
workflow.add_edge("execute_node", "replan_node")

workflow.add_conditional_edges(
    "replan_node",
    should_end,
    ["execute_node", END]
)

app = workflow.compile()

# 그래프 확인 및 이미지 저장
from IPython.display import Image, display
import os

# 이미지 저장
graph_image = app.get_graph(xray=True).draw_mermaid_png()
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
image_path = os.path.join(output_dir, "plan_and_execute_graph.png")

with open(image_path, "wb") as f:
    f.write(graph_image)

print(f"그래프 이미지가 저장되었습니다: {image_path}")

# Jupyter 환경에서도 표시 (선택사항)
display(Image(graph_image))

config = {"recursion_limit": 50}
inputs = {"input": "2025년 가장 핫한 부산 관광지 3개 소개해주세요."}

# rich 라이브러리 import
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box

console = Console()

# 비동기 실행을 위한 함수
async def run_workflow():
    console.print("[bold blue]🚀 워크플로우 시작...[/bold blue]")
    console.print()
    
    async for event in app.astream(inputs, config=config):
        
        # 각 노드의 결과를 처리
        for node_name, node_data in event.items():
            if node_name == "__end__":
                continue
                
            console.print(f"[bold cyan] 노드: {node_name}[/bold cyan]")
            
            # 노드 데이터 내부의 키들을 확인
            for data_key, data_value in node_data.items():
                if data_key == "plan" and data_value:
                    table = Table(title="📋 실행 계획", box=box.ROUNDED)
                    table.add_column("단계", style="cyan", no_wrap=True)
                    table.add_column("내용", style="white")
                    
                    for i, step in enumerate(data_value, 1):
                        table.add_row(f"{i}단계", step)
                    
                    console.print(table)
                    console.print()  # 빈 줄 추가
                
                elif data_key == "past_steps" and data_value:
                    for i, (task, result) in enumerate(data_value, 1):
                        # 작업 제목
                        task_text = Text(f"🔍 단계 {i}: {task}", style="bold green")
                        console.print(Panel(task_text, title="수행 작업", border_style="green"))
                        
                        # 결과 내용 (긴 텍스트는 줄바꿈 처리)
                        result_text = Text(result, style="white")
                        console.print(Panel(result_text, title="수행 결과", border_style="blue"))
                        console.print()  # 빈 줄 추가
                
                elif data_key == "response" and data_value:
                    response_text = Text(data_value, style="white")
                    console.print(Panel(response_text, title="🎯 최종 답변", border_style="yellow", style="bold"))
                    console.print()  # 빈 줄 추가
    
    console.print("[bold green]✅ 워크플로우 완료![/bold green]")

# 비동기 실행
import asyncio
asyncio.run(run_workflow())