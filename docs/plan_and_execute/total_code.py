"""
Plan and Execute 실행 예제
- 참고 : https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/plan-and-execute/plan-and-execute.ipynb
"""

# 1. 환경변수 로드
from dotenv import load_dotenv
load_dotenv()


# 툴 정의
from langchain_community.tools.tavily_search import TavilySearchResults
tools = [TavilySearchResults(max_results=3)]

# llm 정의
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")

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
5. 실행 가능성: 각 단계는 주어진 도구들로 실제로 수행 가능해야 합니다
6. 완성도: 마지막 단계에서 사용자가 원하는 완전한 답변을 제공할 수 있어야 합니다

계획 작성 시 주의사항:
- 각 단계는 이전 단계에서 수집한 정보를 참조해야 합니다
- 단계 간 정보 전달이 명확해야 합니다
- 질문의 핵심 식별자(예: 특정 지역, 시간, 주제, 대상 등)를 각 단계에서 반드시 유지하세요
- 너무 많은 단계를 만들지 말고, 핵심적인 3-4단계로 구성하세요
- 마지막 단계는 반드시 최종 답변 생성을 위한 단계여야 합니다

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

# 노드 정의
from langgraph.graph import END

async def plan_execute_node(state: PlanExecute):
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
        for i, (task, result) in enumerate(past_steps, 1):
            context_info += f"\n{i}. {task}"
    
    # 작업 실행을 위한 프롬프트 구성
    task_formatted = f"""원래 질문: {original_input}

현재 수행할 작업: {current_task}

이전 단계에서 수행했던 작업: {context_info}

위의 정보를 참고하여 현재 작업을 수행해주세요."""

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

async def answer_node(state: PlanExecute):
    """남아있는 계획이 있는지 확인하고, 없으면 최종 답변을 생성합니다."""
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
        return "plan_execute_node"
    
# 그래프 정의
from langgraph.graph import StateGraph, START

graph_builder = StateGraph(PlanExecute)
graph_builder.add_node("plan_node", plan_node)
graph_builder.add_node("answer_node", answer_node)
graph_builder.add_node("plan_execute_node", plan_execute_node)

graph_builder.add_edge(START, "plan_node")
graph_builder.add_edge("plan_node", "plan_execute_node")
graph_builder.add_edge("plan_execute_node", "answer_node")

graph_builder.add_conditional_edges(
    "answer_node",
    should_end,
    ["plan_execute_node", END]
)

graph = graph_builder.compile()

# 그래프 이미지 저장
from utils.util import save_graph_image
save_graph_image(graph=graph,
                 filename="plan_and_execute_graph.png")

config = {"recursion_limit": 50}
inputs = {"input": "2025년 가장 핫한 부산 관광지 3개 소개해주세요."}


from rich.console import Console
from rich.panel import Panel
console = Console()
async def run_workflow():
    console.print()
    console.print(Panel(
        "[bold blue]🚀 워크플로우 시작...[/bold blue]",
        border_style="blue",
        padding=(1, 2),
        width=120,
        title="[bold]LangGraph Plan & Execute[/bold]",
        title_align="center"
    ), justify="center")
    console.print()
    async for event in graph.astream(inputs, config=config):
        console.print("─" * 120)
        console.print()
        console.print(event)
        console.print()

# 비동기 실행
import asyncio
asyncio.run(run_workflow())