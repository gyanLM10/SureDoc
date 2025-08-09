from typing import Literal, Any
from typing_extensions import TypedDict, Annotated
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.schema import OutputParserException
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langgraph.graph import START, StateGraph, END
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent
from prompt_library.prompt import system_prompt
from utils.llms import LLMModel
from Toolkit.toolkits import (
    check_availability_by_doctor,
    check_availability_by_specialization,
    set_appointment,
    cancel_appointment,
    reschedule_appointment
)

# ======== Schema for Supervisor Routing ========
class Router(BaseModel):
    next: Literal['information_node', 'booking_node', 'FINISH']
    reasoning: str

# ======== Shared State Across Nodes ========
class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    id_number: int
    next: str
    query: str
    current_reasoning: str
    step_count: int  # NEW: loop safety counter

# ======== Main Agent ========
class DoctorAppointmentAgent:
    def __init__(self):
        llm_model = LLMModel()
        self.model = llm_model.get_model()
        self.parser = PydanticOutputParser(pydantic_object=Router)

    def supervisor_node(self, state: AgentState) -> Command[Literal['information_node', 'booking_node', '__end__']]:
        print("\n[Supervisor] State:", state)

        # Stop if loop count exceeds threshold
        step_count = state.get("step_count", 0) + 1
        if step_count > 6:  # limit to avoid infinite recursion
            print("[Supervisor] Step limit reached — ending conversation.")
            return Command(goto=END, update={"step_count": step_count})

        # Strict routing instructions
        system_instructions = f"""
        You are a strict JSON generator for routing requests.
        Always respond ONLY with a JSON object that matches this schema:
        {self.parser.get_format_instructions()}
        """

        messages = [
            SystemMessage(content=system_instructions),
            HumanMessage(content=f"user's identification number is {state['id_number']}")
        ] + state["messages"]

        query = state['messages'][0].content if len(state['messages']) == 1 else ""

        raw_response = self.model.invoke(messages)
        print("[Supervisor] Raw output:", raw_response)

        try:
            response = self.parser.parse(raw_response.content)
        except OutputParserException as e:
            print("[Supervisor] Parsing failed:", e)
            # End instead of infinite loop on repeated parse failures
            return Command(goto=END, update={"step_count": step_count})

        goto = response.next
        if goto == "FINISH":
            goto = END

        update_data = {
            'next': goto,
            'current_reasoning': response.reasoning,
            'step_count': step_count
        }

        if query:
            update_data['query'] = query
            update_data['messages'] = [
                HumanMessage(content=f"user's identification number is {state['id_number']}")
            ]

        return Command(goto=goto, update=update_data)

    def information_node(self, state: AgentState) -> Command:
        print("[Information Node] Running")

        system_prompt_text = (
            "You are a specialized hospital information agent. "
            "You can provide details about doctor availability or answer FAQs about the hospital. "
            "Ask for missing details politely. Assume the year is 2024."
        )

        system_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_text),
            ("placeholder", "{messages}"),
        ])

        messages = [
            HumanMessage(content=f"user's identification number is {state['id_number']}")
        ] + state["messages"]

        information_agent = create_react_agent(
            model=self.model,
            tools=[check_availability_by_doctor, check_availability_by_specialization],
            prompt=system_prompt
        )

        result = information_agent.invoke({"messages": messages})
        last_message_content = (
            result["messages"][-1].content
            if "messages" in result else result.get("output", "")
        )

        # Example: End if no further question is needed
        if "anything else" not in last_message_content.lower():
            return Command(
                update={"messages": state["messages"] + [AIMessage(content=last_message_content, name="information_node")]},
                goto=END
            )

        return Command(
            update={"messages": state["messages"] + [AIMessage(content=last_message_content, name="information_node")]},
            goto="supervisor"
        )

    def booking_node(self, state: AgentState) -> Command:
        print("[Booking Node] Running")

        system_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a specialized agent to set, cancel, or reschedule an appointment. "
             "Ask politely for missing details. Assume the year is 2024."),
            ("placeholder", "{messages}")
        ])

        booking_agent = create_react_agent(
            model=self.model,
            tools=[set_appointment, cancel_appointment, reschedule_appointment],
            prompt=system_prompt
        )

        result = booking_agent.invoke({"messages": state["messages"]})
        latest_message = (
            result["messages"][-1].content
            if isinstance(result, dict) and "messages" in result
            else getattr(result, "content", str(result))
        )

        # Example: End if booking is confirmed
        if "appointment confirmed" in latest_message.lower():
            return Command(
                update={"messages": state["messages"] + [AIMessage(content=latest_message, name="booking_node")]},
                goto=END
            )

        return Command(
            update={"messages": state["messages"] + [AIMessage(content=latest_message, name="booking_node")]},
            goto="supervisor"
        )

    def workflow(self):
        self.graph = StateGraph(AgentState)
        self.graph.add_node("supervisor", self.supervisor_node)
        self.graph.add_node("information_node", self.information_node)
        self.graph.add_node("booking_node", self.booking_node)
        self.graph.add_edge(START, "supervisor")
        self.app = self.graph.compile()
        return self.app
