from typing import Dict, TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os
import logging
import json
from datetime import datetime
import re
from functools import lru_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

groq_key = os.getenv('groq_api_key')
gemini_key = os.getenv('gemini_api_key')
cohere_key = os.getenv('cohere_api_key')
tavily_key = os.getenv('tavily_api_key')
if not all([groq_key, gemini_key, cohere_key, tavily_key]):
    raise ValueError("Missing one or more API keys in environment variables")

llm_grok = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    api_key=groq_key,
    temperature=0.6
)
llm_gpt = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_key,
    temperature=0.6
)
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=gemini_key,
    temperature=0.6
)
llm_cohere = ChatCohere(
    api_key=cohere_key,
    temperature=0.6
)

tavily = TavilySearch(max_results=3, api_key=tavily_key)
tools = [tavily]
llm_grok_with_tools = llm_grok.bind_tools(tools)

def get_weather_data(location: str) -> Dict:
    try:
        return {
            "location": location,
            "temperature": "25°C",
            "precipitation": "Moderate",
            "forecast": "Sunny with occasional showers",
            "humidity": "60%"
        }
    except Exception as e:
        logger.error(f"Error fetching weather data: {e}")
        return {"error": "Failed to fetch weather data"}

def get_soil_data(location: str) -> Dict:
    try:
        return {
            "location": location,
            "soil_type": "Loamy",
            "ph_level": 6.5,
            "nutrients": {"N": "High", "P": "Medium", "K": "Low"},
            "moisture": "Adequate"
        }
    except Exception as e:
        logger.error(f"Error fetching soil data: {e}")
        return {"error": "Failed to fetch soil data"}

def sanitize_text(text):
    if not isinstance(text, str):
        return text
    text = text.replace('\u2011', '-')
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def create_crop_planning_workflow(location: str, max_iterations: int = 3):
    class CropPlanState(TypedDict):
        location: str
        weather_data: Optional[Dict]
        soil_data: Optional[Dict]
        crop_plan: Optional[str]
        history: List[str]
        iteration: int
        final_plan: Optional[str]

    workflow = StateGraph(CropPlanState)

    data_collection_prompt = (
        "You are an agricultural expert tasked with collecting data for crop planning in {location}. "
        "Use available tools to gather relevant information about crop suitability, considering weather and soil data: {weather_data}, {soil_data}. "
        "Provide a concise summary of the data (one paragraph) to inform crop selection."
    )

    planning_prompt = (
        "You are an agricultural planner for {location}. "
        "Based on the data summary: {data_summary}, and conversation history: {history}, "
        "propose a crop plan (one paragraph) including specific crops, planting schedules, and considerations, "
        "ensuring it aligns with the weather and soil conditions and counters or builds on previous plans."
    )

    summarizer_prompt = (
        "You are a summarizer agent. "
        "Given the conversation history: {history}, "
        "create a concise, final crop plan (one paragraph) in JSON format, selecting the best elements from all proposals, "
        "considering weather: {weather_data} and soil data: {soil_data} for {location}. "
        "Ensure the output is valid JSON without Markdown code fences (e.g., ```json). "
        "Return only the JSON string."
    )

    @lru_cache(maxsize=100)
    def collect_data(location: str, weather_data: str, soil_data: str) -> str:
        try:
            prompt = data_collection_prompt.format(location=location, weather_data=weather_data, soil_data=soil_data)
            result = sanitize_text(llm_grok_with_tools.invoke(prompt).content)
            logger.info(f"Data collection result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in data collection: {e}")
            return "Error collecting data"

    def data_collection_node(state: CropPlanState) -> CropPlanState:
        weather_data = get_weather_data(state['location'])
        soil_data = get_soil_data(state['location'])
        data_summary = collect_data(
            state['location'],
            json.dumps(weather_data),
            json.dumps(soil_data)
        )
        return {
            "weather_data": weather_data,
            "soil_data": soil_data,
            "history": state['history'] + [f"Data Summary: {data_summary}"],
            "iteration": state['iteration']
        }

    def grok_planning_node(state: CropPlanState) -> CropPlanState:
        try:
            data_summary = state['history'][-1] if state['history'] else "No data available"
            history = "\n".join(state['history'])
            prompt = planning_prompt.format(location=state['location'], data_summary=data_summary, history=history)
            plan = f"Grok Plan: {sanitize_text(llm_grok.invoke(prompt).content)}"
            logger.info(f"Grok plan: {plan}")
            return {
                "crop_plan": plan,
                "history": state['history'] + [plan],
                "iteration": state['iteration'] + 1
            }
        except Exception as e:
            logger.error(f"Error in Grok planning: {e}")
            return {
                "crop_plan": "Error generating Grok plan",
                "history": state['history'] + ["Grok Plan: Error"],
                "iteration": state['iteration'] + 1
            }

    def gemini_planning_node(state: CropPlanState) -> CropPlanState:
        try:
            data_summary = state['history'][-1] if state['history'] else "No data available"
            history = "\n".join(state['history'])
            prompt = planning_prompt.format(location=state['location'], data_summary=data_summary, history=history)
            plan = f"Gemini Plan: {sanitize_text(llm_gemini.invoke(prompt).content)}"
            logger.info(f"Gemini plan: {plan}")
            return {
                "crop_plan": plan,
                "history": state['history'] + [plan],
                "iteration": state['iteration'] + 1
            }
        except Exception as e:
            logger.error(f"Error in Gemini planning: {e}")
            return {
                "crop_plan": "Error generating Gemini plan",
                "history": state['history'] + ["Gemini Plan: Error"],
                "iteration": state['iteration'] + 1
            }

    def gpt_planning_node(state: CropPlanState) -> CropPlanState:
        try:
            data_summary = state['history'][-1] if state['history'] else "No data available"
            history = "\n".join(state['history'])
            prompt = planning_prompt.format(location=state['location'], data_summary=data_summary, history=history)
            plan = f"GPT Plan: {sanitize_text(llm_gpt.invoke(prompt).content)}"
            logger.info(f"GPT plan: {plan}")
            return {
                "crop_plan": plan,
                "history": state['history'] + [plan],
                "iteration": state['iteration'] + 1
            }
        except Exception as e:
            logger.error(f"Error in GPT planning: {e}")
            return {
                "crop_plan": "Error generating GPT plan",
                "history": state['history'] + ["GPT Plan: Error"],
                "iteration": state['iteration'] + 1
            }

    def summarizer_node(state: CropPlanState) -> CropPlanState:
        try:
            history = "\n".join(state['history'])
            prompt = summarizer_prompt.format(
                history=history,
                weather_data=json.dumps(state['weather_data']),
                soil_data=json.dumps(state['soil_data']),
                location=state['location']
            )
            final_plan = sanitize_text(llm_cohere.invoke(prompt).content)
            final_plan = re.sub(r'^```json\s*|\s*```$', '', final_plan, flags=re.MULTILINE).strip()
            logger.info(f"Final plan: {final_plan}")
            return {"final_plan": final_plan}
        except Exception as e:
            logger.error(f"Error in summarizer: {e}")
            return {"final_plan": "Error generating final plan"}

    workflow.add_node("data_collection", data_collection_node)
    workflow.add_node("grok_planning", grok_planning_node)
    workflow.add_node("gemini_planning", gemini_planning_node)
    workflow.add_node("gpt_planning", gpt_planning_node)
    workflow.add_node("summarizer", summarizer_node)

    workflow.set_entry_point("data_collection")
    workflow.add_edge("data_collection", "grok_planning")
    workflow.add_edge("grok_planning", "gemini_planning")
    workflow.add_edge("gemini_planning", "gpt_planning")

    def check_iterations(state: CropPlanState) -> str:
        logger.debug(f"Iteration count: {state['iteration']}/{max_iterations}")
        return "summarizer" if state['iteration'] >= max_iterations else "grok_planning"

    workflow.add_conditional_edges(
        "gpt_planning",
        check_iterations,
        {"summarizer": "summarizer", "grok_planning": "grok_planning"}
    )
    workflow.add_edge("summarizer", END)

    return workflow

location = "Punjab, India"
try:
    workflow = create_crop_planning_workflow(location=location, max_iterations=3)
    app = workflow.compile()

    try:
        from IPython.display import Image, display
        png_bytes = app.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_bytes)
        print("Graph saved as graph.png")
    except Exception as e:
        logger.error(f"Error generating graph diagram: {e}")
        print(f"Failed to generate graph diagram: {e}")

    initial_state = {
        "location": location,
        "weather_data": None,
        "soil_data": None,
        "crop_plan": None,
        "history": [],
        "iteration": 0,
        "final_plan": None
    }
    result = app.invoke(initial_state)

    history = result.get('history', [])
    final_plan = result.get('final_plan', 'No final plan generated')

    print("\nPlanning History:")
    for entry in history:
        print(f"- {entry}")
    print("\nFinal Crop Plan:")
    print(final_plan)

    crop_plan_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "location": location,
            "max_iterations": 3,
            "models": {
                "data_collection": "deepseek-r1-distill-llama-70b",
                "grok_planning": "deepseek-r1-distill-llama-70b",
                "gemini_planning": "gemini-2.5-flash",
                "gpt_planning": "openai/gpt-oss-120b",
                "summarizer": "cohere"
            }
        },
        "conversation": {
            "history": history,
            "final_plan": final_plan
        }
    }

    try:
        json_file = "crop_plan.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(crop_plan_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Crop plan saved to '{json_file}'")
    except Exception as e:
        logger.error(f"Error saving to JSON: {e}")
        print(f"Failed to save to JSON: {e}")

except Exception as e:
    logger.error(f"Error running crop planning: {e}")
    print(f"Failed to run crop planning: {e}")