from typing import Dict, TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain_community.tools.google_serper import GoogleSerperResults
from dotenv import load_dotenv
import os
import logging
import json
from datetime import datetime
import re
from functools import lru_cache
from IPython.display import Image, display

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

groq_key = os.getenv('groq_api_key')
gemini_key = os.getenv('gemini_api_key')
cohere_key = os.getenv('cohere_api_key')
serper_key = os.getenv('serper_api_key')
if not all([groq_key, gemini_key, cohere_key, serper_key]):
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

serper = GoogleSerperResults(api_key=serper_key, num_results=3)

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

def create_crop_planning_workflow(location: str, plan_duration: int, max_iterations: int = 3):
    class CropPlanState(TypedDict):
        location: str
        plan_duration: int
        soil_data: Optional[Dict]
        crop_plan: Optional[str]
        history: List[str]
        iteration: int
        final_plan: Optional[str]

    workflow = StateGraph(CropPlanState)

    data_collection_prompt = (
        "You are an agricultural expert tasked with summarizing crop suitability for {location} over a {plan_duration}-month period. "
        "Use the provided search results: {search_results}, and soil data: {soil_data}. "
        "If search results are unavailable or irrelevant, use your knowledge to summarize crop suitability based on the soil data "
        "(loamy soil, pH 6.5, high nitrogen, medium phosphorus, low potassium, adequate moisture). "
        "Provide a concise summary (one paragraph, 100-150 words) of suitable crops and considerations for the specified duration."
    )

    planning_prompt = (
        "You are an agricultural planner for {location} over a {plan_duration}-month period. "
        "Based on the data summary: {data_summary}, and conversation history: {history}, "
        "propose a crop plan (one paragraph) including specific crops, planting schedules, and considerations, "
        "ensuring it aligns with the soil conditions and plan duration, and counters or builds on previous plans."
    )

    summarizer_prompt = (
        "You are a summarizer agent. "
        "Given the conversation history: {history}, "
        "create a concise, final crop plan (one paragraph) in JSON format for a {plan_duration}-month period, "
        "selecting the best elements from all proposals, considering soil data: {soil_data} for {location}. "
        "Ensure the output is valid JSON without Markdown code fences (e.g., ```json). "
        "Return only the JSON string."
    )

    @lru_cache(maxsize=100)
    def collect_data(location: str, plan_duration: int, soil_data: str) -> str:
        try:
            query = f"crop suitability in {location} for {plan_duration}-month period"
            search_results = serper.invoke({"query": query})
            prompt = data_collection_prompt.format(
                location=location,
                plan_duration=plan_duration,
                search_results=json.dumps(search_results),
                soil_data=soil_data
            )
            result = llm_gemini.invoke(prompt).content
            result = sanitize_text(result)
            if not result or result.strip() == "" or "error" in result.lower():
                default_summary = (
                    f"In {location} for a {plan_duration}-month period, the loamy soil with a pH of 6.5, high nitrogen, medium phosphorus, "
                    f"and low potassium supports crops like wheat, rice, chickpeas, and mustard. Adequate moisture levels suit these crops, "
                    f"but potassium supplementation is advised to optimize yields. Crop rotation with legumes like chickpeas enhances soil nitrogen, "
                    f"while mustard improves soil structure. For a {plan_duration}-month plan, prioritize crops like wheat and chickpeas for rabi season "
                    f"or rice for kharif season, depending on the timeframe, with fertilizers to address potassium deficiency."
                )
                logger.warning("No valid search results; using default summary")
                return default_summary
            logger.info(f"Data collection result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in data collection: {e}")
            default_summary = (
                f"In {location} for a {plan_duration}-month period, the loamy soil with a pH of 6.5, high nitrogen, medium phosphorus, "
                f"and low potassium supports crops like wheat, rice, chickpeas, and mustard. Adequate moisture levels suit these crops, "
                f"but potassium supplementation is advised to optimize yields. Crop rotation with legumes like chickpeas enhances soil nitrogen, "
                f"while mustard improves soil structure. For a {plan_duration}-month plan, prioritize crops like wheat and chickpeas for rabi season "
                f"or rice for kharif season, depending on the timeframe, with fertilizers to address potassium deficiency."
            )
            return default_summary

    def data_collection_node(state: CropPlanState) -> CropPlanState:
        soil_data = get_soil_data(state['location'])
        data_summary = collect_data(
            state['location'],
            state['plan_duration'],
            json.dumps(soil_data)
        )
        return {
            "soil_data": soil_data,
            "history": state['history'] + [f"Data Summary: {data_summary}"],
            "iteration": state['iteration']
        }

    def grok_planning_node(state: CropPlanState) -> CropPlanState:
        try:
            data_summary = state['history'][-1] if state['history'] else "No data available"
            history = "\n".join(state['history'])
            prompt = planning_prompt.format(location=state['location'], plan_duration=state['plan_duration'], data_summary=data_summary, history=history)
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
            prompt = planning_prompt.format(location=state['location'], plan_duration=state['plan_duration'], data_summary=data_summary, history=history)
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
            prompt = planning_prompt.format(location=state['location'], plan_duration=state['plan_duration'], data_summary=data_summary, history=history)
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
                soil_data=json.dumps(state['soil_data']),
                location=state['location'],
                plan_duration=state['plan_duration']
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
plan_duration = int(input("Enter the plan duration in months (e.g., 6 or 12): "))
try:
    workflow = create_crop_planning_workflow(location=location, plan_duration=plan_duration, max_iterations=3)
    app = workflow.compile()

    try:
        png_bytes = app.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_bytes)
        print("Graph saved as graph.png")
    except Exception as e:
        logger.error(f"Error generating graph diagram: {e}")
        print(f"Failed to generate graph diagram: {e}")

    initial_state = {
        "location": location,
        "plan_duration": plan_duration,
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
            "plan_duration": plan_duration,
            "max_iterations": 3,
            "models": {
                "data_collection": "gemini-2.5-flash",
                "grok_planning": "deepseek-r1-distill-llama-70b",
                "gemini_planning": "gemini-2.5-flash",
                "gpt_planning": "openai/gpt-oss-120b",
                "summarizer": "cohere"
            },
            "tools": {
                "search": "GoogleSerperResults"
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