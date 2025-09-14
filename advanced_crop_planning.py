from typing import Dict, TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
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
serper_key = os.getenv('serper_api_key')
if not all([groq_key, gemini_key, serper_key]):
    raise ValueError("Missing one or more API keys in environment variables")

llm_deepseek = ChatGroq(
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

def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = text.replace('\u2011', '-')  
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)  
    return text.strip()

def is_valid_paragraph(text: str) -> bool:
    if not text or len(text) < 50:
        return False
    if re.match(r'^[()\-,;\s]+$', text):
        return False
    return True

def create_crop_planning_workflow(location: str, plan_duration: int, language: str, max_iterations: int = 3):
    class CropPlanState(TypedDict):
        location: str
        plan_duration: int
        language: str
        soil_data: Optional[Dict]
        climate_data: Optional[Dict]
        crop_plan: Optional[str]
        history: List[str]
        iteration: int
        final_plan: Optional[str]

    workflow = StateGraph(CropPlanState)

    data_collection_prompt = (
        "You are an agricultural expert tasked with summarizing crop suitability for {location} over a {plan_duration}-month period. "
        "Use the provided search results: {search_results}, soil data: {soil_data}, and climate data: {climate_data}. "
        "If search results are unavailable or irrelevant, use your knowledge to summarize crop suitability based on the provided soil and climate data. "
        "Provide a concise summary (100-150 words) of suitable crops and considerations for the specified duration, "
        "considering both soil conditions and climate patterns."
    )

    planning_prompt = (
        "You are an agricultural planner for {location} over a {plan_duration}-month period. "
        "Based on the data summary: {data_summary}, soil data: {soil_data}, climate data: {climate_data}, and conversation history: {history}, "
        "propose a crop plan (one paragraph) including specific crops, planting schedules, and considerations, "
        "ensuring it aligns with the soil conditions, climate patterns, and plan duration, and counters or builds on previous plans."
    )

    summarizer_prompt = (
        "You are a summarizer agent. Given the conversation history: {history}, "
        "create a detailed crop plan for a {plan_duration}-month period in {location}. "
        "Extract a diverse set of crop varieties (e.g., HD3086 wheat, IR64-SRI rice, SBCC212 barley, HM-5 maize, LC1136 cotton, Jalgaon85 moong) from DeepSeek, Gemini, and GPT plans in the history, ensuring at least one crop from each if possible. "
        "Avoid repeating the same crops unnecessarily and include specific planting months (e.g., October-November), management strategies (e.g., SRI, IPM, crop rotation), and soil/climate considerations. "
        "Consider soil data: {soil_data} and climate data: {climate_data}. "
        "Write the paragraph in {language}, ensuring clarity, coherence, and no garbled text, symbols, or formatting errors. "
        "Return only the paragraph text, no JSON, Markdown, or extra characters."
    )

    @lru_cache(maxsize=100)
    def collect_data(location: str, plan_duration: int, soil_data: str) -> str:
        try:
            query = f"climate and weather forecast for {location} next {plan_duration} months"
            search_results = serper.invoke({"query": query})
            
            climate_query = f"agricultural climate suitability {location} next {plan_duration} months"
            climate_search = serper.invoke({"query": climate_query})
            
            prompt = data_collection_prompt.format(
                location=location,
                plan_duration=plan_duration,
                search_results=json.dumps(search_results),
                soil_data=soil_data,
                climate_data=json.dumps(climate_search)
            )
            result = llm_gemini.invoke(prompt).content
            result = sanitize_text(result)
            if not result or result.strip() == "" or "error" in result.lower():
                default_summary = (
                    f"In {location} for a {plan_duration}-month period, the loamy soil with a pH of 6.5, high nitrogen, medium phosphorus, "
                    f"and low potassium supports crops like wheat, rice, chickpeas, and mustard. The expected climate includes moderate temperatures "
                    f"around 25C with seasonal variations, adequate rainfall during monsoon periods, and cooler winters suitable for rabi crops. "
                    f"Adequate moisture levels suit these crops, but potassium supplementation is advised to optimize yields. "
                    f"Crop rotation with legumes like chickpeas enhances soil nitrogen, while mustard improves soil structure. "
                    f"For a {plan_duration}-month plan, prioritize crops like wheat and chickpeas for rabi season or rice for kharif season, "
                    f"depending on the timeframe, with fertilizers to address potassium deficiency and irrigation planning based on seasonal rainfall."
                )
                logger.warning("No valid search results; using default summary")
                return default_summary
            logger.info(f"Data collection result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in data collection: {e}")
            default_summary = (
                f"In {location} for a {plan_duration}-month period, the loamy soil with a pH of 6.5, high nitrogen, medium phosphorus, "
                f"and low potassium supports crops like wheat, rice, chickpeas, and mustard. The expected climate includes moderate temperatures "
                f"around 25C with seasonal variations, adequate rainfall during monsoon periods, and cooler winters suitable for rabi crops. "
                f"Adequate moisture levels suit these crops, but potassium supplementation is advised to optimize yields. "
                f"Crop rotation with legumes like chickpeas enhances soil nitrogen, while mustard improves soil structure. "
                f"For a {plan_duration}-month plan, prioritize crops like wheat and chickpeas for rabi season or rice for kharif season, "
                f"depending on the timeframe, with fertilizers to address potassium deficiency and irrigation planning based on seasonal rainfall."
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
            "climate_data": {"forecast": f"Climate data for {state['location']} next {state['plan_duration']} months"},
            "history": state['history'] + [f"Data Summary: {data_summary}"],
            "iteration": state['iteration'],
            "language": state['language']
        }

    def deepseek_planning_node(state: CropPlanState) -> CropPlanState:
        try:
            data_summary = state['history'][-1] if state['history'] else "No data available"
            history = "\n".join(state['history'])
            prompt = planning_prompt.format(
                location=state['location'], 
                plan_duration=state['plan_duration'], 
                data_summary=data_summary, 
                history=history,
                soil_data=json.dumps(state['soil_data']),
                climate_data=json.dumps(state['climate_data'])
            )
            plan = f"DeepSeek Plan: {sanitize_text(llm_deepseek.invoke(prompt).content)}"
            logger.info(f"DeepSeek plan: {plan}")
            return {
                "crop_plan": plan,
                "history": state['history'] + [plan],
                "iteration": state['iteration'] + 1
            }
        except Exception as e:
            logger.error(f"Error in DeepSeek planning: {e}")
            return {
                "crop_plan": "Error generating DeepSeek plan",
                "history": state['history'] + ["DeepSeek Plan: Error"],
                "iteration": state['iteration'] + 1
            }

    def gemini_planning_node(state: CropPlanState) -> CropPlanState:
        try:
            data_summary = state['history'][-1] if state['history'] else "No data available"
            history = "\n".join(state['history'])
            prompt = planning_prompt.format(
                location=state['location'], 
                plan_duration=state['plan_duration'], 
                data_summary=data_summary, 
                history=history,
                soil_data=json.dumps(state['soil_data']),
                climate_data=json.dumps(state['climate_data'])
            )
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
            prompt = planning_prompt.format(
                location=state['location'], 
                plan_duration=state['plan_duration'], 
                data_summary=data_summary, 
                history=history,
                soil_data=json.dumps(state['soil_data']),
                climate_data=json.dumps(state['climate_data'])
            )
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
            if not state['history'] or not state['soil_data'] or not state['climate_data']:
                logger.warning("Invalid input data for summarizer: missing history, soil_data, or climate_data")
                raise ValueError("Invalid input data")
            
            history = "\n".join(state['history'])
            prompt = summarizer_prompt.format(
                history=history,
                soil_data=json.dumps(state['soil_data']),
                climate_data=json.dumps(state['climate_data']),
                location=state['location'],
                plan_duration=state['plan_duration'],
                language=state['language']
            )
            final_plan = sanitize_text(llm_gemini.invoke(prompt).content)
            if not is_valid_paragraph(final_plan):
                logger.warning(f"Invalid summarizer output: {final_plan}")
                raise ValueError("Invalid or garbled summarizer output")
            logger.info(f"Final plan: {final_plan}")
            return {"final_plan": final_plan}
        except Exception as e:
            logger.error(f"Error in summarizer: {e} - History: {state['history']}")
            default_plan = (
                f"{state['location']} में {state['plan_duration']} महीने की फसल योजना के लिए, मई-जून में गर्मी-सहिष्णु ज्वार (जोवर) और मूंग (जलगांव85) को मई के पहले पखवाड़े में लगाएं, जिसमें गहन सिंचाई और पोटैशियम उर्वरक (30 किग्रा/हेक्टेयर) की आवश्यकता होगी। जुलाई-अगस्त में मानसून के साथ, जून के अंत या जुलाई की शुरुआत में खरीफ फसलें जैसे चावल (IR64-SRI, उत्तर-पश्चिम में जल प्रबंधन के साथ), मक्का (HM-5), कपास (LC1136), और उड़द लगाएं, जिसमें पोटैशियम की दो बार खुराक (20 किग्रा/हेक्टेयर) और मल्चिंग शामिल हो। सितंबर-अक्टूबर में खरीफ फसलों की कटाई करें और रबी मौसम के लिए मिट्टी तैयार करें, जिसमें कार्बनिक खाद (10 टन/हेक्टेयर), गहरी जुताई, और पोटैशियम-युक्त कम्पोस्ट (15 किग्रा/हेक्टेयर) शामिल हो। पूरे अवधि में, मिट्टी की नमी की निगरानी, ड्रिप सिंचाई, और एकीकृत कीट प्रबंधन (IPM) से अधिकतम उपज सुनिश्चित होगी।" if state['language'] == "Hindi" else
                f"{state['location']} ನಲ್ಲಿ {state['plan_duration']} ತಿಂಗಳ ಫಸಲು ಯೋಜನೆಗಾಗಿ, ಮೇ-ಜೂನ್‌ನಲ್ಲಿ ಶಾಖ-ಸಹಿಷ್ಣು ಜೋವರ್ (ಜೋವರ್) ಮತ್ತು ಮೂಂಗ್ (ಜಲಗಾಂವ್85) ಅನ್ನು ಮೇನ ಮೊದಲ ಪಕ್ಷದಲ್ಲಿ ನಾಟಿ ಮಾಡಿ, ಇದಕ್ಕೆ ತೀವ್ರವಾದ ನೀರಾವರಿ ಮತ್ತು ಪೊಟ್ಯಾಸಿಯಂ ರಸಗೊಬ್ಬರ (30 ಕೆಜಿ/ಹೆಕ್ಟೇರ್) ಅಗತ್ಯವಿರುತ್ತದೆ। ಜುಲೈ-ಆಗಸ್ಟ್‌ನಲ್ಲಿ ಮಾನ್ಸೂನ್ ಜೊತೆಗೆ, ಜೂನ್‌ನ ಕೊನೆಯಲ್ಲಿ ಅಥವಾ ಜುಲೈ ಆರಂಭದಲ್ಲಿ ಖರೀಫ್ ಬೆಳೆಗಳಾದ ಭತ್ತ (IR64-SRI, ಉತ್ತರ-ಪಶ್ಚಿಮದಲ್ಲಿ ನೀರಿನ ನಿರ್ವಹಣೆಯೊಂದಿಗೆ), ಮೆಕ್ಕೆಜೋಳ (HM-5), ಹತ್ತಿ (LC1136), ಮತ್ತು ಒಡಲಾಳನ್ನು ನಾಟಿ ಮಾಡಿ, ಇದರಲ್ಲಿ ಪೊಟ್ಯಾಸಿಯಂನ ಎರಡು ಡೋಸ್‌ಗಳು (20 ಕೆಜಿ/ಹೆಕ್ಟೇರ್) ಮತ್ತು ಮಲ್ಚಿಂಗ್ ಸೇರಿರುತ್ತದೆ। ಸೆಪ್ಟೆಂಬರ್-ಅಕ್ಟೋಬರ್‌ನಲ್ಲಿ ಖರೀಫ್ ಬೆಳೆಗಳ ಕೊಯ್ಲು ಮಾಡಿ ಮತ್ತು ರಬಿ ಋತುವಿಗೆ ಮಣ್ಣನ್ನು ಸಿದ್ಧಪಡಿಸಿ, ಇದರಲ್ಲಿ ಸಾವಯವ ಗೊಬ್ಬರ (10 ಟನ್/ಹೆಕ್ಟೇರ್), ಆಳವಾದ ಉಳುಮೆ, ಮತ್ತು ಪೊಟ್ಯಾಸಿಯಂ-ಒಳಗೊಂಡ ಕಾಂಪೋಸ್ಟ್ (15 ಕೆಜಿ/ಹೆಕ್ಟೇರ್) ಸೇರಿರುತ್ತದೆ। ಇಡೀ ಅವಧಿಯಲ್ಲಿ, ಮಣ್ಣಿನ ತೇವಾಂಶದ ಮೇಲ್ವಿಚಾರಣೆ, ಡ್ರಿಪ್ ನೀರಾವರಿ, ಮತ್ತು ಸಂಯೋಜಿತ ಕೀಟ ನಿರ್ವಹಣೆ (IPM) ಮೂಲಕ ಗರಿಷ್ಠ ಇಳುವರಿಯನ್ನು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ।" if state['language'] == "Kannada" else
                f"{state['location']} ൽ {state['plan_duration']} മാസത്തെ വിള യോജനയ്ക്കായി, മേയ്-ജൂൺ മാസങ്ങളിൽ ചൂട്-സഹിക്കുന്ന ജോവർ (ജോവർ), മൂങ് (ജലഗാവ്85) എന്നിവ മേയ് ഒന്നാം പക്ഷത്തിൽ നടുക, ഇതിന് തീവ്രമായ ജലസേചനവും പൊട്ടാസ്യം വളം (30 കി.ഗ്രാം/ഹെക്ടർ) ആവശ്യമാണ്। ജൂലൈ-ഓഗസ്റ്റിൽ മൺസൂൺ വരുന്നതോടെ, ജൂൺ അവസാനം അല്ലെങ്കിൽ ജൂലൈ ആദ്യം ഖരീഫ് വിളകളായ അരി (IR64-SRI, വടക്ക്-പടിഞ്ഞാറ് ജലനിയന്ത്രണത്തോടെ), ചോളം (HM-5), കോട്ടൺ (LC1136), ഉഴുന്ന് എന്നിവ നടുക, ഇതിൽ പൊട്ടാസ്യത്തിന്റെ രണ്ട് ഡോസുകൾ (20 കി.ഗ്രാം/ഹെക്ടർ) ഉം മൾച്ചിംഗും ഉൾപ്പെടുന്നു। സെപ്റ്റംബർ-ഒക്ടോബറിൽ ഖരീഫ് വിളകൾ വിളവെടുക്കുകയും റബി സീസണിനായി മണ്ണ് തയ്യാറാക്കുകയും ചെയ്യുക, ഇതിൽ ജൈവ വളം (10 ടൺ/ഹെക്ടർ), ആഴത്തിലുള്ള ഉഴവ്, പൊട്ടാസ്യം അടങ്ങിയ കമ്പോസ്റ്റ് (15 കി.ഗ്രാം/ഹെക്ടർ) എന്നിവ ഉൾപ്പെടുന്നു। മുഴുവൻ കാലയളവിലും, മണ്ണിന്റെ ഈർപ്പം നിരീക്ഷണം, ഡ്രിപ്പ് ജലസേചനം, സംയോജിത കീടനാശിനി നിയന്ത്രണം (IPM) എന്നിവ ഉപയോഗിച്ച് പരമാവധി വിളവ് ഉറപ്പാക്കുക." if state['language'] == "Malayalam" else
                f"{state['location']} में {state['plan_duration']} महीने की फसल योजना के लिए, मई-जून में गर्मी-सहिष्णु ज्वार (जोवर) और मूंग (जलगांव85) को मई के पहले पखवाड़े में लगाएं, जिसमें गहन सिंचाई और पोटैशियम उर्वरक (30 किग्रा/हेक्टेयर) की आवश्यकता होगी। जुलाई-अगस्त में मानसून के साथ, जून के अंत या जुलाई की शुरुआत में खरीफ फसलें जैसे चावल (IR64-SRI, उत्तर-पश्चिम में जल प्रबंधन के साथ), मक्का (HM-5), कपास (LC1136), और उड़द लगाएं, जिसमें पोटैशियम की दो बार खुराक (20 किग्रा/हेक्टेयर) और मल्चिंग शामिल हो। सितंबर-अक्टूबर में खरीफ फसलों की कटाई करें और रबी मौसम के लिए मिट्टी तैयार करें, जिसमें कार्बनिक खाद (10 टन/हेक्टेयर), गहरी जुताई, और पोटैशियम-युक्त कम्पोस्ट (15 किग्रा/हेक्टेयर) शामिल हो। पूरे अवधि में, मिट्टी की नमी की निगरानी, ड्रिप सिंचाई, और एकीकृत कीट प्रबंधन (IPM) से अधिकतम उपज सुनिश्चित होगी।"
            )
            logger.info("Using default plan due to summarizer error")
            return {"final_plan": default_plan}

    workflow.add_node("data_collection", data_collection_node)
    workflow.add_node("deepseek_planning", deepseek_planning_node)
    workflow.add_node("gemini_planning", gemini_planning_node)
    workflow.add_node("gpt_planning", gpt_planning_node)
    workflow.add_node("summarizer", summarizer_node)

    workflow.set_entry_point("data_collection")
    workflow.add_edge("data_collection", "deepseek_planning")
    workflow.add_edge("deepseek_planning", "gemini_planning")
    workflow.add_edge("gemini_planning", "gpt_planning")

    def check_iterations(state: CropPlanState) -> str:
        logger.debug(f"Iteration count: {state['iteration']}/{max_iterations}")
        return "summarizer" if state['iteration'] >= max_iterations else "deepseek_planning"

    workflow.add_conditional_edges(
        "gpt_planning",
        check_iterations,
        {"summarizer": "summarizer", "deepseek_planning": "deepseek_planning"}
    )
    workflow.add_edge("summarizer", END)

    return workflow

location = input("Enter the location (e.g., Punjab, India): ")
plan_duration = int(input("Enter the plan duration in months (e.g., 6 or 12): "))
language = input("Enter the language for the plan (e.g., English, Hindi, Malayalam): ")
try:
    workflow = create_crop_planning_workflow(location=location, plan_duration=plan_duration, language=language, max_iterations=3)
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
        "language": language,
        "soil_data": None,
        "climate_data": None,
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
            "language": language,
            "max_iterations": 3,
            "models": {
                "data_collection": "gemini-2.5-flash",
                "deepseek_planning": "deepseek-r1-distill-llama-70b",
                "gemini_planning": "gemini-2.5-flash",
                "gpt_planning": "openai/gpt-oss-120b",
                "summarizer": "gemini-2.5-flash"
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