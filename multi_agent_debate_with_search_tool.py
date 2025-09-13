from typing import Dict, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_cohere import ChatCohere
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from functools import lru_cache
import os
import logging
import json
from datetime import datetime
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Validate environment variables
groq_key = os.getenv('groq2_api_key')
gemini_key = os.getenv('gemini_api_key')
cohere_key = os.getenv('cohere_api_key')
tavily_key = os.getenv('tavily_api_key')
if not all([groq_key, gemini_key, cohere_key]):
    raise ValueError("Missing one or more API keys in environment variables")

# Initialize LLMs (unchanged as per your instruction)
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_key,
    temperature=0.6
)
llm2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=gemini_key,
    temperature=0.6
)
llm3 = ChatCohere(
    api_key=cohere_key,
    temperature=0.6
)

tavily = TavilySearch(max_results = 2, api_key = tavily_key)

def sanitize_text(text):
    """Replace or remove problematic Unicode characters."""
    if not isinstance(text, str):
        return text
    # Replace non-breaking hyphen (\u2011) with standard hyphen and remove other problematic characters
    text = text.replace('\u2011', '-')
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text

def create_debate_workflow(debate_topic, max_turns=10):
    output_parser = CommaSeparatedListOutputParser()
    try:
        output = sanitize_text(llm.invoke("I wish to have a debate on {}. What would be the fighting sides called? Output just the names and nothing else as comma separated list".format(debate_topic)).content)
        classes = output_parser.parse(output)
        if len(classes) != 2:
            raise ValueError(f"Expected exactly two debate sides, got: {classes}")
        logger.info(f"Debate sides: {classes}")
    except Exception as e:
        logger.error(f"Error determining debate sides: {e}")
        raise

    class GraphState(TypedDict):
        classification: Optional[str]
        history: str
        current_response: Optional[str]
        count: int
        results: Optional[str]
        greeting: Optional[str]

    workflow = StateGraph(GraphState)

    prefix_start = (
        'You are in support of {}. You are in a debate with {} over the topic: {}. '
        'This is the conversation so far \n{}\n. '
        'Provide a unique, concise argument (one sentence) to support {}, '
        'countering {} with a specific real-world example, statistic, or reasoning, '
        'and avoid repeating prior arguments.'
    )

    @lru_cache(maxsize=100)
    def classify(question, class0, class1):
        try:
            result = sanitize_text(llm2.invoke("classify the sentiment of input as {} or {}. Output just the class. Input:{}".format(class0, class1, question)).content.strip())
            logger.info(f"Classified input '{question[:50]}...' as: {result}")
            return result
        except Exception as e:
            logger.error(f"Error classifying input: {e}")
            return class0

    def classify_input_node(state):
        question = state.get('current_response', '')
        cache_info = classify.cache_info()
        logger.debug(f"Cache info before classify: hits={cache_info.hits}, misses={cache_info.misses}")
        classification = classify(question, '_'.join(classes[0].split(' ')), '_'.join(classes[1].split(' ')))
        logger.debug(f"Cache info after classify: hits={classify.cache_info().hits}, misses={classify.cache_info().misses}")
        return {"classification": classification}

    def handle_greeting_node(state):
        greeting = f"Hello! Today we will witness the fight between {classes[0]} vs {classes[1]}"
        logger.info(f"Generated greeting: {greeting}")
        return {"greeting": greeting}
    
    def search_tool_node(state):
        query = state.get("current_response", "").strip()
        if not query:
            query = state.get("history", "").strip().split("\n")[-1]  # fallback to last turn
        try:
            results = tavily.invoke({"query": query})
            logger.info(f"Tavily search results for '{query}': {results}")
            return {"search_results": results}
        except Exception as e:
            logger.error(f"Error during Tavily search: {e}")
            return {"search_results": "No results found"}


    def handle_pro(state):
        try:
            summary = state.get('history', '').strip()
            current_response = state.get('current_response', '').strip()
            prompt = prefix_start.format(classes[0], classes[1], debate_topic, summary, classes[0], current_response or "Nothing")
            argument = classes[0] + ": " + sanitize_text(llm.invoke(prompt).content)
            logger.info(f"Pro argument: {argument}")
            return {"history": summary + '\n' + argument, "current_response": argument, "count": state.get('count', 0) + 1}
        except Exception as e:
            logger.error(f"Error in handle_pro: {e}")
            return {"history": summary + '\n' + f"{classes[0]}: Error generating argument", "current_response": "Error", "count": state.get('count', 0) + 1}

    def handle_opp(state):
        try:
            summary = state.get('history', '').strip()
            current_response = state.get('current_response', '').strip()
            prompt = prefix_start.format(classes[1], classes[0], debate_topic, summary, classes[1], current_response or "Nothing")
            argument = classes[1] + ": " + sanitize_text(llm2.invoke(prompt).content)
            logger.info(f"Opp argument: {argument}")
            return {"history": summary + '\n' + argument, "current_response": argument, "count": state.get('count', 0) + 1}
        except Exception as e:
            logger.error(f"Error in handle_opp: {e}")
            return {"history": summary + '\n' + f"{classes[1]}: Error generating argument", "current_response": "Error", "count": state.get('count', 0) + 1}

    def result(state):
        try:
            summary = state.get('history', '').strip()
            prompt = "Summarize the conversation and judge who won the debate. No ties are allowed. Conversation: {}".format(summary)
            result = sanitize_text(llm3.invoke(prompt).content)
            logger.info(f"Debate result: {result}")
            return {"results": result}
        except Exception as e:
            logger.error(f"Error in result: {e}")
            return {"results": "Error summarizing debate"}

    workflow.add_node("classify_input", classify_input_node)
    workflow.add_node("handle_greeting", handle_greeting_node)
    workflow.add_node("handle_pro", handle_pro)
    workflow.add_node("handle_opp", handle_opp)
    workflow.add_node("result", result)

    def decide_next_node(state):
        classification = state.get('classification')
        logger.debug(f"Deciding next node based on classification: {classification}")
        return "handle_opp" if classification == '_'.join(classes[0].split(' ')) else "handle_pro"

    def check_conv_length(state):
        count = state.get("count", 0)
        logger.debug(f"Conversation turn count: {count}/{max_turns}")
        return "result" if count >= max_turns else "classify_input"

    workflow.add_conditional_edges(
        "classify_input",
        decide_next_node,
        {"handle_pro": "handle_pro", "handle_opp": "handle_opp"}
    )
    workflow.add_conditional_edges(
        "handle_pro",
        check_conv_length,
        {"result": "result", "classify_input": "classify_input"}
    )
    workflow.add_conditional_edges(
        "handle_opp",
        check_conv_length,
        {"result": "result", "classify_input": "classify_input"}
    )

    workflow.set_entry_point("handle_greeting")
    workflow.add_edge('handle_greeting', "handle_pro")
    workflow.add_edge('result', END)

    return workflow, classes

# Create and compile workflow
debate_topic = "Should AI be used in medical science?"
try:
    workflow, classes = create_debate_workflow(debate_topic=debate_topic, max_turns=10)
    app = workflow.compile()

    # Visualize the graph
    try:
        graph = app.get_graph()
        graph.draw_png("debate_workflow.png")
        logger.info("Graph diagram saved as 'debate_workflow.png'")
    except Exception as e:
        logger.error(f"Error generating graph diagram: {e}")
        print(f"Failed to generate graph diagram: {e}")

    # Run the conversation and save outputs
    conversation = app.invoke({'count': 0, 'history': '', 'current_response': ''})
    history = conversation.get('history', 'No conversation history generated')
    results = conversation.get('results', 'No result generated')
    greeting = conversation.get('greeting', 'No greeting generated')

    print("\nDebate History:\n", history)
    print("\nDebate Summary and Result:\n", results)

    # Parse history into turns
    history_turns = []
    current_turn = None
    for line in history.split("\n"):
        line = line.strip()
        if line.startswith(f"{classes[0]}:") or line.startswith(f"{classes[1]}:"):
            if current_turn:
                history_turns.append(current_turn)
            current_turn = line
        elif line and current_turn:
            current_turn += "\n" + line
    if current_turn:
        history_turns.append(current_turn)

    # Validate outputs
    if history == 'No conversation history generated' or results == 'No result generated':
        logger.warning("Incomplete debate data; files may be incomplete")

    # Save as TXT
    try:
        with open("debate_log.txt", "w", encoding="utf-8") as f:
            f.write("Debate Topic: {}\n\nHistory:\n{}\n\nResult:\n{}".format(debate_topic, history, results))
        logger.info("Conversation saved to 'debate_log.txt'")
    except Exception as e:
        logger.error(f"Error saving to text file: {e}")
        print(f"Failed to save to text file: {e}")

    # Save as JSON
    debate_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "debate_topic": debate_topic,
            "max_turns": 10,
            "debate_sides": classes,
            "models": {
                "pro_arguments": "openai/gpt-oss-120b",
                "opp_arguments": "gemini-2.5-flash",
                "classification": "gemini-2.5-flash",
                "result": "cohere"
            }
        },
        "conversation": {
            "greeting": greeting,
            "history": history_turns,
            "result": results
        }
    }

    try:
        json_file = "debate_log.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(debate_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Debate data saved to '{json_file}'")
    except Exception as e:
        logger.error(f"Error saving to JSON: {e}")
        print(f"Failed to save to JSON: {e}")

except Exception as e:
    logger.error(f"Error running debate: {e}")
    print("Failed to run debate:", str(e))