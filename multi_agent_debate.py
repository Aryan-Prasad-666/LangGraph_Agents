from typing import Dict, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
import os
import random
import time

load_dotenv()

groq_key = os.getenv('groq2_api_key')
gemini_key = os.getenv('gemini_api_key')
cohere_key = os.getenv('cohere_api_key')

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    api_key= groq_key,
    temperature= 0.6
)

llm2 = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    api_key = gemini_key,
    temperature = 0.6
)

llm3 = ChatCohere(
    api_key = cohere_key,
    temperature=0.6
)

debate_topic = "Should AI be used in medical science?"
output_parser = CommaSeparatedListOutputParser()
output = llm.invoke("I wish to have a debate on {}. What would be the fighting sides called? Output just the names and nothing else as comma separated list".format(debate_topic)).content
classes = output_parser.parse(output)
if len(classes) != 2:
    raise ValueError("Expected exactly two debate sides, got: {}".format(classes))

class GraphState(TypedDict):
    classification: Optional[str]
    history: str
    current_response: Optional[str]
    count: int
    results: Optional[str]
    greeting: Optional[str]

workflow = StateGraph(GraphState)

prefix_start = 'You are in support of {}. You are in a debate with {} over the topic: {}. This is the conversation so far \n{}\n. Provide a unique, concise argument (one sentence) to support {}, countering {} with a specific example or reasoning, avoiding repetition.'

def classify(question):
    try:
        return llm.invoke("classify the sentiment of input as {} or {}. Output just the class. Input:{}".format('_'.join(classes[0].split(' ')), '_'.join(classes[1].split(' ')), question)).content.strip()
    except Exception as e:
        print(f"Error classifying input: {e}")
        return "_".join(classes[0].split(' '))  # Fallback

def classify_input_node(state):
    question = state.get('current_response', '')
    classification = classify(question)
    return {"classification": classification}

def handle_greeting_node(state):
    return {"greeting": "Hello! Today we will witness the fight between {} vs {}".format(classes[0], classes[1])}

def handle_pro(state):
    summary = state.get('history', '').strip()
    current_response = state.get('current_response', '').strip()
    prompt = prefix_start.format(classes[0], classes[1], debate_topic, summary, classes[0], current_response or "Nothing")
    argument = classes[0] + ": " + llm.invoke(prompt).content
    return {"history": summary + '\n' + argument, "current_response": argument, "count": state.get('count', 0) + 1}

def handle_opp(state):
    summary = state.get('history', '').strip()
    current_response = state.get('current_response', '').strip()
    prompt = prefix_start.format(classes[1], classes[0], debate_topic, summary, classes[1], current_response or "Nothing")
    argument = classes[1] + ": " + llm2.invoke(prompt).content
    return {"history": summary + '\n' + argument, "current_response": argument, "count": state.get('count', 0) + 1}

def result(state):
    summary = state.get('history', '').strip()
    prompt = "Summarize the conversation and judge who won the debate. No ties are allowed. Conversation: {}".format(summary)
    return {"results": llm3.invoke(prompt).content}

workflow.add_node("classify_input", classify_input_node)
workflow.add_node("handle_greeting", handle_greeting_node)
workflow.add_node("handle_pro", handle_pro)
workflow.add_node("handle_opp", handle_opp)
workflow.add_node("result", result)

def decide_next_node(state):
    return "handle_opp" if state.get('classification') == '_'.join(classes[0].split(' ')) else "handle_pro"

def check_conv_length(state):
    count = state.get("count", 0)
    return "result" if count >= 10 else "classify_input"

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

app = workflow.compile()
conversation = app.invoke({'count': 0, 'history': '', 'current_response': ''})
print(conversation.get('history', 'No conversation history generated'))

