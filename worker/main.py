import asyncio
import json
import os
import logging
import subprocess

from fastapi import FastAPI
from pydantic import BaseModel
import pika
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv

from contextlib import asynccontextmanager

from prometheus_client import start_http_server, Summary

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

mp = {0: "INSULT", 1: "OBSCENITY", 2: "THREAT"}

class RequestData(BaseModel):
    chat_id: str
    reply_id: str
    username: str
    context: list[str]
    tags: list[str]

class ResponseData(BaseModel):
    chat_id: str
    reply_id: str
    username: str
    text: str

def download_lora_adapter():
    lora_url = os.getenv("LORA_ADAPTER_URL", "")
    if not lora_url:
        logger.info("No LORA_ADAPTER_URL set, skipping LoRA download.")
        return

    logger.info(f"Downloading LoRA adapter from {lora_url}...")
    local_zip_path = "/app/lora_adapter.zip"

    subprocess.run(["wget", "-q", "-O", local_zip_path, lora_url], check=True)

    lora_dir = "/app/lora_downloaded"
    subprocess.run(["rm", "-rf", lora_dir], check=True)
    subprocess.run(["mkdir", "-p", lora_dir], check=True)
    subprocess.run(["unzip", "-o", local_zip_path, "-d", lora_dir], check=True)

    os.environ["LORA_WEIGHTS_PATH"] = lora_dir
    logger.info(f"LoRA adapter downloaded and extracted to {lora_dir}.")

download_lora_adapter()

logger.info("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    device_map="cpu",
)

logger.info("Loading LoRA adapter...")
base_model = PeftModel.from_pretrained(base_model, os.getenv("LORA_WEIGHTS_PATH"))

logger.info("Compiling model with torch.compile (PyTorch 2.0+)...")
model = torch.compile(base_model)
torch.set_num_threads(4)
logger.info("Model loaded")

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", 5672))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")
REQUEST_QUEUE = os.getenv("REQUEST_QUEUE", "incoming_tasks")
RESPONSE_QUEUE = os.getenv("RESPONSE_QUEUE", "ready_tasks")

credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
connection_params = pika.ConnectionParameters(
    host=RABBITMQ_HOST,
    port=RABBITMQ_PORT,
    credentials=credentials
)

connection = pika.BlockingConnection(connection_params)
channel = connection.channel()

channel.queue_declare(queue=REQUEST_QUEUE)
channel.queue_declare(queue=RESPONSE_QUEUE)

INFERENCE_TIME = Summary('inference_time_seconds', 'Time spent in inference')

def preprocess(context, tags):
    prompt = f"Tags: {', '.join([mp[t] for t in tags])}\nContext: {context[:300]}\n###\nText:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    return inputs

def trim_text_after_marker(text, marker="Text:"):
    marker_index = text.find(marker)
    if marker_index != -1:
        return text[marker_index + len(marker):].strip()
    return text

@INFERENCE_TIME.time()
def perform_inference(context, tags):
    inputs = preprocess(context, tags)
    logger.info("Inference start")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.5,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result_text = trim_text_after_marker(decoded)
    logger.info("Inference end")
    return result_text

def on_request(ch, method, properties, body):
    request = json.loads(body)
    chat_id = request["chat_id"]
    reply_id = request["reply_id"]
    username = request["username"]
    context = request["context"]
    tags = request["tags"]

    text = perform_inference(context, tags)

    response = {
        "chat_id": chat_id,
        "reply_id": reply_id,
        "username": username,
        "text": text
    }

    channel.basic_publish(
        exchange="",
        routing_key=RESPONSE_QUEUE,
        body=json.dumps(response)
    )

    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_consume(queue=REQUEST_QUEUE, on_message_callback=on_request)

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RabbitMQ consumer...")
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, channel.start_consuming)
    logger.info("RabbitMQ consumer started.")

    from prometheus_client import start_http_server
    start_http_server(8001)

    yield

    logger.info("Stopping RabbitMQ consumer...")
    channel.stop_consuming()
    connection.close()
    logger.info("RabbitMQ consumer stopped.")

app.router.lifespan_context = lifespan

@app.post("/send_request")
async def send_request(request: RequestData):
    channel.basic_publish(
        exchange="",
        routing_key=REQUEST_QUEUE,
        body=request.json()
    )
    return {"status": "Request sent to RabbitMQ"}

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}
