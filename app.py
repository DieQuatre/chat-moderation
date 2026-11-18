"""Real-time chat moderation demo (Use Case 4).

A shared in-memory chat room: open the page in two browser windows (or share the link)
and chat as two players. Every message goes through the selected model BEFORE it is
shown; offensive messages are masked for everyone. The side panel logs each message's
model, label, confidence and response time.

Run:  python app.py            (local, http://127.0.0.1:7860)
      python app.py --share    (also creates a temporary public link)
"""
import html
import sys
import threading
import time

import gradio as gr

from src.models.download import ensure_models

ensure_models()

from src.models.bert import BertFP32, BertInt8  # noqa: E402
from src.models.lexicon_rule import LexiconRule  # noqa: E402
from src.models.tfidf_lr import TfidfLR  # noqa: E402
from src.models.w2v_ffn import W2VFFN  # noqa: E402

DEFAULT_MODEL = "BERT INT8"
print("Loading models ...")
MODELS = {m.name: m for m in [BertInt8(), BertFP32(device="cpu"), TfidfLR.load(),
                              W2VFFN.load(), LexiconRule.load()]}
for _model in MODELS.values():  # warm-up: a model's first calls are slow, so run them before any user message
    for _ in range(3):
        _model.predict_proba(["hello, nice to meet you"])
print("Models ready:", ", ".join(MODELS))

MESSAGES = []          # shared by every browser window
LOCK = threading.Lock()


def moderate(text, model_name):
    model = MODELS[model_name]
    start = time.perf_counter()
    prob = float(model.predict_proba([text])[0])
    ms = (time.perf_counter() - start) * 1000
    label = "offensive" if prob >= 0.5 else "clean"
    confidence = prob if label == "offensive" else 1 - prob
    return label, confidence, ms


def chat_html():
    with LOCK:
        rows = list(MESSAGES[-50:])
    if not rows:
        return "<div class='chat'><i>No messages yet. Say hi!</i></div>"
    parts = []
    for m in rows:
        body = ("<span class='hidden'>[hidden by moderator]</span>" if m["label"] == "offensive"
                else html.escape(m["text"]))
        icon = "⛔" if m["label"] == "offensive" else "✅"
        parts.append(f"<div class='msg'><b>{html.escape(m['user'])}</b>: {body}"
                     f"<span class='tag'>{icon} {m['label']} {m['confidence']:.0%} · "
                     f"{m['ms']:.1f} ms · {m['model']}</span></div>")
    return "<div class='chat'>" + "".join(parts) + "</div>"


def log_table():
    with LOCK:
        rows = list(MESSAGES[-20:])[::-1]
    head = "<tr><th>time</th><th>player</th><th>model</th><th>label</th><th>conf.</th><th>ms</th></tr>"
    body = "".join(f"<tr><td>{m['time']}</td><td>{html.escape(m['user'])}</td><td>{m['model']}</td>"
                   f"<td>{m['label']}</td><td>{m['confidence']:.0%}</td><td>{m['ms']:.1f}</td></tr>"
                   for m in rows)
    return f"<table class='log'>{head}{body}</table>"


def send(user, text, model_name):
    text, user = (text or "").strip(), (user or "").strip() or "Player"
    if text:
        label, confidence, ms = moderate(text, model_name)
        with LOCK:
            MESSAGES.append({"user": user, "text": text, "label": label, "confidence": confidence,
                             "ms": ms, "model": model_name, "time": time.strftime("%H:%M:%S")})
    return "", chat_html(), log_table()


def clear():
    with LOCK:
        MESSAGES.clear()
    return chat_html(), log_table()


CSS = """
.chat {height: 420px; overflow-y: auto; display: flex; flex-direction: column; gap: 6px;}
.msg {padding: 6px 10px; border-radius: 8px; background: rgba(127,127,127,0.12);}
.hidden {font-style: italic; opacity: 0.7;}
.tag {float: right; font-size: 0.8em; opacity: 0.75; margin-left: 12px;}
.log {width: 100%; font-size: 0.85em; border-collapse: collapse;}
.log th, .log td {text-align: left; padding: 4px 6px; border-bottom: 1px solid rgba(127,127,127,0.25);}
"""

with gr.Blocks(title="Chat Moderation") as demo:
    gr.Markdown("## Real-time chat moderation\nEvery message is checked by the selected model "
                "before it appears. Offensive messages are hidden for everyone.")
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.HTML(chat_html())
            with gr.Row():
                user = gr.Textbox(value="Player 1", label="Your name", scale=1)
                message = gr.Textbox(label="Message", placeholder="Type and press Enter", scale=4)
            with gr.Row():
                model = gr.Dropdown(list(MODELS), value=DEFAULT_MODEL, label="Moderation model")
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear chat")
        with gr.Column(scale=2):
            gr.Markdown("### Response time log")
            log = gr.HTML(log_table())

    message.submit(send, [user, message, model], [message, chat, log])
    send_btn.click(send, [user, message, model], [message, chat, log])
    clear_btn.click(clear, None, [chat, log])
    gr.Timer(1.0).tick(lambda: (chat_html(), log_table()), None, [chat, log], show_progress="hidden")

if __name__ == "__main__":
    demo.launch(share="--share" in sys.argv, css=CSS)
