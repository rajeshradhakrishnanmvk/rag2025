import os
import uuid
from pathlib import Path
import gradio as gr
import ollama

MODEL_NAME = "gemma4:e4b"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """
You are a patient math tutor.

Rules:
- Teach math step by step in simple language.
- Do not skip important algebra or arithmetic steps.
- If the notebook image is unclear, say exactly what is unreadable.
- If a question refers to the uploaded page, use the image context first.
- Keep explanations concise but educational.
- End with one short follow-up practice question when appropriate.
- If the student asks a follow-up question, answer in context of the earlier page and chat.
"""

def save_uploaded_image(image):
    if image is None:
        return None
    ext = ".png"
    file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    image.save(file_path)
    return str(file_path)

def build_messages(history, user_text, image_path=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history:
        user_msg = {"role": "user", "content": turn["user"]}
        if turn.get("image_path"):
            user_msg["images"] = [turn["image_path"]]
        messages.append(user_msg)
        messages.append({"role": "assistant", "content": turn["assistant"]})

    current_user = {"role": "user", "content": user_text}
    if image_path:
        current_user["images"] = [image_path]
    messages.append(current_user)

    return messages

def chat_with_tutor(user_text, image, state):
    if not user_text and image is None:
        return state, state, ""

    image_path = save_uploaded_image(image) if image is not None else None
    history = state if state is not None else []

    effective_text = user_text.strip() if user_text else "Read this notebook page and explain the math problem step by step."

    messages = build_messages(history, effective_text, image_path=image_path)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        options={"temperature": 0.2}
    )

    assistant_text = response["message"]["content"]

    history.append({
        "user": effective_text,
        "assistant": assistant_text,
        "image_path": image_path
    })

    chat_view = []
    for item in history:
        user_label = item["user"]
        if item.get("image_path"):
            user_label = f"{user_label}\n[Notebook page uploaded]"
        chat_view.append({"role": "user", "content": user_label})
        chat_view.append({"role": "assistant", "content": item["assistant"]})

    return history, chat_view, ""

def clear_chat():
    return [], [], ""

with gr.Blocks(title="Gemma Math Tutor") as demo:
    gr.Markdown("# Notebook Math Tutor")
    gr.Markdown("Upload a notebook page, then ask questions about the math problem.")

    state = gr.State([])

    chatbot = gr.Chatbot(height=500)

    with gr.Row():
        image_input = gr.Image(type="pil", label="Scan notebook page")
        with gr.Column():
            text_input = gr.Textbox(
                label="Ask a question",
                placeholder="Example: Solve this equation step by step",
                lines=4
            )
            send_btn = gr.Button("Ask tutor", variant="primary")
            clear_btn = gr.Button("Clear")

    send_btn.click(
        fn=chat_with_tutor,
        inputs=[text_input, image_input, state],
        outputs=[state, chatbot, text_input]
    )

    text_input.submit(
        fn=chat_with_tutor,
        inputs=[text_input, image_input, state],
        outputs=[state, chatbot, text_input]
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[state, chatbot, text_input]
    )

demo.launch()
