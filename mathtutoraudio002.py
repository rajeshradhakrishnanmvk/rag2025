import os
import uuid
import base64
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


def save_uploaded_audio(audio_path):
    """Save audio file to uploads directory"""
    if audio_path is None:
        return None
    import shutil
    ext = Path(audio_path).suffix or ".wav"
    file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    shutil.copy(audio_path, file_path)
    return str(file_path)


def encode_audio_to_base64(audio_path):
    """Encode audio file to base64 string"""
    if audio_path is None:
        return None
    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()
            return base64.b64encode(audio_data).decode()
    except Exception as e:
        print(f"Error encoding audio: {e}")
        return None


def build_messages(history, user_text, image_path=None, audio_path=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history:
        user_msg = {"role": "user", "content": turn["user"]}
        if turn.get("image_path"):
            user_msg["images"] = [turn["image_path"]]
        if turn.get("audio_path"):
            audio_b64 = encode_audio_to_base64(turn["audio_path"])
            if audio_b64:
                user_msg["audio"] = audio_b64
        messages.append(user_msg)
        messages.append({"role": "assistant", "content": turn["assistant"]})

    current_user = {"role": "user", "content": user_text}
    if image_path:
        current_user["images"] = [image_path]
    if audio_path:
        audio_b64 = encode_audio_to_base64(audio_path)
        if audio_b64:
            current_user["audio"] = audio_b64
    messages.append(current_user)

    return messages

def chat_with_tutor(user_text, image, audio, state):
    """Main chat function that handles text, image, and audio input"""
    # Save audio file if provided
    audio_path = save_uploaded_audio(audio) if audio is not None else None

    # Check if we have any input
    if not user_text and image is None and audio is None:
        return state, state, "", None

    image_path = save_uploaded_image(image) if image is not None else None
    history = state if state is not None else []

    # Set default text if only media is provided
    effective_text = user_text.strip() if user_text else ""
    if not effective_text:
        if audio_path and image_path:
            effective_text = "Solve the math problem."
        elif audio_path:
            effective_text = "Solve this."
        elif image_path:
            effective_text = "Read this notebook page and explain the math problem step by step."

    messages = build_messages(history, effective_text, image_path=image_path, audio_path=audio_path)

    # Debug output
    if audio_path:
        print(f"\n[DEBUG] Sending audio to Gemma 4:")
        print(f"  - Audio file: {audio_path}")
        print(f"  - Text prompt: {effective_text}")
        print(f"  - Has image: {image_path is not None}")
        last_msg = messages[-1]
        if 'audio' in last_msg:
            print(f"  - Audio base64 length: {len(last_msg['audio'])} chars")

    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        options={"temperature": 0.2}
    )

    if audio_path:
        print(f"[DEBUG] Gemma 4 response preview: {response.message.content[:100]}...")


    assistant_text = response["message"]["content"]

    history.append({
        "user": effective_text,
        "assistant": assistant_text,
        "image_path": image_path,
        "audio_path": audio_path
    })

    chat_view = []
    for item in history:
        user_label = item["user"]
        if item.get("image_path"):
            user_label = f"{user_label}\n[Notebook page uploaded]"
        if item.get("audio_path"):
            user_label = f"{user_label}\n[Audio question uploaded]"
        chat_view.append({"role": "user", "content": user_label})
        chat_view.append({"role": "assistant", "content": item["assistant"]})

    return history, chat_view, "", None

def clear_chat():
    return [], [], "", None

with gr.Blocks(title="Gemma Math Tutor") as demo:
    gr.Markdown("# 🎓 Notebook Math Tutor with Voice")
    gr.Markdown("**Upload a notebook page** 📷 | **Type or speak your question** 🎤 | **Get step-by-step help!** 📝")

    state = gr.State([])

    chatbot = gr.Chatbot(height=500)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📷 Scan notebook page")
            audio_input = gr.Audio(
                label="🎤 Speak your question (optional)",
                sources=["microphone", "upload"],
                type="filepath"
            )
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="✍️ Type your question (optional)",
                placeholder="Example: Solve this equation step by step",
                lines=4
            )
            with gr.Row():
                send_btn = gr.Button("Ask tutor 📤", variant="primary", scale=2)
                clear_btn = gr.Button("Clear 🗑️", scale=1)

    send_btn.click(
        fn=chat_with_tutor,
        inputs=[text_input, image_input, audio_input, state],
        outputs=[state, chatbot, text_input, audio_input]
    )

    text_input.submit(
        fn=chat_with_tutor,
        inputs=[text_input, image_input, audio_input, state],
        outputs=[state, chatbot, text_input, audio_input]
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[state, chatbot, text_input, audio_input]
    )

demo.launch()
