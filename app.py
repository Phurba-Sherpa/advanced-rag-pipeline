import gradio as gr

from chat import answer_question


def format_context(context):
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        result += (
            f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n"
        )
        result += doc.page_content + "\n\n"

    return result


def extract_text(msg_content):
    if isinstance(msg_content, str):
        return msg_content
    elif isinstance(msg_content, list):
        # Gradio 6 can format as [{"text": "...", "type": "text"}] or similar
        return " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in msg_content)
    elif isinstance(msg_content, tuple):
        return str(msg_content[0])
    return str(msg_content)

def chat(history):
    print(f"DEBUG app.py chat: history={history!r}")
    last_message = extract_text(history[-1]["content"])
    prior = history[:-1]
    
    # We should also normalize prior messages for combined_question
    for m in prior:
        m["content"] = extract_text(m["content"])
        
    answer, context = answer_question(last_message, prior)
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)


def main():
    def put_message_in_chatbot(msg, history):
        return "", history + [{"role": "user", "content": msg}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])
    with gr.Blocks(title="Insurellm Expert Assistant") as ui:
        gr.Markdown("# 🏢 Insurellm Expert Assistant\nAsk me anything about Insurellm!")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=600,
                )
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about Insurellm...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        message.submit(
            put_message_in_chatbot,
            inputs=[message, chatbot],
            outputs=[message, chatbot],
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    ui.launch(inbrowser=True, theme=theme)


if __name__ == "__main__":
    main()
