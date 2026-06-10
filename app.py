"""
app.py — Gradio web interface for the Westbrook Unofficial Housing Guide.

Run with:  python app.py
Then open: http://localhost:7860

The interface has:
  - A question text box (also submits on Enter)
  - An "Ask" button
  - An answer output box
  - A sources output box showing which documents were cited
  - Example questions pre-loaded for easy demo
"""

import gradio as gr
from query import ask


def handle_query(question: str):
    """
    Called by Gradio when the user submits a question.
    Returns (answer_text, sources_text) for the two output boxes.
    """
    question = question.strip()
    if not question:
        return "Please enter a question.", ""

    result = ask(question)

    answer = result["answer"]

    if result["sources"]:
        source_lines = []
        for src in result["sources"]:
            # Make filenames human-readable
            readable = src.replace("_", " ").replace(".txt", "").title()
            source_lines.append(f"• {readable}  ({src})")
        sources_text = "\n".join(source_lines)
    else:
        sources_text = "No sources found."

    return answer, sources_text


# ── Gradio UI layout ──────────────────────────────────────────────────────────

with gr.Blocks(
    title="Westbrook Unofficial Housing Guide",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # 🏠 Westbrook University Unofficial Housing Guide
        **The student-sourced guide to off-campus housing near Westbrook.**
        Ask anything about apartments, landlords, leases, neighborhoods, and more.
        Answers are grounded in real student reviews and housing guides — not made up.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            question_box = gr.Textbox(
                label="Your Question",
                placeholder="e.g. Which apartments are closest to campus?",
                lines=2,
            )
        with gr.Column(scale=1):
            ask_btn = gr.Button("Ask", variant="primary", size="lg")

    with gr.Row():
        answer_box = gr.Textbox(
            label="Answer",
            lines=10,
            interactive=False,
            placeholder="Your answer will appear here...",
        )

    sources_box = gr.Textbox(
        label="Retrieved from (sources)",
        lines=4,
        interactive=False,
        placeholder="Source documents will appear here...",
    )

    gr.Examples(
        examples=[
            ["Which apartment is cheapest and closest to Westbrook campus?"],
            ["What do students say about Maplewood Apartments' management?"],
            ["How do I protect my security deposit when moving out?"],
            ["What bus routes serve off-campus housing near Westbrook?"],
            ["What should I check before signing a lease?"],
            ["Is Riverside Commons a good option for students without a car?"],
            ["What are the pros and cons of The Lofts on Cedar?"],
            ["How much should I budget for utilities in a Westbrook apartment?"],
        ],
        inputs=question_box,
        label="Example Questions",
    )

    # Wire up interactions
    ask_btn.click(
        fn=handle_query,
        inputs=question_box,
        outputs=[answer_box, sources_box],
    )
    question_box.submit(
        fn=handle_query,
        inputs=question_box,
        outputs=[answer_box, sources_box],
    )

    gr.Markdown(
        """
        ---
        *Answers are generated from student-contributed documents only.
        Always verify important details directly with landlords.*
        """
    )


if __name__ == "__main__":
    print("Starting Westbrook Unofficial Housing Guide...")
    print("Building vector store (first run may take 30-60 seconds)...")
    demo.launch(show_error=True)
