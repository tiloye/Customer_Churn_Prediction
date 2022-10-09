import gradio as gr
from metrics import show_metrics, estimate_costs

# buid app UI
with gr.Blocks(css=".row {align-items: center}", title="Precision-Recall Trade-off for Churn Prediction Model") as demo:

    with gr.Row():
        pr_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.5)
    with gr.Row():
        with gr.Column(min_width=500):
            pr_confusion_matrix = gr.Plot(label="True Value vs Predicted Value")
        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column():
                    accuracy = gr.Number(label="Accuracy Score")
                    precision = gr.Number(label="Precision Score")
                    recall = gr.Number(label="Recall Score")
                with gr.Column():
                    crc = gr.Number(value=50, label="Customer Retention Cost (€)")
                    cac = gr.Number(value=200, label="Customer Acquisiton Cost(€)")
            with gr.Row():
                with gr.Column():
                    total_crc = gr.Number(label="Total Customer Retention Cost(€)")
                    total_cac = gr.Number(label="Total Customer Aquisition Cost(€)")
                with gr.Column():
                    total_amount = gr.Number(label="Total Amount Spent(€)")
                    amount_saved = gr.Number(label="Amount Saved(€)")

    demo.load(
        fn=show_metrics,
        inputs=[pr_threshold],
        outputs=[pr_confusion_matrix, accuracy, precision, recall]
    )
    demo.load(
        fn=estimate_costs,
        inputs=[pr_threshold, crc, cac],
        outputs=[total_crc, total_cac, total_amount, amount_saved]
    )
    pr_threshold.change(
        fn=show_metrics,
        inputs=[pr_threshold],
        outputs=[pr_confusion_matrix, accuracy, precision, recall]   
    )
    pr_threshold.change(
        fn=estimate_costs,
        inputs=[pr_threshold, crc, cac],
        outputs=[total_crc, total_cac, total_amount, amount_saved]
    )
    crc.change(
        fn=estimate_costs,
        inputs=[pr_threshold, crc, cac],
        outputs=[total_crc, total_cac, total_amount, amount_saved]
    )
    cac.change(
        fn=estimate_costs,
        inputs=[pr_threshold, crc, cac],
        outputs=[total_crc, total_cac, total_amount, amount_saved]
    )

if __name__ == "__main__":
    demo.launch()