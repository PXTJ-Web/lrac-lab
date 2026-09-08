import gradio as gr
from enhance import enhance_file, MODEL_IDS

def process(filepath, model_name):
    if filepath is None:
        raise gr.Error("请先上传音频或用麦克风录一段")
    out = enhance_file(filepath, model_name)
    return filepath, out

with gr.Blocks(title="语音增强对比 Demo") as demo:
    gr.Markdown(
        "## 🎧 语音增强对比 Demo\n"
        "上传或录制一段带噪音频，选择模型，对比增强前后的效果。\n\n"
        "提示：SepFormer 效果好但很慢（CPU 上 10 秒音频约 1 分钟），急就先用 MetricGAN+。"
    )
    with gr.Row():
        inp = gr.Audio(label="输入音频（上传或录音）", type="filepath")
        with gr.Column():
            model_choice = gr.Radio(
                list(MODEL_IDS), value=list(MODEL_IDS)[0], label="增强模型")
            btn = gr.Button("开始增强", variant="primary")
    with gr.Row():
        orig = gr.Audio(label="原始（带噪）", interactive=False)
        out = gr.Audio(label="增强后", interactive=False)
    btn.click(process, inputs=[inp, model_choice], outputs=[orig, out])

if __name__ == "__main__":
    demo.launch(share=True)   # share=True 会生成一个临时公网链接