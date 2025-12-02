import gradio as gr
from pathlib import Path
from pdf_to_structured_md import pdf_to_structured_markdown
import tempfile
import shutil
import os

# ================== 处理逻辑 ==================
def process_uploaded_files(files, selected_contents, model_name, max_tokens, progress=gr.Progress()):
    """处理上传的文件并根据选择生成内容"""
    if not files:
        return "请先上传文件！", None
    
    if not selected_contents:
        return "请至少选择一项要生成的内容！", None

    # 映射 UI 选项到内部键值
    mapping = {
        "摘要": "abstract",
        "引言": "introduction",
        "综述": "literature_review",
        "创新点": "innovation_points",
        "实验安排": "experimental_arrangement"
    }
    
    selected_keys = []
    for content in selected_contents:
        if content in mapping:
            selected_keys.append(mapping[content])
    
    # 特殊处理：如果选了实验安排，建议也包含方法论
    if "experimental_arrangement" in selected_keys:
        if "methodology" not in selected_keys:
            selected_keys.append("methodology")

    logs = []
    results = []
    
    # 创建临时输出目录
    output_dir = Path(tempfile.mkdtemp())
    
    logs.append(f"开始处理 {len(files)} 个文件...")
    logs.append(f"选择生成的内容: {', '.join(selected_contents)}")
    
    for i, file in enumerate(files):
        file_path = Path(file.name)
        
        # 简单的文件类型检查
        if file_path.suffix.lower() != '.pdf':
            logs.append(f"⚠ 跳过非 PDF 文件: {file_path.name} (目前 Demo 仅支持 PDF)")
            continue
            
        msg = f"正在处理 [{i+1}/{len(files)}] {file_path.name}..."
        progress((i)/len(files), desc=msg)
        logs.append(f"\n--- {msg} ---")
        
        try:
            output_path = output_dir / (file_path.stem + ".md")
            
            pdf_to_structured_markdown(
                pdf_path=str(file_path),
                output_md_path=str(output_path),
                model=model_name,
                max_tokens=int(max_tokens),
                selected_sections=selected_keys,
                progress_callback=None 
            )
            
            results.append(str(output_path))
            logs.append(f"✓ 成功生成: {output_path.name}")
            
        except Exception as e:
            logs.append(f"✗ 处理失败: {file_path.name}")
            logs.append(f"  错误信息: {str(e)}")

    progress(1.0, desc="全部处理完成!")
    logs.append("\n全部处理完成！")
    
    if not results:
        logs.append("⚠ 没有生成任何文件，请检查上传的文件是否为 PDF。")
    
    return "\n".join(logs), results

# ================== Gradio 界面 ==================
def create_ui():
    with gr.Blocks(title="论文处理系统", theme=gr.themes.Soft()) as app:
        
        # 标题
        gr.Markdown("<h1 style='text-align: center;'>论文处理系统</h1>")
        
        # 区域 1: 上传文件
        with gr.Group():
            with gr.Column(variant="panel"):
                gr.Markdown("### 上传论文文件")
                gr.Markdown("支持格式：PDF、Word、TXT (Demo版本仅支持PDF)")
                file_input = gr.File(
                    label="选择文件",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".txt"],
                    type="filepath",
                    height=100
                )

        # 区域 2: 选择内容
        with gr.Group():
            with gr.Column(variant="panel"):
                gr.Markdown("### 选择要生成的内容")
                content_choices = ["摘要", "引言", "综述", "创新点", "实验安排"]
                checkbox_group = gr.CheckboxGroup(
                    choices=content_choices,
                    value=["摘要", "引言"], # 默认选中
                    label="",
                    container=False
                )
        
        # 按钮
        process_btn = gr.Button("开始处理", variant="primary", size="lg")
        
        # 隐藏的高级设置 (折叠)
        with gr.Accordion("高级设置", open=False):
            model_name = gr.Dropdown(
                choices=["Qwen3-Next-80B-Instruct", "gpt-4o", "claude-3-5-sonnet-20241022"],
                value="Qwen3-Next-80B-Instruct",
                label="选择模型"
            )
            max_tokens = gr.Number(
                value=16000,
                label="最大 Token 数"
            )
        
        # 输出区域
        with gr.Group():
            log_output = gr.Textbox(label="处理日志", lines=10)
            result_files = gr.File(label="生成的文件", file_count="multiple", interactive=False)

        # 绑定事件
        process_btn.click(
            fn=process_uploaded_files,
            inputs=[file_input, checkbox_group, model_name, max_tokens],
            outputs=[log_output, result_files]
        )
        
    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch(inbrowser=True, share=False)
