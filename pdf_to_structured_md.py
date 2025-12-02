# 文件名: pdf_to_structured_md.py
import pymupdf4llm
import instructor
from pydantic import BaseModel, Field
from typing import List, Optional, Callable
import openai  # 或者用 anthropic
import os
from pathlib import Path
from datetime import datetime
import traceback
import re

# ================== 1. 定义想要的结构 ==================
class ReferenceItem(BaseModel):
    raw: str  # 原始参考文献字符串

class SectionContent(BaseModel):
    title: str = Field(..., description="章节标题")
    content: str = Field(..., description="该章节的完整正文内容（已清洗）")

class PaperStructure(BaseModel):
    title: str = Field(..., description="论文标题")
    authors: Optional[str] = None
    abstract: str = Field(..., description="摘要全文")
    introduction: str
    literature_review: Optional[str] = Field(None, description="文献综述部分，可能叫 Background/Related Work 等")
    methodology: str = Field(..., description="研究方法/实验部分")
    innovation_points: Optional[str] = Field(None, description="论文的主要创新点总结")
    experimental_arrangement: Optional[str] = Field(None, description="具体的实验安排、数据集、对比实验等细节")
    conclusion: str
    references: List[ReferenceItem] = Field(..., description="参考文献列表，每条是一整条文献")

# ================== 2. 使用 Instructor 增强客户端 ==================
# 使用 Qwen 模型配置
client = instructor.from_openai(openai.OpenAI(
    api_key="sk-SR1dZfCptyv3RhBH8d3fC721F2874a30A809BcD45c28Aa35",
    base_url="https://deepseek.perfxlab.cn/v1"
))

# 如果你想用 Claude（强烈推荐学术论文）：
# from anthropic import Anthropic
# client = instructor.from_anthropic(Anthropic())

# ================== 辅助函数：智能章节提取 ==================
def extract_section_by_keywords(md_text: str, start_keywords: List[str], end_keywords: List[str], max_length: int = 50000) -> str:
    """
    根据关键词提取章节内容
    
    Args:
        md_text: 完整的Markdown文本
        start_keywords: 起始关键词列表（如 ["Abstract", "## Abstract"]）
        end_keywords: 结束关键词列表（如 ["Introduction", "## Introduction"]）
        max_length: 最大返回长度
    
    Returns:
        提取的章节内容
    """
    # 找到起始位置
    start_pos = -1
    for keyword in start_keywords:
        pattern = re.compile(rf'^#{1,3}\s*{re.escape(keyword)}', re.IGNORECASE | re.MULTILINE)
        match = pattern.search(md_text)
        if match:
            start_pos = match.start()
            break
    
    if start_pos == -1:
        # 如果没找到标题格式，尝试纯文本匹配
        for keyword in start_keywords:
            pos = md_text.lower().find(keyword.lower())
            if pos != -1:
                start_pos = pos
                break
    
    if start_pos == -1:
        return ""
    
    # 找到结束位置
    end_pos = len(md_text)
    for keyword in end_keywords:
        pattern = re.compile(rf'^#{1,3}\s*{re.escape(keyword)}', re.IGNORECASE | re.MULTILINE)
        match = pattern.search(md_text, start_pos + 10)  # 从起始位置后一点开始搜索
        if match:
            end_pos = match.start()
            break
    
    # 提取内容并限制长度
    section_text = md_text[start_pos:end_pos]
    if len(section_text) > max_length:
        section_text = section_text[:max_length] + "\n\n[内容过长，已截断]"
    
    return section_text.strip()

# ================== 3. 主函数 ==================
def pdf_to_structured_markdown(
    pdf_path: str,
    output_md_path: Optional[str] = None,
    model: str = "Qwen3-Next-80B-Instruct",  # 使用 Qwen 模型
    max_tokens: int = 160_000,   # Claude 3.5 支持 200k，GPT-4o 支持 128k
    progress_callback: Optional[Callable[[str], None]] = None,
    selected_sections: Optional[List[str]] = None,
) -> str:
    """
    将 PDF 学术论文转换为结构化 Markdown
    
    Args:
        pdf_path: PDF 文件路径
        output_md_path: 输出 Markdown 文件路径（可选）
        model: 使用的模型名称
        max_tokens: 最大生成 token 数
        progress_callback: 进度回调函数（可选）
        selected_sections: 需要生成的章节列表（可选）
    
    Returns:
        生成的 Markdown 文本
    """
    def log(msg: str):
        """统一的日志输出"""
        print(msg)
        if progress_callback:
            progress_callback(msg)
    
    # Step 1: 用 pymupdf4llm 把 PDF 转成高质量 Markdown（保留层级）
    log(f"正在解析 PDF: {Path(pdf_path).name}...")
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path,
                                         show_progress=False,
                                         write_images=False,
                                         page_chunks=False)
        
        # 添加空值检查
        if md_text is None:
            log("⚠️ 解析PDF失败，返回了空结果，尝试使用基本文本提取...")
            # 使用基本的pymupdf提取文本作为备用方案
            import fitz
            doc = fitz.open(pdf_path)
            md_text = ""
            for page_num, page in enumerate(doc):
                md_text += f"# 第 {page_num + 1} 页\n\n"
                md_text += page.get_text() + "\n\n"
            doc.close()
        
        log(f"原始文档长度: {len(md_text)} 字符")
    except Exception as e:
        log(f"⚠️ pymupdf4llm解析失败: {str(e)}")
        # 使用备用方案
        import fitz
        doc = fitz.open(pdf_path)
        md_text = ""
        for page_num, page in enumerate(doc):
            md_text += f"# 第 {page_num + 1} 页\n\n"
            md_text += page.get_text() + "\n\n"
        doc.close()
        log(f"备用提取方案完成，文档长度: {len(md_text)} 字符")
    
    # 智能章节提取：处理超长文档
    MAX_CHARS = 80000  # 约 53k tokens，留出安全余量
    
    if len(md_text) > MAX_CHARS:
        log(f"⚠️ 文档过长，启用智能章节提取...")
        
        # 定义章节关键词（支持多种常见写法）
        sections = {
            "title_and_abstract": {
                "start": ["Abstract", "ABSTRACT"],
                "end": ["Introduction", "INTRODUCTION", "Background", "1."]
            },
            "introduction": {
                "start": ["Introduction", "INTRODUCTION", "1. Introduction", "1 Introduction"],
                "end": ["Related Work", "Literature Review", "Background", "Method", "2."]
            },
            "method": {
                "start": ["Method", "Methodology", "Approach", "Experiments", "3.", "4."],
                "end": ["Results", "Evaluation", "Conclusion", "Discussion"]
            },
            "conclusion": {
                "start": ["Conclusion", "CONCLUSION", "Discussion", "Summary"],
                "end": ["References", "REFERENCES", "Acknowledgment", "Appendix"]
            },
            "references": {
                "start": ["References", "REFERENCES", "Bibliography"],
                "end": ["Appendix", ""]
            }
        }
        
        # 提取各部分
        extracted_parts = {}
        for section_name, keywords in sections.items():
            content = extract_section_by_keywords(
                md_text, 
                keywords["start"], 
                keywords["end"],
                max_length=30000
            )
            extracted_parts[section_name] = content
            if content:
                log(f"  提取 {section_name}: {len(content)} 字符")
        
        # 组装提示词
        text_for_llm = f"""以下是一篇学术论文的关键章节内容（已智能提取）：

### 标题和摘要部分
{extracted_parts['title_and_abstract'][:15000]}

### 引言部分
{extracted_parts['introduction'][:15000]}

### 方法部分
{extracted_parts['method'][:20000]}

### 结论部分
{extracted_parts['conclusion'][:10000]}

### 参考文献部分
{extracted_parts['references'][:20000]}

请根据以上内容提取论文的结构化信息。如果某些字段找不到，请填空字符串。
请特别注意提取【创新点】和【实验安排】细节。"""
        
    else:
        # 文档不太长，使用完整内容
        log("文档长度适中，使用完整内容进行提取")
        text_for_llm = f"""以下是一篇学术论文的完整 Markdown 格式内容（已经过布局还原）：
    
{md_text}

请严格按照 JSON 格式提取以下字段，不要遗漏，不要胡乱添加内容。
如果某些部分不存在，请填入空字符串 "" 或 null。
"""

    log("正在调用大模型进行结构化提取（这可能需要 20-60 秒）...")
    
    # Step 2: 调用大模型强制结构化输出
    structured: PaperStructure = client.chat.completions.create(
        model=model,
        response_model=PaperStructure,
        temperature=0.0,
        max_tokens=16000,  # 增加到16000以容纳完整的论文内容
        messages=[
            {
                "role": "user",
                "content": text_for_llm
            }
        ],
    )

    # Step 3: 拼成最终的 Markdown
    def should_include(section_key: str) -> bool:
        """检查是否应该包含该章节"""
        if selected_sections is None:  # 如果没有指定，默认全部包含
            return True
        return section_key in selected_sections

    final_md = f"# {structured.title.strip()}\n\n"
    if structured.authors:
        final_md += f"**作者**: {structured.authors.strip()}\n\n"

    if should_include("abstract"):
        final_md += f"## 摘要\n{structured.abstract.strip()}\n\n"
    
    if should_include("introduction"):
        final_md += f"## 引言\n{structured.introduction.strip()}\n\n"

    if should_include("literature_review") and structured.literature_review:
        final_md += f"## 文献综述\n{structured.literature_review.strip()}\n\n"

    if should_include("innovation_points") and structured.innovation_points:
        final_md += f"## 创新点\n{structured.innovation_points.strip()}\n\n"

    if should_include("methodology"):
        final_md += f"## 研究方法\n{structured.methodology.strip()}\n\n"

    if should_include("experimental_arrangement") and structured.experimental_arrangement:
        final_md += f"## 实验安排\n{structured.experimental_arrangement.strip()}\n\n"

    if should_include("conclusion"):
        final_md += f"## 结论\n{structured.conclusion.strip()}\n\n"

    if should_include("references"):
        final_md += "## 参考文献\n"
        for i, ref in enumerate(structured.references, 1):
            final_md += f"{i}. {ref.raw.strip()}\n"

    # 保存文件
    if not output_md_path:
        output_md_path = Path(pdf_path).with_suffix(".structured.md")
    
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(final_md)
    
    log(f"✓ 成功保存：{output_md_path}")
    return final_md


# ================== 4. 批量处理函数 ==================
def batch_process_pdfs(
    folder_path: str,
    output_folder: Optional[str] = None,
    model: str = "Qwen3-Next-80B-Instruct",
    max_tokens: int = 16000,
    progress_callback: Optional[Callable[[str], None]] = None,
    selected_sections: Optional[List[str]] = None,
) -> dict:
    """
    批量处理文件夹中的所有 PDF 文件
    
    Args:
        folder_path: 包含 PDF 文件的文件夹路径
        output_folder: 输出文件夹路径（可选，默认为输入文件夹下的 output 子文件夹）
        model: 使用的模型名称
        max_tokens: 最大生成 token 数
        progress_callback: 进度回调函数（可选）
        selected_sections: 需要生成的章节列表（可选）
    
    Returns:
        处理结果字典，包含成功和失败的文件信息
    """
    def log(msg: str):
        """统一的日志输出"""
        print(msg)
        if progress_callback:
            progress_callback(msg)
    
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")
    
    # 查找所有 PDF 文件
    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        log(f"⚠ 文件夹中没有找到 PDF 文件: {folder_path}")
        return {"success": [], "failed": []}
    
    # 创建输出文件夹
    if output_folder is None:
        output_folder = folder / "output"
    else:
        output_folder = Path(output_folder)
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    log(f"\n开始批量处理 {len(pdf_files)} 个 PDF 文件...")
    log(f"输出文件夹: {output_folder}\n")
    
    results = {
        "success": [],
        "failed": [],
        "total": len(pdf_files),
        "start_time": datetime.now()
    }
    
    for idx, pdf_file in enumerate(pdf_files, 1):
        log(f"\n{'='*60}")
        log(f"[{idx}/{len(pdf_files)}] 处理: {pdf_file.name}")
        log(f"{'='*60}")
        
        try:
            # 构造输出文件路径
            output_path = output_folder / f"{pdf_file.stem}.structured.md"
            
            # 处理单个文件
            pdf_to_structured_markdown(
                pdf_path=str(pdf_file),
                output_md_path=str(output_path),
                model=model,
                max_tokens=max_tokens,
                progress_callback=progress_callback,
                selected_sections=selected_sections
            )
            
            results["success"].append({
                "file": pdf_file.name,
                "output": str(output_path)
            })
            
        except Exception as e:
            error_msg = f"✗ 处理失败: {pdf_file.name}\n错误: {str(e)}"
            log(error_msg)
            
            results["failed"].append({
                "file": pdf_file.name,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
    
    # 生成处理报告
    results["end_time"] = datetime.now()
    results["duration"] = (results["end_time"] - results["start_time"]).total_seconds()
    
    log(f"\n{'='*60}")
    log(f"批量处理完成!")
    log(f"{'='*60}")
    log(f"总文件数: {results['total']}")
    log(f"成功: {len(results['success'])}")
    log(f"失败: {len(results['failed'])}")
    log(f"耗时: {results['duration']:.1f} 秒")
    
    if results['failed']:
        log(f"\n失败的文件:")
        for item in results['failed']:
            log(f"  - {item['file']}: {item['error']}")
    
    # 保存处理报告
    report_path = output_folder / f"处理报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"批量处理报告\n")
        f.write(f"{'='*60}\n")
        f.write(f"处理时间: {results['start_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总文件数: {results['total']}\n")
        f.write(f"成功: {len(results['success'])}\n")
        f.write(f"失败: {len(results['failed'])}\n")
        f.write(f"耗时: {results['duration']:.1f} 秒\n\n")
        
        if results['success']:
            f.write(f"成功处理的文件:\n")
            for item in results['success']:
                f.write(f"  ✓ {item['file']} -> {item['output']}\n")
        
        if results['failed']:
            f.write(f"\n失败的文件:\n")
            for item in results['failed']:
                f.write(f"  ✗ {item['file']}\n")
                f.write(f"     错误: {item['error']}\n")
                f.write(f"     详细信息:\n{item['traceback']}\n\n")
    
    log(f"\n处理报告已保存: {report_path}")
    
    return results


# ================== 使用示例 ==================
if __name__ == "__main__":
    pdf_file = r"D:\GraduateStudent\导师\实验素材论文\1.基线算法源代码\AutoPrompt Eliciting Knowledge from Language Models with Automatically Generated Prompts.pdf"  # 改成你的路径
    pdf_to_structured_markdown(pdf_file)