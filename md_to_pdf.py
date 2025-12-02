#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将Markdown文件转换为PDF的脚本
"""

import markdown
from weasyprint import HTML
import os
import sys


def markdown_to_pdf(md_file_path, pdf_file_path):
    """
    将Markdown文件转换为PDF文件
    
    Args:
        md_file_path: Markdown文件路径
        pdf_file_path: 输出PDF文件路径
    """
    try:
        # 读取Markdown文件
        with open(md_file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # 将Markdown转换为HTML
        html_content = markdown.markdown(md_content, extensions=['tables'])
        
        # 添加基本样式
        html_with_style = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>学术论文</title>
            <style>
                body {
                    font-family: SimSun, serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 210mm;
                    margin: 0 auto;
                    padding: 2cm;
                }
                h1, h2, h3 {
                    color: #000;
                    margin-top: 1.5em;
                    margin-bottom: 0.5em;
                }
                h1 {
                    text-align: center;
                    font-size: 2em;
                }
                h2 {
                    border-bottom: 1px solid #ccc;
                    padding-bottom: 0.3em;
                    font-size: 1.5em;
                }
                p {
                    margin-bottom: 1em;
                    text-align: justify;
                    text-indent: 2em;
                }
                ul, ol {
                    margin-bottom: 1em;
                    padding-left: 2em;
                }
                li {
                    margin-bottom: 0.5em;
                }
                blockquote {
                    margin: 1em 0;
                    padding: 0.5em 1em;
                    border-left: 4px solid #ddd;
                    background-color: #f9f9f9;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin: 1em 0;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                }
            </style>
        </head>
        <body>
            {content}
        </body>
        </html>
        """
        
        html_with_style = html_with_style.format(content=html_content)
        
        # 将HTML转换为PDF
        HTML(string=html_with_style).write_pdf(pdf_file_path)
        
        print(f"成功将 {md_file_path} 转换为 {pdf_file_path}")
        return True
        
    except Exception as e:
        print(f"转换失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 默认参数
    default_md_file = "sample_paper.md"
    default_pdf_file = "sample_paper.pdf"
    
    # 使用命令行参数或默认值
    md_file = sys.argv[1] if len(sys.argv) > 1 else default_md_file
    pdf_file = sys.argv[2] if len(sys.argv) > 2 else default_pdf_file
    
    # 检查输入文件是否存在
    if not os.path.exists(md_file):
        print(f"错误: 输入文件 {md_file} 不存在")
        sys.exit(1)
    
    # 执行转换
    markdown_to_pdf(md_file, pdf_file)
