# 论文处理系统 - Web应用
import os
import tempfile
import re
from flask import Flask, request, render_template, send_file, jsonify
from pathlib import Path
import threading
import uuid

# 导入现有功能
from pdf_to_structured_md import pdf_to_structured_markdown

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'doc', 'docx', 'txt'}

# 处理任务状态跟踪
tasks = {}
# 批次任务跟踪
batches = {}

# 检查文件类型是否允许
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 添加直接从PDF提取文本的功能
import fitz  # pymupdf

def extract_raw_text_from_pdf(pdf_path):
    """直接从PDF中提取原始文本"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"从PDF提取原始文本时出错: {str(e)}")
        return ""

# 处理论文文件
def process_paper(task_id, file_path, content_types, original_filename):
    """处理论文文件，生成指定内容"""
    try:
        tasks[task_id]['status'] = 'processing'
        
        # 首先转换为结构化Markdown
        temp_md_path = Path(app.config['UPLOAD_FOLDER']) / f"{uuid.uuid4()}.md"
        
        # 调用现有功能进行PDF处理
        structured_content = pdf_to_structured_markdown(
            pdf_path=str(file_path),
            output_md_path=str(temp_md_path),
            model="Qwen3-Next-80B-Instruct",
            max_tokens=16000,
            progress_callback=lambda msg: tasks[task_id].update({'progress': msg})
        )
        
        # 同时提取原始PDF文本，作为备份提取源
        raw_pdf_text = extract_raw_text_from_pdf(str(file_path))
        
        # 提取请求的内容类型
        result = {}
        
        tasks[task_id].update({'progress': '正在提取所需内容...'})
        
        # 打印详细的调试信息
        print("\n=== 调试信息 ===")
        print(f"文件路径: {file_path}")
        print(f"原始文件名: {original_filename}")
        print(f"请求的内容类型: {content_types}")
        
        # 打印结构化内容的前1000字符
        print("\n=== 结构化内容前1000字符 ===")
        print(structured_content[:1000])
        
        # 打印原始PDF文本的前1000字符
        print("\n=== 原始PDF文本前1000字符 ===")
        print(raw_pdf_text[:1000])
        
        # 创建一个综合内容，包含结构化内容和原始文本，增加提取成功率
        combined_content = structured_content + "\n\n" + raw_pdf_text
        
        # 改进的提取逻辑：尝试从多个来源提取
        for content_type in content_types:
            tasks[task_id].update({'progress': f'正在提取{content_type}...'})
            
            if content_type == '摘要':
                # 尝试从综合内容中提取摘要
                abstract = extract_section_improved(combined_content, ['摘要', 'Abstract', '# 摘要', '## 摘要', 'ABSTRACT'])
                if not abstract:
                    # 尝试更宽松的匹配
                    abstract = extract_abstract(combined_content)
                result['摘要'] = abstract if abstract else "未找到摘要内容"
            
            elif content_type == '引言':
                # 尝试从综合内容中提取引言
                introduction = extract_section_improved(combined_content, ['引言', 'Introduction', '# 引言', '## 引言', 'INTRODUCTION', '导言'])
                if not introduction:
                    # 尝试更宽松的匹配
                    introduction = extract_introduction(combined_content)
                result['引言'] = introduction if introduction else "未找到引言内容"
            
            elif content_type == '综述':
                # 尝试从综合内容中提取综述
                literature_review = extract_section_improved(combined_content, ['综述', '文献综述', 'Related Work', '# 文献综述', '## 文献综述', 'LITERATURE REVIEW'])
                if not literature_review:
                    # 尝试更宽松的匹配
                    literature_review = extract_literature_review(combined_content)
                result['综述'] = literature_review if literature_review else "未找到综述内容"
            
            elif content_type == '创新点':
                # 提取创新点（从结论和研究方法中）
                conclusion = extract_section_improved(combined_content, ['结论', 'Conclusion', '# 结论', '## 结论', 'CONCLUSION'])
                methodology = extract_section_improved(combined_content, ['研究方法', 'Methodology', '# 研究方法', '## 研究方法', 'METHODOLOGY'])
                
                # 提取创新点
                innovation_points = extract_innovation_points(methodology, conclusion, combined_content)
                result['创新点'] = innovation_points if innovation_points else "未找到创新点内容"
            
            elif content_type == '实验安排':
                # 从研究方法中提取实验安排
                methodology = extract_section_improved(combined_content, ['研究方法', 'Methodology', '# 研究方法', '## 研究方法', 'METHODOLOGY'])
                experiment_design = extract_experiment_design(methodology, combined_content)
                result['实验安排'] = experiment_design if experiment_design else "未找到实验安排内容"
        
        # 保存结果到临时文件，文件名包含原始文件名
        base_filename = os.path.splitext(original_filename)[0]
        # 替换文件名中的特殊字符，避免文件系统问题
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', base_filename)
        result_file = Path(app.config['UPLOAD_FOLDER']) / f"{safe_filename}_论文处理结果.txt"
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"# 论文处理结果: {original_filename}\n\n")
            for content_type, content in result.items():
                f.write(f"## {content_type}\n\n{content}\n\n{'='*50}\n\n")
        
        tasks[task_id].update({
            'status': 'completed',
            'result': result,
            'result_file': str(result_file),
            'original_filename': original_filename
        })
        
        # 更新批次信息
        batch_id = tasks[task_id].get('batch_id')
        if batch_id and batch_id in batches:
            batches[batch_id]['completed_tasks'] += 1
            batches[batch_id]['task_results'].append({
                'task_id': task_id,
                'file_name': original_filename,
                'result_file': str(result_file),
                'status': 'completed'
            })
        
    except Exception as e:
        print(f"处理错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        error_info = {
            'status': 'error',
            'error': str(e)
        }
        tasks[task_id].update(error_info)
        
        # 更新批次信息
        batch_id = tasks[task_id].get('batch_id')
        if batch_id and batch_id in batches:
            batches[batch_id]['completed_tasks'] += 1
            batches[batch_id]['task_results'].append({
                'task_id': task_id,
                'file_name': original_filename,
                'status': 'error',
                'error': str(e)
            })

def extract_section_improved(content, possible_headings):
    """改进的章节提取函数，支持多种可能的标题格式"""
    import re
    
    # 尝试多种匹配策略
    strategies = [
        # 策略1: 严格匹配标题行（用于Markdown格式）
        lambda h: (f"(?m)^{re.escape(h)}\\s*[\\s\\S]*?(?=(?:^#|$))", re.MULTILINE),
        # 策略2: 匹配任何包含标题的行（不区分大小写）
        lambda h: (f"(?i){re.escape(h)}[\\s:]*[\\s\\S]*?(?=(?:^#|$))", re.MULTILINE),
        # 策略3: 匹配标题前后有特殊字符的情况（如：【摘要】、[摘要]等）
        lambda h: (f"(?i)[[【]?{re.escape(h)}[]】]?[\\s:]*[\\s\\S]*?(?=(?:^#|$))", re.MULTILINE),
        # 策略4: 匹配标题作为段落开头的情况
        lambda h: (f"(?i)^[\\s]*{re.escape(h)}[\\s:]*[\\s\\S]*?(?=(?:^[#一]|$))", re.MULTILINE),
    ]
    
    for heading in possible_headings:
        for strategy in strategies:
            pattern, flags = strategy(heading)
            match = re.search(pattern, content, flags)
            
            if match:
                # 获取匹配的内容
                section_content = match.group(0)
                
                # 移除标题部分（使用多种可能的标题格式）
                for h in possible_headings:
                    # 尝试移除各种格式的标题
                    section_content = re.sub(f"(?i)^{re.escape(h)}[\\s:]*", "", section_content, flags=re.MULTILINE)
                    section_content = re.sub(f"(?i)[[【]?{re.escape(h)}[]】]?[\\s:]*", "", section_content, flags=re.MULTILINE)
                
                # 清理内容
                section_content = section_content.strip()
                
                # 如果内容不为空且足够长，返回结果
                if section_content and len(section_content) > 50:  # 确保内容长度合理
                    print(f"成功匹配标题: {heading}, 内容长度: {len(section_content)}")
                    return section_content
    
    # 如果没有找到任何匹配
    print(f"没有找到精确匹配的标题: {possible_headings}")
    return None

# 专门的摘要提取函数
def extract_abstract(content):
    """专门用于提取摘要的函数，使用多种模式"""
    import re
    
    # 摘要的常见模式
    patterns = [
        # 中英文摘要，支持多种格式
        r"(?i)(?:abstract|摘要)[\\s:：]*([\\s\\S]{100,3000}?)(?=\\n\\s*(?:[#一二三四五六七八九十]|1\.|2\.|introduction|引言|\\Z))",
        # 摘要作为文档开头部分
        r"(?i)^[\\s\\S]{100,3000}?\\s*(?:[#一二三四五六七八九十]|1\.|2\.|introduction|引言)",
        # 寻找包含"摘要"关键词的段落
        r"(?i)[\\s\\S]*?(摘要[\\s:：]*[\\s\\S]{100,3000}?)[\\s\\S]*?",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            abstract = match.group(1) if len(match.groups()) > 0 else match.group(0)
            
            # 清理摘要内容
            abstract = re.sub(r"(?i)^[\\s:：]*abstract[\\s:：]*", "", abstract)
            abstract = re.sub(r"(?i)^[\\s:：]*摘要[\\s:：]*", "", abstract)
            abstract = abstract.strip()
            
            if abstract and len(abstract) > 100:  # 确保摘要长度合理
                print(f"使用专门的摘要提取函数找到摘要，长度: {len(abstract)}")
                return abstract
    
    return None

# 专门的引言提取函数
def extract_introduction(content):
    """专门用于提取引言的函数，使用多种模式"""
    import re
    
    # 引言的常见模式
    patterns = [
        # 中英文引言，支持多种格式
        r"(?i)(?:introduction|引言|导言)[\\s:：]*([\\s\\S]{200,5000}?)(?=\\n\\s*(?:[#一二三四五六七八九十]|1\.|2\.|文献综述|研究方法|related|literature|method|\\Z))",
        # 寻找第二个主要章节（通常是引言）
        r"(?i)(?:[#一二三四五六七八九十]|1\.|2\.)[\\s:：]*[\\s\\S]{200,5000}?\\s*(?:[#一二三四五六七八九十]|1\.|2\.|\\Z)",
        # 寻找包含"引言"关键词的段落
        r"(?i)[\\s\\S]*?(引言[\\s:：]*[\\s\\S]{200,5000}?)[\\s\\S]*?",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            introduction = match.group(1) if len(match.groups()) > 0 else match.group(0)
            
            # 清理引言内容
            introduction = re.sub(r"(?i)^[\\s:：]*introduction[\\s:：]*", "", introduction)
            introduction = re.sub(r"(?i)^[\\s:：]*引言[\\s:：]*", "", introduction)
            introduction = re.sub(r"(?i)^[\\s:：]*导言[\\s:：]*", "", introduction)
            introduction = introduction.strip()
            
            if introduction and len(introduction) > 200:  # 确保引言长度合理
                print(f"使用专门的引言提取函数找到引言，长度: {len(introduction)}")
                return introduction
    
    return None

# 专门的文献综述提取函数
def extract_literature_review(content):
    """专门用于提取文献综述的函数，使用多种模式"""
    import re
    
    # 文献综述的常见模式
    patterns = [
        # 中英文文献综述，支持多种格式
        r"(?i)(?:literature\\s*review|文献综述|相关工作|related\\s*work)[\\s:：]*([\\s\\S]{200,5000}?)(?=\\n\\s*(?:[#一二三四五六七八九十]|1\.|2\.|研究方法|method|\\Z))",
        # 寻找包含"文献综述"关键词的段落
        r"(?i)[\\s\\S]*?(文献综述[\\s:：]*[\\s\\S]{200,5000}?)[\\s\\S]*?",
        # 寻找包含大量参考文献引用的段落（通常是综述）
        r"(?i)[\\s\\S]*?(\\([^)]{4,6}\\)[\\s\\S]{200,5000}?)(?=\\n\\s*(?:[#一二三四五六七八九十]|1\.|2\.|研究方法|method|\\Z))",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            review = match.group(1) if len(match.groups()) > 0 else match.group(0)
            
            # 清理综述内容
            review = re.sub(r"(?i)^[\\s:：]*literature\\s*review[\\s:：]*", "", review)
            review = re.sub(r"(?i)^[\\s:：]*文献综述[\\s:：]*", "", review)
            review = re.sub(r"(?i)^[\\s:：]*相关工作[\\s:：]*", "", review)
            review = re.sub(r"(?i)^[\\s:：]*related\\s*work[\\s:：]*", "", review)
            review = review.strip()
            
            if review and len(review) > 200:  # 确保综述长度合理
                print(f"使用专门的文献综述提取函数找到综述，长度: {len(review)}")
                return review
    
    return None

def extract_innovation_points(methodology, conclusion, combined_content):
    """从研究方法、结论和综合内容中提取创新点"""
    import re
    innovation_points = []
    
    # 创新点的关键词
    innovation_keywords = [
        r"创新", r"novel", r"new", r"突破", r"改进", r"首次", 
        r"贡献", r"contribution", r"novelty", r"创新点", r"innovation"
    ]
    
    # 从研究方法中提取
    if methodology:
        for keyword in innovation_keywords:
            # 使用更灵活的模式，匹配关键词附近的内容
            pattern = f"(?i)(?:^|\\s|[。.!?])[^。.!?]*?{keyword}[^。.!?]*?(?:[。.!?]|$)"
            matches = re.finditer(pattern, methodology)
            for match in matches:
                point = match.group(0).strip()
                if point and point not in innovation_points and len(point) > 10:
                    innovation_points.append(point)
    
    # 从结论中提取
    if conclusion:
        for keyword in innovation_keywords:
            pattern = f"(?i)(?:^|\\s|[。.!?])[^。.!?]*?{keyword}[^。.!?]*?(?:[。.!?]|$)"
            matches = re.finditer(pattern, conclusion)
            for match in matches:
                point = match.group(0).strip()
                if point and point not in innovation_points and len(point) > 10:
                    innovation_points.append(point)
    
    # 直接从综合内容中提取
    for keyword in innovation_keywords:
        pattern = f"(?i)(?:^|\\s|[。.!?])[^。.!?]*?{keyword}[^。.!?]*?(?:[。.!?]|$)"
        matches = re.finditer(pattern, combined_content)
        for match in matches:
            point = match.group(0).strip()
            if point and point not in innovation_points and len(point) > 10:
                innovation_points.append(point)
    
    # 寻找明确标记为"创新点"的部分
    innovation_section_patterns = [
        r"(?i)创新点[\\s:：]*[\\s\\S]{200,2000}?(?=\\n\\s*(?:[#一二三四五六七八九十]|1\.|2\.|\\Z))",
        r"(?i)本文的创新点[\\s:：]*[\\s\\S]{200,2000}?(?=\\n\\s*(?:[#一二三四五六七八九十]|1\.|2\.|\\Z))",
    ]
    
    for pattern in innovation_section_patterns:
        match = re.search(pattern, combined_content, re.MULTILINE)
        if match:
            section = match.group(0)
            # 清理内容
            section = re.sub(r"(?i)^创新点[\\s:：]*", "", section, flags=re.MULTILINE)
            section = re.sub(r"(?i)^本文的创新点[\\s:：]*", "", section, flags=re.MULTILINE)
            section = section.strip()
            
            if section:
                print("找到明确标记的创新点部分")
                return f"论文创新点：\n\n{section}"
    
    # 如果找到了创新点，格式化输出
    if innovation_points:
        result = "论文创新点：\n\n"
        for i, point in enumerate(innovation_points[:10]):  # 最多返回10个创新点
            result += f"{i+1}. {point}\n\n"
        return result
    
    # 如果没有找到明确的创新点，尝试从方法和结论中提取相关内容
    result = "从论文中提取的创新相关内容：\n\n"
    
    if methodology:
        # 提取方法中的关键部分
        method_key_points = re.findall(r"(?i)(?:提出|设计|开发|实现|改进|采用)[^。.!?]*[。.!?]", methodology)
        if method_key_points:
            result += "## 研究方法中的创新点：\n"
            for i, point in enumerate(method_key_points[:5]):
                result += f"{i+1}. {point}\n\n"
        else:
            result += "## 研究方法中的创新：\n"
            result += methodology[:1000] + "\n\n"
    
    if conclusion:
        # 提取结论中的关键部分
        conclusion_key_points = re.findall(r"(?i)(?:创新|贡献|改进|突破)[^。.!?]*[。.!?]", conclusion)
        if conclusion_key_points:
            result += "## 结论中的创新点：\n"
            for i, point in enumerate(conclusion_key_points[:5]):
                result += f"{i+1}. {point}\n\n"
        else:
            result += "## 结论中的创新：\n"
            result += conclusion[:1000]
    
    return result if result != "从论文中提取的创新相关内容：\n\n" else None

def extract_experiment_design(methodology, combined_content):
    """从研究方法和综合内容中提取实验安排"""
    import re
    
    # 实验相关的关键词
    experiment_keywords = [
        r"实验", r"experiment", r"方法", r"method", r"设计", r"design", 
        r"setup", r"procedure", r"流程", r"步骤", r"数据集", r"dataset",
        r"参数", r"parameter", r"配置", r"configuration", r"评估", r"evaluation"
    ]
    
    experiment_sections = []
    
    # 从研究方法中提取
    if methodology:
        for keyword in experiment_keywords:
            # 查找包含实验关键词的段落
            pattern = f"(?i){keyword}[\\s:：]*[\\s\\S]{200,2000}?(?=\\n\\s*(?:[#一二三四五六七八九十]|1\.|2\.|\\Z))"
            matches = re.finditer(pattern, methodology, re.MULTILINE)
            for match in matches:
                section = match.group(0).strip()
                if section and section not in experiment_sections and len(section) > 100:
                    experiment_sections.append(section)
    
    # 直接从综合内容中提取实验相关部分
    for keyword in experiment_keywords:
        pattern = f"(?i){keyword}[\\s:：]*[\\s\\S]{200,2000}?(?=\\n\\s*(?:[#一二三四五六七八九十]|1\.|2\.|\\Z))"
        matches = re.finditer(pattern, combined_content, re.MULTILINE)
        for match in matches:
            section = match.group(0).strip()
            if section and section not in experiment_sections and len(section) > 100:
                experiment_sections.append(section)
    
    # 寻找明确的实验章节
    experiment_chapter_patterns = [
        r"(?i)(?:实验|experiment|实验方法|实验设计|experimental\\s*method|实验设置|实验结果|experimental\\s*results)[\\s:：]*[\\s\\S]{300,3000}?(?=\\n\\s*(?:[#一二三四五六七八九十]|1\.|2\.|\\Z))",
    ]
    
    for pattern in experiment_chapter_patterns:
        match = re.search(pattern, combined_content, re.MULTILINE)
        if match:
            section = match.group(0)
            # 清理内容
            section = re.sub(r"(?i)^(?:实验|experiment|实验方法|实验设计|experimental\\s*method|实验设置|实验结果|experimental\\s*results)[\\s:：]*", "", section, flags=re.MULTILINE)
            section = section.strip()
            
            if section:
                print("找到明确的实验章节")
                return f"实验安排：\n\n{section}"
    
    # 寻找数据集和参数设置
    dataset_pattern = r"(?i)(?:数据集|dataset|数据|data)[\\s:：]*[\\s\\S]{100,1000}?(?=\\n\\s*(?:[#一二三四五六七八九十]|1\.|2\.|\\Z))"
    parameter_pattern = r"(?i)(?:参数|parameter|设置|setup|配置|configuration)[\\s:：]*[\\s\\S]{100,1000}?(?=\\n\\s*(?:[#一二三四五六七八九十]|1\.|2\.|\\Z))"
    
    dataset_match = re.search(dataset_pattern, combined_content, re.MULTILINE)
    parameter_match = re.search(parameter_pattern, combined_content, re.MULTILINE)
    
    if dataset_match or parameter_match:
        result = "实验安排：\n\n"
        if dataset_match:
            result += "## 数据集：\n"
            result += dataset_match.group(0) + "\n\n"
        if parameter_match:
            result += "## 参数设置：\n"
            result += parameter_match.group(0) + "\n\n"
        return result
    
    # 如果找到了实验相关内容，格式化输出
    if experiment_sections:
        result = "实验安排：\n\n"
        for i, section in enumerate(experiment_sections[:3]):  # 最多返回3个实验部分
            result += f"## 实验部分 {i+1}：\n"
            result += section + "\n\n"
        return result
    
    # 如果没有找到明确的实验部分，返回整个研究方法
    if methodology:
        return f"研究方法（实验安排）：\n\n{methodology[:2000]}"
    
    return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({'error': '请选择文件'})
        
        file = request.files['file']
        
        # 检查文件是否为空
        if file.filename == '':
            return jsonify({'error': '请选择文件'})
        
        # 检查文件类型
        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件类型，请上传PDF、Word或TXT文件'})
        
        # 保存文件
        filename = str(uuid.uuid4()) + '_' + file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # 获取选择的内容类型
        content_types = request.form.getlist('content_types')
        
        if not content_types:
            return jsonify({'error': '请至少选择一种要生成的内容'})
        
        # 创建任务ID
        task_id = str(uuid.uuid4())
        
        # 检查是否有批次ID
        batch_id = request.form.get('batch_id')
        task_info = {
            'status': 'pending',
            'file_path': file_path,
            'content_types': content_types,
            'progress': '准备开始处理...',
            'original_filename': file.filename
        }
        
        # 如果有批次ID，将任务添加到批次中
        if batch_id:
            task_info['batch_id'] = batch_id
            if batch_id not in batches:
                batches[batch_id] = {
                    'total_tasks': 0,
                    'completed_tasks': 0,
                    'task_results': []
                }
            batches[batch_id]['total_tasks'] += 1
        
        tasks[task_id] = task_info
        
        # 启动后台线程处理任务
        thread = threading.Thread(target=process_paper, args=(task_id, file_path, content_types, file.filename))
        thread.daemon = True
        thread.start()
        
        return jsonify({'task_id': task_id})
    
    return render_template('index.html')

@app.route('/task/<task_id>')
def get_task_status(task_id):
    """获取任务处理状态"""
    if task_id not in tasks:
        return jsonify({'error': '任务不存在'})
    
    task = tasks[task_id]
    return jsonify({
        'status': task['status'],
        'progress': task.get('progress', ''),
        'result': task.get('result', {}),
        'error': task.get('error', '')
    })

@app.route('/download/<task_id>')
def download_result(task_id):
    """下载单个处理结果"""
    if task_id not in tasks or tasks[task_id]['status'] != 'completed':
        return jsonify({'error': '任务未完成或不存在'})
    
    result_file = tasks[task_id]['result_file']
    original_filename = tasks[task_id]['original_filename']
    base_filename = os.path.splitext(original_filename)[0]
    safe_filename = re.sub(r'[<>:"/\\|?*]', '_', base_filename)
    
    return send_file(result_file, as_attachment=True, download_name=f'{safe_filename}_论文处理结果.txt')

@app.route('/download_batch/<batch_id>')
def download_batch_results(batch_id):
    """批量下载处理结果"""
    if batch_id not in batches:
        return jsonify({'error': '批次不存在'})
    
    batch = batches[batch_id]
    
    # 检查是否有完成的任务
    completed_tasks = [task for task in batch['task_results'] if task['status'] == 'completed']
    if not completed_tasks:
        return jsonify({'error': '该批次中没有可下载的结果'})
    
    # 创建临时目录存储结果文件
    import zipfile
    import tempfile
    
    # 创建临时zip文件
    zip_filename = f'论文处理结果批量_{batch_id[:8]}.zip'
    zip_file_path = os.path.join(app.config['UPLOAD_FOLDER'], zip_filename)
    
    try:
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for task_info in completed_tasks:
                if 'result_file' in task_info and os.path.exists(task_info['result_file']):
                    # 添加文件到zip，使用原始文件名作为存档名
                    zipf.write(task_info['result_file'], os.path.basename(task_info['result_file']))
        
        # 发送zip文件给用户
        return send_file(zip_file_path, as_attachment=True, download_name=zip_filename)
    
    except Exception as e:
        print(f"创建批量下载文件时出错: {str(e)}")
        return jsonify({'error': f'创建批量下载文件时出错: {str(e)}'})
    
    finally:
        # 清理临时文件（可选，取决于您是否想保留这些文件）
        # 注意：如果立即删除，可能会影响文件下载
        # os.remove(zip_file_path)
        pass

if __name__ == '__main__':
    # 创建templates文件夹
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, port=5000)
