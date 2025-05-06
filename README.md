[English Version](README.en.md) | [中文版本](README.md)

# Herbarium-OCR

## 项目概览
Herbarium-OCR 是一个开源 OCR 工具，主要用于提取**中央欧亚各国植物志、论文和标本手写或打印标签文本**，旨在支持植物系统学和生态学研究。它也可处理其他地区和语言的扫描文档和照片。**建议用户优先尝试商业 OCR 解决方案以获得稳定支持**，如 [ABBYY](https://www.abbyy.com/)，[google document ai](https://cloud.google.com/document-ai) 和 [TextIn](https://www.textin.com/)。

工作流程包括：
1. 可选的**自动旋转**（如启用），校正完整图像/页面方向。
2. 使用 [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) 模型进行版面分析，提取文本块图像。
3. 对裁剪的文本块应用可选的**图像增强**（对比度、降噪、锐化）。
4. 使用支持的 OCR 引擎进行文本识别。

**支持的 OCR 引擎**：
- **通过 OpenAI 兼容接口的大型语言模型** ([OpenAI SDK](https://github.com/openai/openai-python))，如 Gemini (`gemini-2.0-flash`)、Qwen (`qwen-vl-plus`)、ChatGLM (`glm-4v-plus`)。理论上支持本地部署的 [Ollama](https://ollama.com/blog/openai-compatibility) 或 [vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server)。
- **讯飞 OCR 服务** 通过 HTTP API（OCR技术由科大讯飞提供）：
  - **通用文字识别** (`xfyun-general-ocr`)：支持中英文 ([API 文档](https://www.xfyun.cn/doc/words/universal_character_recognition/API.html))。
  - **多语种印刷文字识别** (`xfyun-printed-ocr`)：支持多种语言 ([API 文档](https://www.xfyun.cn/doc/words/multi_print_recognition/API.html))。  
    **注意**：讯飞接口未充分测试。
- **本地 OCR 引擎**：
    *   **Surya OCR** (`surya-ocr`)：基于Torch的OCR引擎([GitHub](https://github.com/VikParuchuri/surya))。当前集成方式下版面识别+文字识别的**单个进程**约占用 **7GB** VRAM，使用前请确认您的 **cuda 设备信息**。
    **注意**：本地OCR接口未充分测试。

**图像预处理功能**（可配置）：
- **自动旋转**：使用 Tesseract OCR 校正图像方向（需 Tesseract）。默认禁用。
- **图像增强**：对裁剪文本块应用对比度增强、降噪（彩色或灰度模式）和锐化。默认禁用。

**输出格式**：支持 Markdown、JSON、XML 和 HTML。默认仅生成 `full_json`，其他格式通过 `--output_format` 指定。

**批处理**：`pdf_batch` 和 `image_batch` 模式支持多进程并行处理，进程数可配置（默认为1）。

## 开发状态与维护
本项目由作者在研究生学习期间开发。由于时间和科研任务限制，**后续维护和功能开发主要依赖社区协作**。欢迎用户：
- 在 [Gitee](https://gitee.com/esenzhou/Herbarium-OCR-Public) 或 [GitHub](https://github.com/GrootOtter/Herbarium-OCR-Public) 的 **Issues** 区提交问题或建议。
- 通过 **Pull Requests** 贡献代码。

## 系统要求
- **Python**：3.10 或更高
- **Git**：用于源码安装
- **硬件**：支持 CUDA 的 GPU（可选，加速版面分析和本地OCR运行）
- **依赖库**（见 `requirements.txt`）：
  - `toml`：解析配置文件
  - 核心库：PyTorch、OpenCV、Pillow、PyMuPDF、openai、doclayout_yolo、tqdm
- **可选依赖**：
  - **Tesseract OCR 引擎** 和 `pytesseract`：用于自动旋转
  - `surya-ocr`： 用于本地 OCR ，强烈建议使用cuda设备加速 (VRAM>8GB)
  - `requests`：用于讯飞 OCR 服务

## 安装

### 从 PyPI 安装（推荐）
通过 PyPI 安装 Herbarium-OCR：
```bash
pip install herbarium-ocr
```

启用完整功能（包括自动旋转和其他 OCR 客户端）：
```bash
pip install herbarium-ocr[full]
```

**注意**：若启用自动旋转，需单独为您的操作系统安装 [Tesseract OCR 引擎](https://github.com/UB-Mannheim/tesseract/wiki)。

**GPU 支持**：为加速处理，从 [PyTorch 官网](https://pytorch.org/) 安装支持 CUDA 的 PyTorch。

### 从源码安装
若需贡献代码或使用最新开发版本，可克隆仓库：
从 Gitee：
```bash
git clone https://gitee.com/esenzhou/Herbarium-OCR-Public.git
cd Herbarium-OCR-Public
```
从 GitHub：
```bash
git clone https://github.com/GrootOtter/Herbarium-OCR-Public.git
cd Herbarium-OCR-Public
```

安装依赖：
```bash
pip install -r requirements.txt
```

启用自动旋转：
```bash
pip install pytesseract
```

**安装 Tesseract OCR 引擎** ：
- **Linux**：大多数发行版可通过包管理器安装，如 `sudo apt install tesseract-ocr` (Debian/Ubuntu) 
- **Windows**：下载并安装官方安装包：[Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki)。安装后确保将 Tesseract 可执行文件路径添加到系统 PATH 环境变量中。

**GPU 支持**：从 [PyTorch 官网](https://pytorch.org/) 安装支持 CUDA 的 PyTorch。

## 使用方法

### 从 PyPI 安装运行
从 PyPI 安装后，使用以下命令行工具：

#### 主处理：`herbarium-ocr`
处理 PDF 或图像的 OCR。
```bash
herbarium-ocr --mode <mode> --input <input_path> --model <model_name> [options]
```
- **模式**：`pdf`、`pdf_batch`、`image`、`image_batch`
- **选项**：
  - `--languages`：逗号分隔的语言代码（如 `hy,ru`）
  - `--output_format`：`markdown`、`json`、`xml`、`html`
  - `--preprocess`：启用图像块画质增强
  - `-v, --verbose`：启用调试日志
  - `-c, --config`：自定义 TOML 配置文件路径

**示例**：
```bash
herbarium-ocr --mode pdf --input document.pdf --model gemini --output_format html
```

#### 转换输出：`herbarium-ocr-convert`
将 `full.json` 输出转换为其他格式。
```bash
herbarium-ocr-convert <input_path> --to <format> [-v]
```
- **格式**：`markdown`、`md`、`html`、`htm`、`xml`、`json`

**示例**：
```bash
herbarium-ocr-convert output_full.json --to markdown
```

#### 测试预处理：`herbarium-ocr-preprocess`
测试预处理管道（旋转、增强）。
```bash
herbarium-ocr-preprocess --input <input_path>
```

**示例**：
```bash
herbarium-ocr-preprocess --input image.jpg
```

#### 检查布局模型：`herbarium-ocr-check-layout`
显示支持的布局类别。
```bash
herbarium-ocr-check-layout --input <input_path> [-c <config_path>] [-v]
```

**示例**：
```bash
herbarium-ocr-check-layout --input image.jpg -c my_config.toml -v
```

### 从源码运行
若克隆了仓库，请在项目根目录使用 `python -m` 运行脚本：

#### 主处理
```bash
python -m Main.herbarium_ocr --mode <mode> --input <input_path> --model <model_name> [options]
```

**示例**：
```bash
python -m Main.herbarium_ocr --mode pdf --input document.pdf --model gemini --output_format html
```

#### 转换输出
```bash
python -m Main.convert <input_path> --to <format> [-v]
```

**示例**：
```bash
python -m Main.convert output_full.json --to markdown
```

#### 测试预处理
```bash
python -m Main.image_processer --input <input_path>
```

**示例**：
```bash
python -m Main.image_processer --input image.jpg
```

#### 检查布局模型
```bash
python -m Main.check_layout_model [-c <config_path>] [-v]
```

**示例**：
```bash
python -m Main.check_layout_model -c my_config.toml -v
```

## 配置
通过 `herbarium_ocr_config.toml` 自定义配置，按以下顺序加载：
1. 命令行：`-c <path>` 或 `--config <path>`
2. 用户目录：
   - Linux/macOS：`~/.config/herbarium-ocr/herbarium_ocr_config.toml`
   - Windows：`%APPDATA%\HerbariumOCR\herbarium_ocr_config.toml`
3. 默认：`Main/config.py`

仅需包含要覆盖的设置。

**示例配置** (`herbarium_ocr_config.toml`)：
```toml
[OCR_CONFIG]
languages = "hy,ru"                      # 亚美尼亚语和俄语

# --- 预处理 ---
preprocess_images = true                 # 是否启用裁剪块的图像增强 (总开关)
enhance_contrast = true
denoise = true
sharpen = true
denoise_mode = "color"
attempt_auto_rotation = true
# tesseract_cmd_path = " "               # Tesseract 路径 (如果不在系统路径中)
min_rotation_confidence = 60             # 旋转置信度(0-100)
dpi = 300                                # PDF 渲染 DPI

# --- 输出 ---
batch_output_subdir_name = "OCR_Output_Batch" # 批处理输出子目录名

# --- 性能 ---
max_workers = 2
max_dimension = 1500
jpeg_quality = 90

[DOCLAYOUT_CONFIG]
RELEVANT_TEXT_CLASSES = ["title", "plain text"]
DOCLAYOUT_CONF_THRESHOLD = 0.3

[MODEL_CONFIGS]
# 添加新模型
[MODEL_CONFIGS.openrouter]                # 命令行调用时的参数名(e.g., --model openrouter)
type = "openai_compatible"                # 添加openai兼容模型
language_mode = "list_hint"               # Language mode for OpenAI models
api_key_env = "OPENROUTER_API_KEY"        # 定义环境变量名
base_url = "https://openrouter.ai/api/v1" # 供应商 OpenRouter 的 API 端口
model_id = "google/gemma-3-27b-it:free"   # 模型参数名，从模型供应商处查看
rpm_limit = 20                            # 限流，从模型供应商处查看

# 添加新模型 (例如本地 Ollama 仅示例，未测试)
[MODEL_CONFIGS.ollama]
type = "openai_compatible"
language_mode = "list_hint"
api_key_env = "OLLAMA_API_KEY"
base_url = "http://localhost:11434/v1"
model_id = "gemma3:27b"
rpm_limit = 10

# 修改调用的 gemini 模型和对应的 RPM 限制
[MODEL_CONFIGS.gemini]
model_id = "gemini-2.0-flash-lite"
rpm_limit = 30

# 修改讯飞印刷文字识别参数
[MODEL_CONFIGS.xfyun-printed-ocr]
param_value = "ka"                         # 格鲁吉亚语
max_dimension = 2000
jpeg_quality = 90
```

**注意**：运行 `herbarium-ocr-check-layout` 或 `python -m Main.check_layout_model` 查看支持的 `RELEVANT_TEXT_CLASSES`。

**API 密钥/凭证设置 (环境变量)**：
根据你希望使用的模型 (`--model` 参数)，设置相应的环境变量。
*   **OpenAI 兼容模型:**

    * 从相应的 LLM 提供商处获取 API 密钥。本项目接入以下模型：

    - [gemini](https://ai.google.dev/gemini-api/docs)，默认配置`gemini-2.0-flash`
    - [grok](https://docs.x.ai/docs/overview)，默认配置`grok-2-vision-1212`
    - [qwen](https://help.aliyun.com/zh/model-studio/)，默认配置`qwen-vl-plus`
    - [glm-4](https://bigmodel.cn/dev/welcome)，默认配置`glm-4v-plus-0111`
    - [yi](https://platform.lingyiwanwu.com/docs)，默认配置`yi-vision-v2`
    - [kimi](https://platform.moonshot.cn/docs/intro)，默认配置`moonshot-v1-8k-vision-preview`
    - [doubao](https://www.volcengine.com/docs/82379/1399008)，默认配置`doubao-1.5-vision-pro-250328`
    - 其他支持 openai 接口的 LLM (需要在 `herbarium_ocr_config.toml` 中配置)

    * 将相应的 API 密钥设置为环境变量。
    #### Linux/macOS

    **临时设置**（仅当前会话有效）：

    ```bash
    export GOOGLE_API_KEY="your-google-api-key"          # For Gemini
    export XAI_API_KEY="your-xai-api-key"                # For Grok
    export DASHSCOPE_API_KEY="your-dashscope-api-key"    # For Qwen
    export ZHIPUAI_API_KEY="your-zhipuai-api-key"        # For GLM-4
    export YI_API_KEY="your-yi-api-key"                  # For Yi
    ...
    ```

    **永久设置**：

    将上述 `export` 命令添加到您的 shell 配置文件（例如， `~/.bashrc`, `~/.zshrc`）：

    ```bash
    echo 'export GOOGLE_API_KEY="your-google-api-key"' >> ~/.bashrc
    ```

    重新加载 shell 配置：

    ```bash
    source ~/.bashrc  # or source ~/.zshrc
    ```

    #### Windows

    **临时设置**（仅当前会话有效）：

    打开 PowerShell 并运行：

    ```powershell
    $env:GOOGLE_API_KEY = "your-google-api-key"
    ```

    **永久设置**：

    ```powershell
    [System.Environment]::SetEnvironmentVariable("GOOGLE_API_KEY", "your-google-api-key", "User")
    ```

    或者，通过 GUI 设置环境变量：

    1. 在 Windows 开始菜单中搜索“环境变量”。
    2. 选择“编辑系统环境变量”或“编辑账户的环境变量”。
    3. 在“用户变量”下添加新变量，名称（例如 `GOOGLE_API_KEY`）和值（例如 `your-google-api-key`）。


*   **讯飞 OCR API** (`--model`参数：`xfyun-general-ocr`, `xfyun-printed-ocr`):
    *   需要设置三个环境变量：`SPARK_APPID`, `SPARK_API_KEY`, `SPARK_API_SECRET`。
    *   从[讯飞开放平台控制台的应用](https://console.xfyun.cn/app/myapp)中获取这三个值。
    *   按上述方法设置环境变量。

## 故障排除
- **版面识别没有返回结果**：修改 `DOCLAYOUT_CONFIG` 部分的识别对象和置信度阈值。可以运行预处理 `herbarium-ocr-preprocess` 提高识别结果的置信度。
- **API 密钥错误**：确认环境变量已正确设置。用 `-v` 查看检查日志。
- **讯飞 403 Forbidden**：检查 API 凭证和系统时钟准确性（与 UTC 误差 5 分钟内）。
- **Tesseract 错误**：确保 Tesseract 已安装并在 PATH 中，或在 TOML 中配置 `tesseract_cmd_path`。

## 贡献
欢迎通过以下方式贡献：
- **Issues**：在 [Gitee](https://gitee.com/esenzhou/Herbarium-OCR-Public) 或 [GitHub](https://github.com/GrootOtter/Herbarium-OCR-Public) 报告问题或建议。
- **Pull Requests**：提交代码修复或新功能。

## 许可信息
本项目基于 GNU Affero General Public License v3.0 (AGPL-3.0) 许可证发布。详细信息请查看 ([LICENSE](https://www.gnu.org/licenses/agpl-3.0.html)) 文件。本项目包含的 `doclayout_yolo_docstructbench_imgsz1024.pt` 模型文件同样基于 AGPL-3.0 许可证。

## 致谢
感谢 DocLayout-YOLO、PyMuPDF、Pillow、OpenAI SDK、Requests、Tesseract OCR 和 PyTorch 的开发者。特别感谢 [Gemini](https://aistudio.google.com/app/prompts/new_chat) 和 [Grok](https://grok.com/) 的代码指导。也感谢中国科学院新疆生态与地理研究所标本馆 (XJBI) 对本项目开发的支持。

## 查看其他优秀的 OCR 项目
- [Zerox](https://github.com/getomni-ai/zerox)
- [Marker](https://github.com/VikParuchuri/marker)
- [MinerU](https://github.com/opendatalab/MinerU)
- [GPTPDF](https://github.com/CosmosShadow/gptpdf)