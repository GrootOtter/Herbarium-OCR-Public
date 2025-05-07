[Chinese Version](README.md) | [English Version](README.en.md)

# Herbarium-OCR

## Project Overview
Herbarium-OCR is an open-source OCR tool primarily designed for extracting text from **floras, papers, and handwritten or printed labels of herbarium specimens from Central Eurasian countries**, aiming to support research in plant systematics and ecology. It can also process scanned documents and photos from other regions and languages. **Users are advised to first consider commercial OCR solutions for stable service and support**, such as [ABBYY](https://www.abbyy.com/), [Google Document AI](https://cloud.google.com/document-ai), and [TextIn](https://www.textin.com/).

The workflow includes:
1. Optional **auto-rotation** (if enabled) to correct full image/page orientation.
2. Layout analysis using the [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) model to extract text block images.
3. Optional **image enhancement** (contrast, denoising, sharpening) applied to cropped text blocks.
4. Text recognition using a supported OCR engine.

**Supported OCR Engines**:
- **Large Language Models via OpenAI-compatible interface** ([OpenAI SDK](https://github.com/openai/openai-python)), such as Gemini (`gemini-2.0-flash`), Qwen (`qwen-vl-plus`), ChatGLM (`glm-4v-plus`). Theoretically compatible with local [Ollama](https://ollama.com/blog/openai-compatibility) or [vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server).
- **XFYun OCR Services** via HTTP API (OCR technology provided by iFlytek):
  - **General Text Recognition** (`xfyun-general-ocr`): Supports Chinese/English ([API Doc](https://www.xfyun.cn/doc/words/universal_character_recognition/API.html) - Chinese).
  - **Printed Text Recognition (Multilingual)** (`xfyun-printed-ocr`): Supports various languages ([API Doc](https://www.xfyun.cn/doc/words/multi_print_recognition/API.html) - Chinese).  
    (**Note**: XFYun integration is not extensively tested and requires paid quotas).
- **Local OCR Engine**:
  - **Surya OCR** (`surya-ocr`): A Torch-based OCR engine ([GitHub](https://github.com/VikParuchuri/surya)). The current integration requires approximately **7GB of VRAM per process** for layout+OCR. Please check your CUDA device specifications before use.
    (**Note**: Local OCR integration is not extensively tested).

**Image Preprocessing Features** (Configurable):
- **Auto-Rotation**: Corrects image orientation using Tesseract OCR (requires Tesseract). Disabled by default.
- **Image Enhancement**: Applies contrast enhancement, denoising (color or grayscale mode), and sharpening to cropped text blocks. Disabled by default.
- The project provides a separate preprocessing entry point: `herbarium-ocr-preprocess` (or `python -m Main.image_processer` when running from source). This tool can apply the **full preprocessing pipeline** (including optional auto-rotation and all enhancement steps, behavior depends on the configuration file) to a single file or an entire directory of files. **Note**: Preprocessing many or high-resolution full images can be time-consuming. The processed files are saved under the input path for subsequent OCR with the main script.

**Output Formats**: Supports Markdown, JSON, XML, and HTML. By default, only a `full.json` file containing all details is generated. Other formats can be requested via the `--output_format` argument.

**Batch Processing**: `pdf_batch` and `image_batch` modes support parallel processing using multiple processes. The number of worker processes is configurable (default is 1).

## Development Status and Maintenance
This project was developed by the author during graduate studies. Due to time constraints and research commitments, **future maintenance and feature development will primarily rely on community collaboration**. Users are encouraged to:
- Submit bug reports or detailed feature requests in the **Issues** section on [Gitee](https://gitee.com/esenzhou/Herbarium-OCR-Public) or [GitHub](https://github.com/GrootOtter/Herbarium-OCR-Public).
- Contribute code fixes or new features via **Pull Requests**.

## System Requirements
- **Python**: 3.10 or higher
- **Git**: For source installation
- **Hardware**: CUDA-enabled GPU (Optional, accelerates layout analysis and local OCR)
- **Dependencies** (See `requirements.txt`):
  - `toml`: For parsing configuration files
  - Core libraries: PyTorch, OpenCV, Pillow, PyMuPDF, openai, doclayout_yolo, tqdm
- **Optional Dependencies**:
  - **Tesseract OCR engine** and `pytesseract`: For auto-rotation feature
  - `surya-ocr`: For local OCR (CUDA device with >8GB VRAM strongly recommended)
  - `requests`: For XFYun OCR services

## Installation

### Installation from PyPI (Recommended)
Install Herbarium-OCR via PyPI:
```bash
pip install herbarium-ocr
```

To install with all optional features (Tesseract support, XFYun client, Surya client):
```bash
pip install "herbarium-ocr[full]"
```

**Note**: If enabling auto-rotation, you must separately install the [Tesseract OCR engine](https://github.com/UB-Mannheim/tesseract/wiki) for your operating system.

**GPU Support**: For accelerated processing, install a CUDA-enabled version of PyTorch from the [PyTorch website](https://pytorch.org/).

### Installation from Source 
Clone the repository if you need to contribute or use the latest development version:

From Gitee:
```bash
git clone https://gitee.com/esenzhou/Herbarium-OCR-Public.git
cd Herbarium-OCR-Public
```
From GitHub:
```bash
git clone https://github.com/GrootOtter/Herbarium-OCR-Public.git
cd Herbarium-OCR-Public
```

Install dependencies (ideally in a virtual environment):
```bash
pip install -r requirements.txt
```

Install optional dependencies (e.g., To enable auto-rotation):
```bash
pip install pytesseract
# Also install the Tesseract OCR engine itself (see below)
```

**Install Tesseract OCR Engine** (Only if enabling auto-rotation):
- **Linux**: Use package manager, e.g., `sudo apt install tesseract-ocr` (Debian/Ubuntu).
- **Windows**: Download and install from [Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki). Ensure the executable path is added to the system PATH environment variable.

**GPU Support**: Install a CUDA-enabled version of PyTorch from the [PyTorch website](https://pytorch.org/).

## Usage

### Running from PyPI Installation 
Use the following command-line tools after installation:

#### Main Processing: `herbarium-ocr`
Process PDF or image files for OCR.
```bash
herbarium-ocr --mode <mode> --input <input_path> --model <model_name> [options]
```
- **Modes**: `pdf`, `pdf_batch`, `image`, `image_batch`
- **Options**:
  - `--languages`: Comma-separated language codes (e.g., `hy,ru`)
  - `--output_format`: `markdown`, `json`, `xml`, `html` (generates this in addition to `full.json`)
  - `--preprocess_images`: Enable image block enhancements
  - `-v, --verbose`: Enable debug logging
  - `-c, --config`: Path to custom TOML config file

**Example**:
```bash
herbarium-ocr --mode pdf --input document.pdf --model gemini --output_format html
```

#### Convert Output: `herbarium-ocr-convert`
Convert an existing `full.json` output file to other formats.
```bash
herbarium-ocr-convert <input_path_full.json> --to <format>... [-v]
```
- **Formats**: `markdown`, `md`, `html`, `htm`, `xml`, `json` (filtered version)

**Example**:
```bash
herbarium-ocr-convert output_full.json --to markdown html
```

#### Test Preprocessing: `herbarium-ocr-preprocess`
Test the preprocessing pipeline (rotation attempt, enhancements).
```bash
herbarium-ocr-preprocess --input <input_path> [-c <config_path>] [-v]
```

**Example**:
```bash
herbarium-ocr-preprocess --input image.jpg
```

#### Check Layout Model: `herbarium-ocr-check-layout`
Display supported layout classes from the built-in model.
```bash
herbarium-ocr-check-layout [-c <config_path>] [-v]
```

**Example**:
```bash
herbarium-ocr-check-layout -c my_config.toml -v
```

### Running from Source 
If you cloned the repository, run scripts from the project root using `python -m`:

#### Main Processing
```bash
python -m Main.herbarium_ocr --mode <mode> --input <input_path> --model <model_name> [options]
```

**Example**:
```bash
python -m Main.herbarium_ocr --mode pdf --input document.pdf --model gemini --output_format html
```

#### Convert Output
```bash
python -m Main.convert <input_path_full.json> --to <format>... [-v]
```

**Example**:
```bash
python -m Main.convert output_full.json --to markdown
```

#### Test Preprocessing
```bash
python -m Main.image_processer --input <input_path> [-c <config_path>] [-v]
```

**Example**:
```bash
python -m Main.image_processer --input image.jpg
```

#### Check Layout Model
```bash
python -m Main.check_layout_model [-c <config_path>] [-v]
```

**Example**:
```bash
python -m Main.check_layout_model -c my_config.toml -v
```

## Configuration
Customize via `herbarium_ocr_config.toml`. Search order: `-c` path > User dir > Defaults. Only include settings you want to override.

**Example Config** (`herbarium_ocr_config.toml`):
```toml
[OCR_CONFIG]
languages = "en,ru"         # Default language hints
output_format = "html"      # Default conversion format
preprocess_images = true    # Enable block enhancement
enhance_contrast = true
denoise = false             # Disable slow denoising
sharpen = true
attempt_auto_rotation = true # Enable Tesseract rotation
# tesseract_cmd_path = "/usr/local/bin/tesseract" # Tesseract path (Example)
min_rotation_confidence = 50
max_workers = 0             # Use all CPU cores for batch

[DOCLAYOUT_CONFIG]
RELEVANT_TEXT_CLASSES = ["title", "plain text"]
DOCLAYOUT_CONF_THRESHOLD = 0.25

[MODEL_CONFIGS]
# Add a new model definition (Example using OpenRouter)
  [MODEL_CONFIGS.openrouter]                # Name used with the --model argument (e.g., --model openrouter)
  type = "openai_compatible"                # Specifies which client handles this (OpenAI compatible)
  language_mode = "list_hint"               # How the client uses the --languages arg (accepts list as hint)
  api_key_env = "OPENROUTER_API_KEY"        # Environment variable name holding the API key
  base_url = "https://openrouter.ai/api/v1" # Base URL for the API endpoint (provider: OpenRouter)
  model_id = "google/gemma-3-27b-it:free"   # Specific model identifier (get from provider's documentation)
  rpm_limit = 20                            # Requests Per Minute limit (check provider's documentation/limits)

  # Add local Ollama model (Example, untested)
  [MODEL_CONFIGS.ollama_llava]
  type = "openai_compatible"
  language_mode = "list_hint"
  api_key_env = "OLLAMA_API_KEY" # Can be dummy value like "ollama"
  base_url = "http://localhost:11434/v1"
  model_id = "gemma3:27b" # Your loaded model name
  rpm_limit = 10
  max_dimension = 0 # Disable client image processing

  # Modify existing gemini config
  [MODEL_CONFIGS.gemini]
  model_id = "gemini-2.0-flash-lite"
  rpm_limit = 30

  # Modify XFyun printed OCR params
  [MODEL_CONFIGS.xfyun-printed-ocr]
  param_value = "ru" # Default language Russian
  max_dimension = 2000
  jpeg_quality = 90
```

**Note**: Run `herbarium-ocr-check-layout` or `python -m Main.check_layout_model` to see supported `RELEVANT_TEXT_CLASSES`.

**API Key/Credential Setup (Environment Variables)**:
* **OpenAI-Compatible Models**:

* Obtain the corresponding API keys from the respective LLM provider’s website. Before using this project, you can send a text image to test if the model supports it. This project accesses the following models (`--model` parameter):

  - [gemini](https://ai.google.dev/gemini-api/docs) `gemini-2.0-flash`
  - [grok](https://docs.x.ai/docs/overview) `grok-2-vision-1212`
  - [qwen](https://help.aliyun.com/zh/model-studio/) `qwen-vl-plus`
  - [glm-4](https://bigmodel.cn/dev/welcome) `glm-4v-plus-0111`
  - [yi](https://platform.lingyiwanwu.com/docs) `yi-vision-v2`
  - [kimi](https://platform.moonshot.cn/docs/intro) `moonshot-v1-8k-vision-preview`
  - [doubao](https://www.volcengine.com/docs/82379/1399008) `doubao-1.5-vision-pro-250328`
  - Other LLMs supporting the OpenAI interface (configure the API endpoint in `herbarium_ocr_config.toml`)

* Set environment variables:
  #### Linux

  **Temporary Setup** (current session only):

  ```bash
  export GOOGLE_API_KEY="your-google-api-key"          # For Gemini
  export XAI_API_KEY="your-xai-api-key"                # For Grok
  export DASHSCOPE_API_KEY="your-dashscope-api-key"    # For Qwen
  export ZHIPUAI_API_KEY="your-zhipuai-api-key"        # For GLM-4 
  export YI_API_KEY="your-yi-api-key"                  # For Yi
  ...
  ```

  **Permanent Setup** :

  Add the above `export` commands to your shell configuration file (e.g., `~/.bashrc`, `~/.zshrc`):

  ```bash
  echo 'export GOOGLE_API_KEY="your-google-api-key"' >> ~/.bashrc
  ```

  Reload the shell configuration:

  ```bash
  source ~/.bashrc  # or source ~/.zshrc
  ```

  #### Windows

  **Temporary Setup** (current session only):

  Open PowerShell and run:

  ```powershell
  $env:GOOGLE_API_KEY = "your-google-api-key"
  ```

  **Permanent Setup** :

  ```powershell
  [System.Environment]::SetEnvironmentVariable("GOOGLE_API_KEY", "your-google-api-key", "User")
  ```

  Alternatively, set environment variables via the GUI:

  1. Search for "Environment Variables" in the Windows Start menu.
  2. Select "Edit the system environment variables" or "Edit environment variables for your account."
  3. Under "User variables," add a new variable with the name (e.g., `GOOGLE_API_KEY`) and value (e.g., `your-google-api-key`).


* **XFYun OCR API** (`--model` parameter: `xfyun-general-ocr`, `xfyun-printed-ocr`):  
  * Requires setting three environment variables: `SPARK_APPID`, `SPARK_API_KEY`, `SPARK_API_SECRET`.  
  * Obtain these values from your application in the [XFYun Open Platform Console](https://console.xfyun.cn/app/myapp).  
  * Set the environment variables as described above.

## Troubleshooting
- **Layout Detection Failures**: Modify `RELEVANT_TEXT_CLASSES` and `DOCLAYOUT_CONF_THRESHOLD` in config. Enabling `herbarium-ocr-preprocess` might improve detection confidence.
- **API Key Errors**: Use `-v` to verify environment variables are correctly set and checked.
- **XFYun 403 Forbidden**: Check API credentials and ensure your system clock is accurate (within 5 minutes of UTC).
- **Tesseract Errors**: Ensure Tesseract engine and `pytesseract` library are installed and configured correctly (PATH or `tesseract_cmd_path`).

## Contributing
Contributions are welcome! Please use:
- **Issues**: Report bugs or suggest features on [Gitee](https://gitee.com/esenzhou/Herbarium-OCR-Public) or [GitHub](https://github.com/GrootOtter/Herbarium-OCR-Public).
- **Pull Requests**: Submit code fixes or new features.
Future development relies significantly on community involvement.

## License
This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENSE](https://www.gnu.org/licenses/agpl-3.0.html) file. The included `doclayout_yolo_docstructbench_imgsz1024.pt` model file is also under AGPL-3.0.

## Acknowledgments
Thanks to the developers of key open-source projects and libraries such as DocLayout-YOLO, PyMuPDF, Pillow, OpenAI Python SDK, Requests, Tesseract OCR, and PyTorch. Special thanks to [Gemini](https://aistudio.google.com/app/prompts/new_chat) and [Grok](https://grok.com/) for their code instructions. Also, thanks to the Herbarium of Xinjiang Institute of Ecology and Geography, CAS (XJBI) for supporting this work.

## See Other Excellent OCR Projects
*   [Zerox](https://github.com/getomni-ai/zerox)
*   [Marker](https://github.com/VikParuchuri/marker)
*   [MinerU](https://github.com/opendatalab/MinerU)
*   [GPTPDF](https://github.com/CosmosShadow/gptpdf)