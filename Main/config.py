# Main/config.py
# Default configuration values for Herbarium-OCR

# --- Default Constants for Image Processing ---
# Used if not overridden in specific model configs or OCR_CONFIG
DEFAULT_RESIZE_MAX_DIMENSION = 1500 # Default max dimension (longest side) for resizing
DEFAULT_JPEG_QUALITY = 90           # Default JPEG quality (1-100)

# --- DocLayout-YOLO Configuration ---
DOCLAYOUT_CONFIG = {
    # Path to the layout model file (.pt). Relative paths resolved from Main package dir.
    "DOCLAYOUT_MODEL_PATH": "./layout_models/doclayout_yolo_docstructbench_imgsz1024.pt",
    # Target image size for layout model inference.
    "DOCLAYOUT_IMG_SIZE": 1024,
    # Minimum confidence score (0-1) for layout block detection.
    "DOCLAYOUT_CONF_THRESHOLD": 0.3,
    # List of layout block class names to extract text from.
    # Run `python -m Main.check_layout_model` to see available classes.
    "RELEVANT_TEXT_CLASSES": [
        'title', 
        'plain text', 
        'figure', 
        'figure_caption',
        'table', 
        'table_caption', 
        'table_footnote',
        'isolate_formula', 
        'formula_caption'
        ]
}

# --- OCR, Preprocessing, Output Configuration ---
OCR_CONFIG = {
    # Default language hint(s) passed to OCR clients (comma-separated string or list).
    "languages": "",

    # --- Image Preprocessing ---
    # Master switch for applying enhancement steps to *cropped* blocks.
    "preprocess_images": False, # Overridden by --preprocess_images flag
    # Specific enhancement steps (only active if preprocess_images is True):
    "enhance_contrast": True,
    "denoise": True,
    "sharpen": True,
    # Denoising mode ('gray' is faster, 'color' might be better but slower).
    "denoise_mode": "gray",

    # --- Auto-Rotation Settings ---
    # Attempt to auto-rotate the *full* input image/page using Tesseract.
    "attempt_auto_rotation": False, # Config file only. Requires Tesseract if True.
    # DPI for rendering PDF/DjVu pages to images.
    "dpi": 300,
    # Full path to Tesseract executable if not in system PATH.
    "tesseract_cmd_path": None,
    # Minimum confidence (0-100) from Tesseract OSD to apply rotation.
    "min_rotation_confidence": 60,

    # --- Output ---
    # Name of subdirectory created within input dir for batch mode output.
    "batch_output_subdir_name": "herbariumOCR_output",

    # --- Default Client Image Processing Parameters ---
    # These are fallback values if not specified per-model.
    # Standardized name for max dimension (longest side) for resizing.
    "max_dimension": DEFAULT_RESIZE_MAX_DIMENSION,
    # Standardized name for JPEG quality.
    "jpeg_quality": DEFAULT_JPEG_QUALITY,
    # REMOVED: "resize_max_pixels" # Old inconsistent key

    # --- Batch Processing Performance ---
    # Default number of parallel worker processes (0 = use os.cpu_count()).
    "max_workers": 1
}

# --- Model-Specific Configurations ---
# Defines supported OCR models/APIs and their parameters.
MODEL_CONFIGS = {
    # --- OpenAI Compatible Models ---
    "gemini": {
        "type": "openai_compatible",        # Client type
        "language_mode": "list_hint",       # Accepts language list as hint
        "api_key_env": "GOOGLE_API_KEY",    # Env var for API key
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", # API endpoint
        "model_id": "gemini-2.0-flash",   # Model name for API
        "rpm_limit": 15,                    # Rate limit hint
    },
    "qwen": {
        "type": "openai_compatible",
        "language_mode": "list_hint",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_id": "qwen-vl-plus",
        "rpm_limit": 1200
    },
    "glm-4": {
        "type": "openai_compatible",
        "language_mode": "list_hint",
        "api_key_env": "ZHIPUAI_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model_id": "glm-4v-plus-0111",
        "rpm_limit": 300
    },
    "kimi": {
        "type": "openai_compatible",
        "language_mode": "list_hint",
        "api_key_env": "MOONSHOT_API_KEY",
        "base_url": "https://api.moonshot.cn/v1",
        "model_id": "moonshot-v1-8k-vision-preview",
        "rpm_limit": 3
    },
    "doubao": {
        "type": "openai_compatible",
        "language_mode": "list_hint",
        "api_key_env": "ARK_API_KEY",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model_id": "doubao-1.5-vision-pro-250328",
        "rpm_limit": 1200
    },
    "yi": {
        "type": "openai_compatible",
        "language_mode": "list_hint",
        "api_key_env": "YI_API_KEY",
        "base_url": "https://api.lingyiwanwu.com/v1",
        "model_id": "yi-vision-v2", # Verify model name
        "rpm_limit": 4
    },
    # --- UNTESTED OpenAI Compatible Models ---
    "grok": {
        "type": "openai_compatible",
        "language_mode": "list_hint",
        "api_key_env": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "model_id": "grok-2-vision-1212", # Verify model name
        "rpm_limit": 180
    },

    # --- XFYun HTTP OCR Models ---
    "xfyun-general-ocr": {
        "type": "xfyun_http_ocr",
        "language_mode": "ignore", # API is CN/EN only
        "app_id_env": "SPARK_APPID",
        "api_key_env": "SPARK_API_KEY",
        "api_secret_env": "SPARK_API_SECRET",
        "api_url": "https://api.xf-yun.com/v1/private/sf8e6aca1",
        "service_key": "sf8e6aca1",
        "param_type": "category",
        "param_value": "ch_en_public_cloud",
        "rpm_limit": 60,
        "max_dimension": DEFAULT_RESIZE_MAX_DIMENSION, # Use default constant
        "jpeg_quality": DEFAULT_JPEG_QUALITY      # Use default constant
    },
    "xfyun-printed-ocr": {
        "type": "xfyun_http_ocr",
        "language_mode": "single_required", # Requires one language code
        "app_id_env": "SPARK_APPID",
        "api_key_env": "SPARK_API_KEY",
        "api_secret_env": "SPARK_API_SECRET",
        "api_url": "https://cn-east-1.api.xf-yun.com/v1/ocr",
        "service_key": "ocr",
        "param_type": "language",
        "param_value": "ch_en", # Default language if none provided
        "rpm_limit": 60,
        "max_dimension": DEFAULT_RESIZE_MAX_DIMENSION,
        "jpeg_quality": DEFAULT_JPEG_QUALITY
    },

    # --- Local Surya OCR Model ---
    "surya-ocr": {
        "type": "surya_ocr",
    },
    
}