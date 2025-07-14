# Development Guide

This guide is for developers who want to contribute to or modify the DBD Auto Skill Check project.

## Project Structure

```
dbd_autoSkillCheck/
├── app.py                 # Main application entry point
├── config.py             # Configuration management
├── setup.py              # Installation and setup script
├── requirements.txt      # Python dependencies
├── run_app.bat          # Windows launcher script
├── run_app.sh           # Unix/Linux launcher script
├── dbd/                 # Core application modules
│   ├── AI_model.py      # AI model handling (ONNX/TensorRT)
│   ├── utils/           # Utility modules
│   │   ├── monitor.py   # Screen capture utilities
│   │   └── directkeys.py # Windows key input handling
│   └── ...
├── models/              # AI model files (.onnx, .trt)
├── images/              # Documentation images
└── logs/                # Application logs
```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git (for development)
- For GPU support: CUDA-compatible GPU with appropriate drivers

### Quick Setup

1. Clone the repository:
```bash
git clone https://github.com/Manuteaa/dbd_autoSkillCheck.git
cd dbd_autoSkillCheck
```

2. Run the setup script:
```bash
python setup.py
```

3. Download model files:
   - Go to [Releases](https://github.com/Manuteaa/dbd_autoSkillCheck/releases)
   - Download `model.onnx` and place it in the `models/` directory

### Manual Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create required directories:
```bash
mkdir -p models logs saved_images
```

## Code Quality

### Type Hints
The codebase uses comprehensive type hints for better IDE support and error detection:

```python
def monitor(ai_model_path: str, device: str, monitor_id: int, 
           hit_ante: float, nb_cpu_threads: int) -> Generator[Tuple[Any, ...], None, None]:
    """Function with full type annotations."""
    pass
```

### Error Handling
All functions include proper error handling with logging:

```python
try:
    result = risky_operation()
    logger.info("Operation successful")
    return result
except SpecificException as e:
    logger.error(f"Specific error occurred: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    return default_value
```

### Logging
The application uses structured logging throughout:

```python
import logging

logger = logging.getLogger(__name__)

logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
```

## Architecture Overview

### AI Model Pipeline

1. **Screen Capture**: MSS library captures screen regions
2. **Preprocessing**: Images are resized and normalized
3. **Inference**: ONNX/TensorRT model processes images
4. **Decision**: Based on predictions, key presses are triggered

### Configuration Management

The `config.py` module provides centralized configuration:

```python
from config import get_config

config = get_config()
threads = config.default_cpu_threads
```

### Resource Management

Proper cleanup is implemented using context managers:

```python
with AI_model(model_path, use_gpu=True) as model:
    # Use model
    pass
# Automatic cleanup on exit
```

## Testing

### Running Tests
```bash
python -m pytest tests/  # When test suite is available
```

### Manual Testing
1. Start the application: `python app.py`
2. Test different configurations
3. Verify error handling with invalid inputs
4. Check resource cleanup

## Performance Optimization

### CPU Usage
- Adjust `nb_cpu_threads` based on system capabilities
- Use lower CPU workload for battery-powered devices

### GPU Acceleration
- Install appropriate ONNX runtime: `onnxruntime-gpu` or `onnxruntime-directml`
- For TensorRT: requires NVIDIA GPU with TensorRT installation

### Memory Management
- Models are loaded once and reused
- Proper cleanup prevents memory leaks
- Screenshots are processed in-place when possible

## Security Considerations

### Anti-Cheat Compatibility
- Use only in private games or training environments
- The application uses Windows SendInput API which may be detected
- Consider virtual machine isolation for testing

### Input Validation
All user inputs are validated:

```python
if not isinstance(monitor_id, int) or monitor_id < 1:
    raise ValueError("Invalid monitor ID")
```

## Contributing

### Code Style
- Use meaningful variable and function names
- Add docstrings to all public functions
- Follow PEP 8 style guidelines
- Include type hints for all function parameters and returns

### Commit Guidelines
- Use clear, descriptive commit messages
- Make small, focused commits
- Include tests for new features when applicable

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Make your changes with proper documentation
4. Test thoroughly
5. Submit a pull request with description

## Common Issues and Solutions

### Import Errors
```bash
# Install missing dependencies
pip install -r requirements.txt
```

### GPU Not Working
```bash
# For NVIDIA GPUs
pip uninstall onnxruntime
pip install onnxruntime-gpu

# For AMD/Intel GPUs
pip install onnxruntime-directml
```

### Model Loading Errors
- Verify model file exists in `models/` directory
- Check file permissions
- Ensure model file is not corrupted

### Screen Capture Issues
- Check monitor IDs in the UI
- Verify screen resolution and scaling
- Run as administrator if needed (Windows)

## Debugging

### Enable Debug Logging
Modify the logging level in `app.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Common Debug Steps
1. Check log files in `logs/` directory
2. Verify model outputs with test images
3. Monitor resource usage during operation
4. Test with different screen configurations

## API Reference

### AI_model Class
```python
class AI_model:
    def __init__(self, model_path: str, use_gpu: bool = False, 
                 nb_cpu_threads: Optional[int] = None, monitor_id: int = 1)
    def predict(self, img_np: np.ndarray) -> Tuple[int, str, Dict[str, float], bool]
    def cleanup(self) -> None
```

### Configuration
```python
class AppConfig:
    models_folder: str = "models"
    default_cpu_threads: int = 4
    target_fps: int = 60
    # ... other configuration options
```

For more details, see the inline documentation in the source code.