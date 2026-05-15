@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found.
  echo Run install.bat first.
  exit /b 1
)

set "PYTHON=.venv\Scripts\python.exe"
set "ONNX_PATH=data\model\models\model.onnx"
set "TRT_FP32_PATH=data\model\models\model-fp32.trt"
set "TRT_FP16_PATH=data\model\models\model-fp16.trt"
set "TRT_INT8_PATH=data\model\models\model-int8.trt"

if exist "%ONNX_PATH%" (
  echo [SKIP] ONNX model already exists: %ONNX_PATH%
) else (
  echo [STEP] Exporting ONNX model: %ONNX_PATH%
  "%PYTHON%" -m scripts.convert_to_onnx --output "%ONNX_PATH%"
  if errorlevel 1 exit /b %errorlevel%
)

if exist "%TRT_FP32_PATH%" (
  echo [SKIP] TensorRT FP32 engine already exists: %TRT_FP32_PATH%
) else (
  echo [STEP] Building TensorRT FP32 engine: %TRT_FP32_PATH%
  "%PYTHON%" -m scripts.convert_to_tensorrt --onnx "%ONNX_PATH%" --output "%TRT_FP32_PATH%"
  if errorlevel 1 exit /b %errorlevel%
)

if exist "%TRT_FP16_PATH%" (
  echo [SKIP] TensorRT FP16 engine already exists: %TRT_FP16_PATH%
) else (
  echo [STEP] Building TensorRT FP16 engine: %TRT_FP16_PATH%
  "%PYTHON%" -m scripts.convert_to_tensorrt --onnx "%ONNX_PATH%" --output "%TRT_FP16_PATH%" --fp16
  if errorlevel 1 exit /b %errorlevel%
)

if exist "%TRT_INT8_PATH%" (
  echo [SKIP] TensorRT INT8 engine already exists: %TRT_INT8_PATH%
) else (
  echo [STEP] Building TensorRT INT8 engine: %TRT_INT8_PATH%
  "%PYTHON%" -m scripts.convert_to_tensorrt --onnx "%ONNX_PATH%" --output "%TRT_INT8_PATH%" --int8
  if errorlevel 1 exit /b %errorlevel%
)

echo [DONE] Conversion pipeline finished.
exit /b 0
