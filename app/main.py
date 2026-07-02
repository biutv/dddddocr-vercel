import uvicorn
import logging
import requests
import os
import tempfile
import sys
from fastapi import Request
from typing import Dict, Any, Optional, Union
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import base64
import time
import json

# Vercel 环境检测和适配
IS_VERCEL = os.environ.get('VERCEL', '0') == '1'

# 在 Vercel 环境中设置临时目录
if IS_VERCEL:
    temp_dir = tempfile.gettempdir()
    os.environ['TMPDIR'] = temp_dir
    os.environ['TEMP'] = temp_dir
    os.environ['TMP'] = temp_dir
    # 添加当前目录到 sys.path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入 services - 兼容 Vercel 环境
try:
    from .services import ocr_service
except ImportError:
    try:
        from services import ocr_service
    except ImportError:
        # 如果都导入失败，创建一个占位服务
        logger = logging.getLogger(__name__)
        logger.error("无法导入 ocr_service，请检查服务文件")
        ocr_service = None

# 配置常量
DEFAULT_PORT = int(os.getenv('PORT', '5702'))
MAX_REQUEST_SIZE = int(os.getenv('MAX_REQUEST_SIZE', '50'))  # MB

app = FastAPI(
    title="OCR Service",
    description="通用验证码识别服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from starlette.datastructures import UploadFile as StarletteUploadFile


async def decode_image(data: Union[UploadFile, StarletteUploadFile, str, None]) -> bytes:
    """解码图片数据，支持文件、Base64、URL"""
    if data is None:
        raise HTTPException(status_code=400, detail="No image provided")

    if isinstance(data, (UploadFile, StarletteUploadFile)):
        content = await data.read()
        if len(content) > MAX_REQUEST_SIZE * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File too large. Max size: {MAX_REQUEST_SIZE}MB")
        return content
    
    if isinstance(data, str):
        # URL处理
        if data.startswith('http://') or data.startswith('https://'):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                }
                resp = requests.get(data, timeout=30, headers=headers)
                resp.raise_for_status()
                if len(resp.content) > MAX_REQUEST_SIZE * 1024 * 1024:
                    raise HTTPException(status_code=413, detail=f"Image too large. Max size: {MAX_REQUEST_SIZE}MB")
                return resp.content
            except requests.RequestException as e:
                raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {str(e)}")
        
        # Base64处理
        try:
            # 移除 data:image 前缀
            if ',' in data and data.split(',')[0].startswith('data:image'):
                data = data.split(',')[1]
            # 修复 Base64 padding
            missing_padding = len(data) % 4
            if missing_padding:
                data += '=' * (4 - missing_padding)
            decoded = base64.b64decode(data)
            if len(decoded) > MAX_REQUEST_SIZE * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"Image too large. Max size: {MAX_REQUEST_SIZE}MB")
            return decoded
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 string: {str(e)}")
    
    raise HTTPException(status_code=400, detail="Invalid image input")


async def log_request_response(request: Request, response_data: Dict, process_time: float):
    """记录请求响应日志"""
    client_host = request.client.host if request.client else "unknown"
    client_port = request.client.port if request.client else 0
    
    logger.info(
        f"{client_host}:{client_port} - {request.method} {request.url.path} - "
        f"Process Time: {process_time:.3f}s - Status: {response_data.get('status', 'unknown')}"
    )


# ==================== 根路径 ====================
@app.get("/")
async def root():
    """根路径 - 服务信息"""
    # 检查 OCR 服务是否可用
    ocr_status = "available" if ocr_service else "unavailable"
    
    return {
        "service": "OCR Service",
        "version": "1.0.0",
        "environment": "vercel" if IS_VERCEL else "local",
        "ocr_status": ocr_status,
        "endpoints": {
            "/health": "健康检查",
            "/ocr": "通用验证码识别",
            "/ocr/b64/json": "通用验证码识别(兼容路径)",
            "/rotate": "旋转验证码识别",
            "/slide": "滑动验证码识别",
            "/detection": "目标检测"
        },
        "docs": "/docs"
    }


# ==================== CORS 预检请求处理 ====================
@app.options("/{path:path}")
async def options_handler(path: str):
    """处理所有 OPTIONS 预检请求"""
    return {}


# ==================== 健康检查接口 ====================
@app.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        # 检查 OCR 服务是否可用
        ocr_available = ocr_service is not None
        
        return {
            "status": 0, 
            "msg": "success",
            "service": "ocr-service",
            "environment": "vercel" if IS_VERCEL else "local",
            "ocr_available": ocr_available,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": -1, "msg": str(e)}


# ==================== 辅助函数：解析请求体 ====================
async def parse_request_body(request: Request) -> tuple:
    """
    智能解析请求体，支持多种格式
    返回: (data_value, probability, charsets, png_fix)
    """
    content_type = request.headers.get('content-type', '').lower()
    raw_body = await request.body()
    
    # 尝试解析为JSON（无论Content-Type是什么）
    try:
        body_str = raw_body.decode('utf-8')
        
        # 尝试解析JSON
        try:
            body = json.loads(body_str)
            
            if isinstance(body, str):
                # 纯Base64字符串
                logger.info("[OCR] 检测到纯Base64字符串格式")
                return body, False, None, False
            elif isinstance(body, dict):
                # JSON对象
                data_value = body.get('data') or body.get('image') or body.get('img') or body.get('file')
                probability = body.get('probability', False)
                charsets = body.get('charsets', None)
                png_fix = body.get('png_fix', False)
                logger.info("[OCR] 检测到JSON对象格式")
                return data_value, probability, charsets, png_fix
        except json.JSONDecodeError:
            # 不是JSON，当作纯Base64字符串处理
            logger.info("[OCR] 检测到纯文本Base64格式")
            return body_str, False, None, False
            
    except UnicodeDecodeError:
        # 不是文本，可能是二进制数据
        pass
    
    # 处理 multipart/form-data
    if 'multipart/form-data' in content_type:
        try:
            form = await request.form()
            data_file = form.get('data') or form.get('image') or form.get('img') or form.get('file')
            data_value = form.get('data') or form.get('image') or form.get('img') or form.get('file')
            
            probability = form.get('probability', 'false').lower() == 'true'
            charsets = form.get('charsets', None)
            png_fix = form.get('png_fix', 'false').lower() == 'true'
            
            if data_file and hasattr(data_file, 'read'):
                logger.info("[OCR] 检测到文件上传格式")
                return data_file, probability, charsets, png_fix
            elif data_value and isinstance(data_value, str):
                return data_value, probability, charsets, png_fix
        except Exception as e:
            logger.warning(f"[OCR] 解析form-data失败: {str(e)}")
    
    return None, False, None, False


# ==================== OCR 识别接口 ====================
@app.post("/ocr")
async def ocr_endpoint(request: Request):
    start_time = time.time()
    
    try:
        # 检查 OCR 服务是否可用
        if ocr_service is None:
            response = {"status": -1, "msg": "OCR service not available"}
            await log_request_response(request, response, time.time() - start_time)
            return response
        
        # 智能解析请求体
        data_value, probability, charsets, png_fix = await parse_request_body(request)
        
        if not data_value:
            response = {"status": -1, "msg": "请提供 data/image/img/file 参数或纯Base64字符串"}
            await log_request_response(request, response, time.time() - start_time)
            return response
        
        # 解码图片
        image_bytes = await decode_image(data_value)
        
        if not image_bytes:
            response = {"status": -1, "msg": "无法获取图片数据"}
            await log_request_response(request, response, time.time() - start_time)
            return response
        
        # 调用OCR服务
        result = ocr_service.ocr_classification(image_bytes, probability, charsets, png_fix)
        
        logger.info(f"[OCR] 识别结果: {result}")
        
        response = {"status": 0, "data": {"code": result}, "msg": "success"}
        await log_request_response(request, response, time.time() - start_time)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        response = {"status": -1, "msg": str(e)}
        logger.error(f"[OCR] 识别错误: {str(e)}", exc_info=True)
        await log_request_response(request, response, time.time() - start_time)
        return response


# ==================== OCR 兼容路径（/ocr/b64/json） ====================
@app.post("/ocr/b64/json")
async def ocr_b64_json_endpoint(request: Request):
    """兼容 /ocr/b64/json 路径"""
    return await ocr_endpoint(request)


# ==================== 旋转验证码接口 ====================
@app.post("/rotate")
async def rotate_endpoint(request: Request):
    start_time = time.time()
    
    try:
        # 检查 OCR 服务是否可用
        if ocr_service is None:
            response = {"status": -1, "msg": "OCR service not available"}
            await log_request_response(request, response, time.time() - start_time)
            return response
        
        thumb_bytes = None
        bg_bytes = None
        
        content_type = request.headers.get('content-type', '').lower()
        
        if 'application/json' in content_type:
            try:
                body = await request.json()
            except Exception as e:
                response = {"status": -1, "msg": f"Invalid JSON: {str(e)}"}
                await log_request_response(request, response, time.time() - start_time)
                return response
            
            thumb_data = body.get('thumb') or body.get('thumb_img') or body.get('slider')
            bg_data = body.get('bg') or body.get('background') or body.get('bg_img')
            
            if not thumb_data or not bg_data:
                response = {"status": -1, "msg": "请提供 thumb 和 bg 参数"}
                await log_request_response(request, response, time.time() - start_time)
                return response
            
            thumb_bytes = await decode_image(thumb_data)
            bg_bytes = await decode_image(bg_data)
        
        elif 'multipart/form-data' in content_type:
            try:
                form = await request.form()
            except Exception as e:
                response = {"status": -1, "msg": f"Invalid form data: {str(e)}"}
                await log_request_response(request, response, time.time() - start_time)
                return response
            
            thumb_file = form.get('thumb') or form.get('thumb_img') or form.get('slider')
            bg_file = form.get('bg') or form.get('background') or form.get('bg_img')
            
            if not thumb_file or not bg_file:
                response = {"status": -1, "msg": "请提供 thumb 和 bg 参数"}
                await log_request_response(request, response, time.time() - start_time)
                return response
            
            thumb_bytes = await decode_image(thumb_file)
            bg_bytes = await decode_image(bg_file)
        
        else:
            response = {"status": -1, "msg": f"Unsupported Content-Type: {content_type}"}
            await log_request_response(request, response, time.time() - start_time)
            return response
        
        result = ocr_service.rotate_match(thumb_bytes, bg_bytes)
        
        logger.info(f"[ROTATE] 识别结果: cw={result.get('cw', 0)}, ccw={result.get('ccw', 0)}")
        
        response = {"status": 0, "data": result, "msg": "success"}
        await log_request_response(request, response, time.time() - start_time)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        response = {"status": -1, "msg": str(e)}
        logger.error(f"[ROTATE] 识别错误: {str(e)}", exc_info=True)
        await log_request_response(request, response, time.time() - start_time)
        return response


# ==================== 滑动验证码接口 ====================
@app.post("/slide")
async def slide_endpoint(request: Request):
    start_time = time.time()
    
    try:
        # 检查 OCR 服务是否可用
        if ocr_service is None:
            response = {"status": -1, "msg": "OCR service not available"}
            await log_request_response(request, response, time.time() - start_time)
            return response
        
        thumb_bytes = None
        bg_bytes = None
        slide_type = "match"
        
        content_type = request.headers.get('content-type', '').lower()
        
        if 'application/json' in content_type:
            try:
                body = await request.json()
            except Exception as e:
                response = {"status": -1, "msg": f"Invalid JSON: {str(e)}"}
                await log_request_response(request, response, time.time() - start_time)
                return response
            
            thumb_data = body.get('thumb') or body.get('thumb_img') or body.get('slider')
            bg_data = body.get('bg') or body.get('background') or body.get('bg_img')
            slide_type = body.get('type', 'match')
            
            if not thumb_data or not bg_data:
                response = {"status": -1, "msg": "请提供 thumb 和 bg 参数"}
                await log_request_response(request, response, time.time() - start_time)
                return response
            
            thumb_bytes = await decode_image(thumb_data)
            bg_bytes = await decode_image(bg_data)
        
        elif 'multipart/form-data' in content_type:
            try:
                form = await request.form()
            except Exception as e:
                response = {"status": -1, "msg": f"Invalid form data: {str(e)}"}
                await log_request_response(request, response, time.time() - start_time)
                return response
            
            thumb_file = form.get('thumb') or form.get('thumb_img') or form.get('slider')
            bg_file = form.get('bg') or form.get('background') or form.get('bg_img')
            slide_type = form.get('type', 'match')
            
            if not thumb_file or not bg_file:
                response = {"status": -1, "msg": "请提供 thumb 和 bg 参数"}
                await log_request_response(request, response, time.time() - start_time)
                return response
            
            thumb_bytes = await decode_image(thumb_file)
            bg_bytes = await decode_image(bg_file)
        
        else:
            response = {"status": -1, "msg": f"Unsupported Content-Type: {content_type}"}
            await log_request_response(request, response, time.time() - start_time)
            return response
        
        result = ocr_service.slide_match_compatible(thumb_bytes, bg_bytes, slide_type)
        
        logger.info(f"[SLIDE] 识别结果: x={result.get('x', 0)}, y={result.get('y', 0)}")
        
        response = {"status": 0, "data": result, "msg": "success"}
        await log_request_response(request, response, time.time() - start_time)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        response = {"status": -1, "msg": str(e)}
        logger.error(f"[SLIDE] 识别错误: {str(e)}", exc_info=True)
        await log_request_response(request, response, time.time() - start_time)
        return response


# ==================== 目标检测接口 ====================
@app.post("/detection")
async def detection_endpoint(request: Request):
    start_time = time.time()
    
    try:
        # 检查 OCR 服务是否可用
        if ocr_service is None:
            response = {"status": -1, "msg": "OCR service not available"}
            await log_request_response(request, response, time.time() - start_time)
            return response
        
        image_bytes = None
        
        content_type = request.headers.get('content-type', '').lower()
        
        if 'application/json' in content_type:
            try:
                body = await request.json()
            except Exception as e:
                response = {"status": -1, "msg": f"Invalid JSON: {str(e)}"}
                await log_request_response(request, response, time.time() - start_time)
                return response
            
            data_value = body.get('data') or body.get('image') or body.get('img') or body.get('file')
            if not data_value:
                response = {"status": -1, "msg": "请提供 data/image/img/file 参数"}
                await log_request_response(request, response, time.time() - start_time)
                return response
            
            image_bytes = await decode_image(data_value)
        
        elif 'multipart/form-data' in content_type:
            try:
                form = await request.form()
            except Exception as e:
                response = {"status": -1, "msg": f"Invalid form data: {str(e)}"}
                await log_request_response(request, response, time.time() - start_time)
                return response
            
            data_file = form.get('data') or form.get('image') or form.get('img') or form.get('file')
            
            if data_file and hasattr(data_file, 'read'):
                image_bytes = await decode_image(data_file)
            else:
                response = {"status": -1, "msg": "请提供 data/image/img/file 参数"}
                await log_request_response(request, response, time.time() - start_time)
                return response
        
        else:
            response = {"status": -1, "msg": f"Unsupported Content-Type: {content_type}"}
            await log_request_response(request, response, time.time() - start_time)
            return response
        
        bboxes = ocr_service.detection(image_bytes)
        
        logger.info(f"[DETECTION] 检测结果: {bboxes}")
        
        response = {"status": 0, "data": bboxes, "msg": "success"}
        await log_request_response(request, response, time.time() - start_time)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        response = {"status": -1, "msg": str(e)}
        logger.error(f"[DETECTION] 检测错误: {str(e)}", exc_info=True)
        await log_request_response(request, response, time.time() - start_time)
        return response


if __name__ == "__main__":
    # 只在非 Vercel 环境下启动服务器
    if not IS_VERCEL:
        logger.info(f"Starting OCR Service on port {DEFAULT_PORT}")
        logger.info(f"Max request size: {MAX_REQUEST_SIZE}MB")
        logger.info(f"API documentation available at: http://localhost:{DEFAULT_PORT}/docs")
        
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=DEFAULT_PORT,
            limit_concurrency=100,
            timeout_keep_alive=5
        )
    else:
        logger.info("OCR Service running in Vercel environment")