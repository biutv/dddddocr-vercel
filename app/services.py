import ddddocr
import cv2
import numpy as np
from typing import Union, List, Optional, Dict, Any

class OCRService:
    def __init__(self):
        self.ocr = ddddocr.DdddOcr()
        self.det = ddddocr.DdddOcr(det=True)
        # slide 不再需要，改用 OpenCV 算法

    def ocr_classification(self, image: bytes, probability: bool = False, charsets: Optional[str] = None, png_fix: bool = False) -> Union[str, dict]:
        """通用 OCR 识别"""
        if charsets:
            self.ocr.set_ranges(charsets)
        result = self.ocr.classification(image, probability=probability, png_fix=png_fix)
        return result

    def slide_match(self, target: bytes, background: bytes, simple_target: bool = False) -> List[int]:
        """滑动验证码匹配（使用 OpenCV 边缘检测算法）"""
        try:
            # 将字节数据解码为 OpenCV 图像
            target_np = np.frombuffer(target, np.uint8)
            bg_np = np.frombuffer(background, np.uint8)
            
            target_img = cv2.imdecode(target_np, cv2.IMREAD_COLOR)
            bg_img = cv2.imdecode(bg_np, cv2.IMREAD_COLOR)
            
            if target_img is None or bg_img is None:
                return [0, 0, 0, 0]
            
            # 转为灰度图
            gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
            gray_bg = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
            
            if simple_target:
                # 差异算法
                diff = cv2.absdiff(gray_target, gray_bg)
                _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
                kernel = np.ones((3, 3), np.uint8)
                morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                max_area = 0
                best_x, best_y, best_w, best_h = 0, 0, 0, 0
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    area = w * h
                    if area > max_area:
                        max_area = area
                        best_x, best_y, best_w, best_h = x, y, w, h
                return [best_x, best_y, best_w, best_h]
            else:
                # 边缘检测算法（默认，与 ddddocr-node-bin 一致）
                blur_target = cv2.GaussianBlur(gray_target, (3, 3), 0)
                blur_bg = cv2.GaussianBlur(gray_bg, (3, 3), 0)
                
                edge_target = cv2.Canny(blur_target, 100, 200)
                edge_bg = cv2.Canny(blur_bg, 100, 200)
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                cv2.dilate(edge_target, edge_target, kernel, iterations=2)
                cv2.dilate(edge_bg, edge_bg, kernel, iterations=2)
                
                result = cv2.matchTemplate(edge_bg, edge_target, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                h, w = gray_target.shape
                return [max_loc[0], max_loc[1], w, h]
                
        except Exception as e:
            print(f"Slide match error: {str(e)}")
            return [0, 0, 0, 0]

    def detection(self, image: bytes) -> List[List[int]]:
        """目标检测"""
        bboxes = self.det.detection(image)
        return bboxes

    def rotate_match(self, thumb: bytes, bg: bytes) -> Dict[str, int]:
        """
        旋转验证码识别 - 移植自 ddddocr-node-bin
        参数:
            thumb: 滑块图片字节
            bg: 背景图片字节
        返回:
            {"cw": 顺时针角度, "ccw": 逆时针角度}
        """
        try:
            # 将字节数据解码为 OpenCV 图像
            thumb_np = np.frombuffer(thumb, np.uint8)
            bg_np = np.frombuffer(bg, np.uint8)
            
            thumb_img = cv2.imdecode(thumb_np, cv2.IMREAD_COLOR)
            bg_img = cv2.imdecode(bg_np, cv2.IMREAD_COLOR)
            
            if thumb_img is None or bg_img is None:
                return {"cw": 0, "ccw": 0}
            
            # 转为灰度图
            gray_bg = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
            gray_thumb = cv2.cvtColor(thumb_img, cv2.COLOR_BGR2GRAY)
            
            h, w = gray_thumb.shape
            
            # 取背景图中心区域（与滑块图大小相同）
            y1 = max(0, (gray_bg.shape[0] - h) // 2)
            x1 = max(0, (gray_bg.shape[1] - w) // 2)
            roi = gray_bg[y1:y1+h, x1:x1+w]
            
            best_angle = 0
            max_score = -1
            
            # 粗算：步长 5 度
            for angle in range(0, 360, 5):
                # 旋转滑块图
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                rotated = cv2.warpAffine(gray_thumb, M, (w, h))
                
                # 模板匹配
                result = cv2.matchTemplate(roi, rotated, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(result)
                
                if score > max_score:
                    max_score = score
                    best_angle = angle
            
            # 精算：在最佳角度附近 ±5 度，步长 1 度
            for angle in range(best_angle - 5, best_angle + 6):
                if angle < 0 or angle > 360:
                    continue
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                rotated = cv2.warpAffine(gray_thumb, M, (w, h))
                result = cv2.matchTemplate(roi, rotated, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(result)
                
                if score > max_score:
                    max_score = score
                    best_angle = angle
            
            # 返回结果（与 Node.js 版本一致）
            cw = (360 - best_angle) % 360
            ccw = best_angle % 360
            
            return {"cw": cw, "ccw": ccw}
            
        except Exception as e:
            print(f"Rotate match error: {str(e)}")
            return {"cw": 0, "ccw": 0}

    def slide_match_compatible(self, thumb: bytes, bg: bytes, slide_type: str = "match") -> Dict[str, Any]:
        """
        滑动验证码识别（完全兼容 ddddocr-node-bin）
        参数:
            thumb: 滑块图片字节
            bg: 背景图片字节
            slide_type: 'match' 边缘算法, 'comparison' 差异算法
        返回:
            包含 x, y, target 等多种格式
        """
        try:
            simple_target = (slide_type == "comparison")
            result = self.slide_match(thumb, bg, simple_target)
            
            x = result[0]
            y = result[1]
            
            return {
                "x": x,
                "y": y,
                "target": result,        # [x, y, w, h]
                "target_x": x,
                "bestX": x
            }
        except Exception as e:
            print(f"Slide match compatible error: {str(e)}")
            return {"x": 0, "y": 0, "target": [0, 0, 0, 0], "target_x": 0, "bestX": 0}


ocr_service = OCRService()