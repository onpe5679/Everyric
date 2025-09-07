import os
import logging
from typing import List, Dict

# 로거 설정
logger = logging.getLogger(__name__)

def save_transcription_text(segments: List[Dict], output_dir: str) -> str:
    """
    Whisper 음성 인식 결과를 텍스트 파일로 저장합니다.
    
    Args:
        segments: 음성 인식 세그먼트 리스트 (start, end, text 키 포함)
        output_dir: 출력 디렉토리 경로
        
    Returns:
        str: 생성된 텍스트 파일의 전체 경로
    """
    try:
        logger.info(f" 음성 인식 결과 저장 중... (디렉토리: {output_dir})")
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "transcription.txt")
        
        # 텍스트 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== Whisper 음성 인식 결과 ===\n\n")
            
            for i, segment in enumerate(segments):
                start_time = _format_time(segment.get('start', 0))
                end_time = _format_time(segment.get('end', 0))
                text = segment.get('text', '').strip()
                
                f.write(f"[{start_time} → {end_time}] {text}\n")
                
                # 첫 5개와 마지막 5개 세그먼트만 로깅
                if i < 5 or i >= len(segments) - 5:
                    logger.debug(f"세그먼트 {i+1}/{len(segments)}: [{start_time} → {end_time}] {text}")
        
        file_size = os.path.getsize(output_path) / 1024  # KB 단위
        logger.info(f" 음성 인식 결과 저장 완료: {output_path} (크기: {file_size:.2f}KB, 세그먼트: {len(segments)}개)")
        
        return output_path
        
    except Exception as e:
        logger.error(f" 음성 인식 결과 저장 중 오류 발생: {str(e)}")
        raise

def _format_time(seconds: float) -> str:
    """초 단위 시간을 MM:SS.sss 형식으로 변환"""
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes):02d}:{seconds:06.3f}"
