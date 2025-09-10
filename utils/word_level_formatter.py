import os
import json
import logging
from typing import List, Dict, Optional
from datetime import timedelta

logger = logging.getLogger(__name__)

def format_time_srt(seconds: float) -> str:
    """SRT 형식의 시간 포맷 (HH:MM:SS,mmm)"""
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    secs = td.total_seconds() % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')

def extract_word_level_segments(whisperx_segments: List[Dict]) -> List[Dict]:
    """
    WhisperX 세그먼트에서 단어 단위 세그먼트를 추출합니다.
    
    Args:
        whisperx_segments: WhisperX 결과 세그먼트 (words 배열 포함)
        
    Returns:
        단어 단위 세그먼트 리스트
    """
    word_segments = []
    
    for seg in whisperx_segments:
        words = seg.get("words", [])
        if not words:
            # words가 없으면 기존 세그먼트 그대로 사용
            word_segments.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": seg.get("text", "").strip(),
                "type": "segment"
            })
            continue
            
        # 각 단어를 개별 세그먼트로 변환
        for word in words:
            word_text = word.get("word", "").strip()
            if word_text:
                word_segments.append({
                    "start": float(word.get("start", 0.0)),
                    "end": float(word.get("end", 0.0)),
                    "text": word_text,
                    "type": "word",
                    "score": float(word.get("score", 0.0)),
                    "speaker": word.get("speaker", None)
                })
    
    # 시간순 정렬
    word_segments.sort(key=lambda x: x["start"])
    
    logger.info(f"📝 단어 단위 세그먼트 추출: {len(word_segments)}개")
    return word_segments

def save_word_level_subtitles(whisperx_segments: List[Dict], output_dir: str, 
                             filename_prefix: str = "word_level") -> List[str]:
    """
    WhisperX 단어 단위 정보를 사용하여 정밀 자막을 저장합니다.
    
    Args:
        whisperx_segments: WhisperX 결과 세그먼트
        output_dir: 출력 디렉토리
        filename_prefix: 파일명 접두사
        
    Returns:
        생성된 파일 경로 리스트
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 단어 단위 세그먼트 추출
        word_segments = extract_word_level_segments(whisperx_segments)
        
        output_files = []
        
        # SRT 파일 생성
        srt_path = os.path.join(output_dir, f"{filename_prefix}_subtitles.srt")
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, word_seg in enumerate(word_segments, 1):
                if word_seg['start'] >= 0 and word_seg['end'] > word_seg['start']:
                    start_time = format_time_srt(word_seg['start'])
                    end_time = format_time_srt(word_seg['end'])
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{word_seg['text']}\n\n")
        
        output_files.append(srt_path)
        logger.info(f"✅ 단어 단위 SRT 저장: {srt_path}")
        
        # JSON 파일 생성
        json_path = os.path.join(output_dir, f"{filename_prefix}_aligned_subtitles.json")
        output_data = {
            "metadata": {
                "total_segments": len(word_segments),
                "word_level": True,
                "format": "whisperx_word_level_timing"
            },
            "subtitles": word_segments
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        output_files.append(json_path)
        logger.info(f"✅ 단어 단위 JSON 저장: {json_path}")
        
        return output_files
        
    except Exception as e:
        logger.error(f"❌ 단어 단위 자막 저장 실패: {str(e)}")
        raise
