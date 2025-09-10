import os
import json
import logging
from typing import List, Dict, Optional
from datetime import timedelta
from text.pronunciation import convert_pronunciation

logger = logging.getLogger(__name__)

def format_time_srt(seconds: float) -> str:
    """SRT 형식의 시간 포맷 (HH:MM:SS,mmm)"""
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    secs = td.total_seconds() % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')

def save_as_srt(aligned_lyrics: List[Dict], output_dir: str, include_pronunciation: bool = False, 
                source_lang: str = 'auto', translated_lyrics: Optional[List[str]] = None,
                filename_prefix: str = "") -> str:
    """정렬된 가사를 SRT 자막 파일로 저장"""
    try:
        logger.info(f"💾 SRT 자막 파일 생성 중...")
        
        os.makedirs(output_dir, exist_ok=True)
        prefix = f"{filename_prefix}_" if filename_prefix else ""
        output_path = os.path.join(output_dir, f"{prefix}subtitles.srt")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, item in enumerate(aligned_lyrics, 1):
                if item['start'] > 0 or item['end'] > 0:  # 유효한 타이밍만
                    start_time = format_time_srt(item['start'])
                    end_time = format_time_srt(item['end'])
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    
                    # 원문 작성
                    f.write(f"{item['text']}")
                    
                    # 번역 추가
                    if translated_lyrics and i-1 < len(translated_lyrics):
                        translation = translated_lyrics[i-1]
                        if translation and translation.strip():
                            f.write(f"\n{translation}")
                    
                    # 한글 발음 표기 추가
                    if include_pronunciation:
                        pronunciation = convert_pronunciation(item['text'], source_lang)
                        if pronunciation != item['text']:  # 변환된 경우만 추가
                            f.write(f"\n({pronunciation})")
                    
                    f.write(f"\n\n")
        
        file_size = os.path.getsize(output_path) / 1024
        logger.info(f"✅ SRT 파일 저장 완료: {output_path} (크기: {file_size:.2f}KB)")
        return output_path
        
    except Exception as e:
        logger.error(f"❌ SRT 파일 저장 실패: {str(e)}")
        raise

def save_as_json(aligned_lyrics: List[Dict], output_dir: str, filename_prefix: str = "") -> str:
    """정렬된 가사를 JSON 파일로 저장"""
    try:
        logger.info(f"💾 JSON 자막 파일 생성 중...")
        
        os.makedirs(output_dir, exist_ok=True)
        prefix = f"{filename_prefix}_" if filename_prefix else ""
        output_path = os.path.join(output_dir, f"{prefix}aligned_subtitles.json")
        
        # 상세 정보 포함한 JSON 저장
        output_data = {
            "metadata": {
                "total_lines": len(aligned_lyrics),
                "valid_timings": len([item for item in aligned_lyrics if item['start'] > 0 or item['end'] > 0]),
                "format": "aligned_lyrics_with_timing"
            },
            "subtitles": aligned_lyrics
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        file_size = os.path.getsize(output_path) / 1024
        logger.info(f"✅ JSON 파일 저장 완료: {output_path} (크기: {file_size:.2f}KB)")
        return output_path
        
    except Exception as e:
        logger.error(f"❌ JSON 파일 저장 실패: {str(e)}")
        raise

def save_aligned_subtitles(aligned_lyrics: List[Dict], output_dir: str, formats: List[str] = None, 
                          include_pronunciation: bool = False, source_lang: str = 'auto', 
                          translated_lyrics: Optional[List[str]] = None,
                          filename_prefix: str = "") -> List[str]:
    """
    정렬된 가사를 여러 형식으로 저장
    
    Args:
        aligned_lyrics: 정렬된 가사 데이터
        output_dir: 출력 디렉토리
        formats: 저장할 형식 리스트 ['srt', 'json'] (기본값: 둘 다)
        include_pronunciation: 한글 발음 표기 포함 여부
        source_lang: 원본 언어 ('en', 'ja', 'auto')
        translated_lyrics: 번역된 가사 리스트
        
    Returns:
        생성된 파일 경로 리스트
    """
    if formats is None:
        formats = ['srt', 'json']
    
    output_files = []
    
    if 'srt' in formats:
        srt_file = save_as_srt(
            aligned_lyrics, output_dir, include_pronunciation, source_lang, translated_lyrics,
            filename_prefix=filename_prefix
        )
        output_files.append(srt_file)
    
    if 'json' in formats:
        json_file = save_as_json(aligned_lyrics, output_dir, filename_prefix=filename_prefix)
        output_files.append(json_file)
    
    return output_files
