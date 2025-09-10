import os
import re
import logging
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

def create_dynamic_output_folder(base_output_dir: str, audio_source: str, 
                                dynamic_enabled: bool = True, 
                                folder_format: str = "{date}_{time}_{title}") -> str:
    """
    동적 출력 폴더를 생성합니다.
    
    Args:
        base_output_dir: 기본 출력 디렉토리
        audio_source: 오디오 소스 (URL 또는 파일 경로)
        dynamic_enabled: 동적 폴더 생성 활성화 여부
        folder_format: 폴더명 형식 (예: "{date}_{time}_{title}")
        
    Returns:
        생성된 출력 폴더 경로
    """
    if not dynamic_enabled:
        # 동적 폴더 비활성화 시 기본 폴더 사용
        os.makedirs(base_output_dir, exist_ok=True)
        return base_output_dir
    
    # 현재 날짜시각
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    
    # 제목 추출
    title = extract_title_from_source(audio_source)
    
    # 폴더명 생성
    folder_name = folder_format.format(
        date=date_str,
        time=time_str,
        title=title
    )
    
    # 파일명에 사용할 수 없는 문자 제거
    folder_name = sanitize_folder_name(folder_name)
    
    # 최종 출력 경로
    output_path = os.path.join(base_output_dir, folder_name)
    
    # 폴더 생성
    os.makedirs(output_path, exist_ok=True)
    
    logger.info(f"📁 동적 출력 폴더 생성: {output_path}")
    return output_path

def extract_title_from_source(audio_source: str) -> str:
    """
    오디오 소스에서 제목을 추출합니다.
    
    Args:
        audio_source: 오디오 소스 (URL 또는 파일 경로)
        
    Returns:
        추출된 제목
    """
    if audio_source.startswith(("http://", "https://")):
        # YouTube URL에서 제목 추출 시도
        title = extract_youtube_title(audio_source)
        if title:
            return title
        
        # URL에서 기본 제목 추출
        parsed = urlparse(audio_source)
        if parsed.netloc:
            return f"web_{parsed.netloc.replace('.', '_')}"
        
        return "unknown_web"
    else:
        # 로컬 파일에서 제목 추출
        if os.path.exists(audio_source):
            filename = os.path.basename(audio_source)
            # 확장자 제거
            title = os.path.splitext(filename)[0]
            return sanitize_folder_name(title)
        
        return "unknown_file"

def extract_youtube_title(url: str) -> Optional[str]:
    """
    YouTube URL에서 비디오 ID를 추출하여 간단한 제목을 생성합니다.
    실제 제목 추출은 yt-dlp를 사용하지만, 여기서는 간단한 방식 사용.
    
    Args:
        url: YouTube URL
        
    Returns:
        추출된 제목 또는 None
    """
    try:
        # YouTube 비디오 ID 추출
        video_id = None
        
        if "youtube.com/watch" in url:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)
            video_id = query_params.get('v', [None])[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        
        if video_id:
            return f"youtube_{video_id[:8]}"  # 비디오 ID 앞 8자리만 사용
        
        return "youtube_unknown"
        
    except Exception as e:
        logger.warning(f"YouTube 제목 추출 실패: {str(e)}")
        return "youtube_error"

def extract_youtube_title_with_ytdlp(url: str) -> Optional[str]:
    """
    yt-dlp를 사용하여 실제 YouTube 제목을 추출합니다.
    
    Args:
        url: YouTube URL
        
    Returns:
        추출된 제목 또는 None
    """
    try:
        from yt_dlp import YoutubeDL
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', '')
            if title:
                return sanitize_folder_name(title)
        
        return None
        
    except Exception as e:
        logger.warning(f"yt-dlp 제목 추출 실패: {str(e)}")
        return None

def sanitize_folder_name(name: str) -> str:
    """
    폴더명에 사용할 수 없는 문자를 제거하고 정리합니다.
    
    Args:
        name: 원본 이름
        
    Returns:
        정리된 폴더명
    """
    # Windows에서 사용할 수 없는 문자 제거
    invalid_chars = r'[<>:"/\\|?*]'
    name = re.sub(invalid_chars, '_', name)
    
    # 연속된 공백을 언더스코어로 변경
    name = re.sub(r'\s+', '_', name)
    
    # 연속된 언더스코어 정리
    name = re.sub(r'_+', '_', name)
    
    # 앞뒤 언더스코어 제거
    name = name.strip('_')
    
    # 길이 제한 (Windows 경로 길이 제한 고려)
    if len(name) > 50:
        name = name[:50]
    
    # 빈 이름 처리
    if not name:
        name = "untitled"
    
    return name

def get_enhanced_youtube_title(url: str) -> str:
    """
    향상된 YouTube 제목 추출 (yt-dlp 우선, 실패 시 폴백)
    
    Args:
        url: YouTube URL
        
    Returns:
        추출된 제목
    """
    # 먼저 yt-dlp로 실제 제목 추출 시도
    title = extract_youtube_title_with_ytdlp(url)
    if title:
        logger.info(f"🎵 YouTube 제목 추출 성공: {title}")
        return title
    
    # 실패 시 비디오 ID 기반 제목 생성
    fallback_title = extract_youtube_title(url)
    logger.info(f"📹 YouTube 비디오 ID 기반 제목: {fallback_title}")
    return fallback_title or "youtube_unknown"

def save_asr_output(segments, output_dir: str, engine: str) -> dict:
    """
    ASR 단계(가사 파싱 전)에서 생성된 세그먼트를 원본 형태로 보존합니다.

    생성 파일:
    - asr_{engine}_segments.json: 세그먼트 전체를 JSON으로 저장
    - asr_{engine}_segments.txt: 사람이 읽기 쉬운 텍스트 형식 저장

    Returns:
        dict: 생성된 파일 경로 사전
    """
    import json
    import os

    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"asr_{engine}_segments.json")
    txt_path = os.path.join(output_dir, f"asr_{engine}_segments.txt")

    # JSON 저장
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 ASR 원본(JSON) 저장: {json_path}")
    except Exception as e:
        logger.warning(f"ASR 원본 JSON 저장 실패: {e}")

    # TXT 저장 (간단한 가독성 포맷)
    try:
        lines = []
        for seg in segments or []:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            text = (seg.get("text") or "").strip()
            lines.append(f"[{start:8.3f} - {end:8.3f}] {text}")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"💾 ASR 원본(TXT) 저장: {txt_path}")
    except Exception as e:
        logger.warning(f"ASR 원본 TXT 저장 실패: {e}")

    return {"json": json_path, "txt": txt_path}
