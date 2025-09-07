import os
import shutil
from yt_dlp import YoutubeDL
import logging
from typing import Optional

# 로거 설정
logger = logging.getLogger(__name__)

def download_audio(source: str, output_dir: str) -> str:
    """
    YouTube URL 또는 로컬 파일 경로에서 오디오를 다운로드하거나 복사합니다.
    Returns: 로컬 오디오 파일 경로
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"📂 출력 디렉토리 확인: {output_dir}")
        
        if source.startswith(("http://", "https://")):
            logger.info(f"🌐 유튜브 다운로드 시작: {source}")
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
                'quiet': False,
                'logger': logger,
                'progress_hooks': [lambda d: _progress_hook(d, logger)]
            }
            
            with YoutubeDL(ydl_opts) as ydl:
                logger.info("🔍 동영상 정보 추출 중...")
                info = ydl.extract_info(source, download=True)
                filename = ydl.prepare_filename(info)
                logger.info(f"✅ 다운로드 완료: {filename}")
                return filename
                
        else:  # 로컬 파일 처리
            logger.info(f"📁 로컬 파일 복사: {source} → {output_dir}")
            if not os.path.exists(source):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {source}")
                
            dest = os.path.join(output_dir, os.path.basename(source))
            shutil.copy(source, dest)
            logger.info(f"✅ 파일 복사 완료: {dest}")
            return dest
            
    except Exception as e:
        logger.error(f"❌ 오디오 다운로드 실패: {str(e)}")
        raise

def _progress_hook(d: dict, logger: logging.Logger) -> None:
    """다운로드 진행 상황을 로깅하는 훅"""
    if d['status'] == 'downloading':
        percent = d.get('_percent_str', '0%').strip()
        speed = d.get('_speed_str', 'N/A').strip()
        eta = d.get('_eta_str', 'N/A').strip()
        
        # 진행률이 10% 단위로 갱신될 때만 로깅
        if percent.endswith('%'):
            try:
                pct = float(percent[:-1])
                if pct % 10 < 0.1:  # 10% 단위로만 로깅
                    logger.info(f"⬇️ 다운로드 진행 중... {percent} 완료 | 속도: {speed}/s | 남은 시간: {eta}")
            except ValueError:
                pass
