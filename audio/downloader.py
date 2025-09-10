import os
import shutil
from yt_dlp import YoutubeDL
import logging
from typing import Optional

# 로거 설정
logger = logging.getLogger(__name__)

def download_audio(source: str, output_dir: str, save_audio: bool = False, audio_format: str = "wav") -> str:
    """
    YouTube URL 또는 로컬 파일 경로에서 오디오를 다운로드하거나 복사합니다.
    
    Args:
        source: 오디오 소스 (URL 또는 로컬 파일 경로)
        output_dir: 출력 디렉토리
        save_audio: 오디오 파일 보존 여부
        audio_format: 저장할 오디오 형식 (wav, mp3 등)
        
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
                
                # 오디오 파일 보존 옵션이 활성화된 경우 최종 출력 디렉토리에 복사
                if save_audio:
                    saved_audio_path = _save_audio_file(filename, output_dir, audio_format)
                    logger.info(f"💾 오디오 파일 보존: {saved_audio_path}")
                
                return filename
                
        else:  # 로컬 파일 처리
            logger.info(f"📁 로컬 파일 복사: {source} → {output_dir}")
            if not os.path.exists(source):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {source}")
                
            dest = os.path.join(output_dir, os.path.basename(source))
            shutil.copy(source, dest)
            logger.info(f"✅ 파일 복사 완료: {dest}")
            
            # 로컬 파일도 보존 옵션 적용
            if save_audio:
                saved_audio_path = _save_audio_file(dest, output_dir, audio_format)
                logger.info(f"💾 오디오 파일 보존: {saved_audio_path}")
            
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

def _save_audio_file(source_path: str, output_dir: str, audio_format: str) -> str:
    """
    오디오 파일을 최종 출력 디렉토리에 지정된 형식으로 저장합니다.
    
    Args:
        source_path: 원본 오디오 파일 경로
        output_dir: 출력 디렉토리
        audio_format: 저장할 오디오 형식
        
    Returns: 저장된 오디오 파일 경로
    """
    try:
        import subprocess
        from pathlib import Path
        
        # 파일명 생성 (확장자 변경)
        source_name = Path(source_path).stem
        output_filename = f"audio.{audio_format}"
        output_path = os.path.join(output_dir, output_filename)
        
        # FFmpeg를 사용하여 오디오 형식 변환
        if audio_format.lower() == "wav":
            # WAV 형식으로 변환
            cmd = [
                "ffmpeg", "-i", source_path, 
                "-acodec", "pcm_s16le", 
                "-ar", "44100", 
                "-y", output_path
            ]
        elif audio_format.lower() == "mp3":
            # MP3 형식으로 변환
            cmd = [
                "ffmpeg", "-i", source_path,
                "-acodec", "libmp3lame",
                "-ab", "192k",
                "-y", output_path
            ]
        else:
            # 기본적으로 원본 파일 복사
            shutil.copy(source_path, output_path)
            return output_path
        
        # FFmpeg 실행 (Windows cp949 환경에서의 디코딩 오류 방지)
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.returncode == 0:
            logger.info(f"✅ 오디오 변환 완료: {output_path}")
            return output_path
        else:
            logger.warning(f"⚠️ FFmpeg 변환 실패, 원본 파일 복사: {result.stderr}")
            # 변환 실패 시 원본 파일 복사
            fallback_path = os.path.join(output_dir, f"audio{Path(source_path).suffix}")
            shutil.copy(source_path, fallback_path)
            return fallback_path
            
    except Exception as e:
        logger.warning(f"⚠️ 오디오 저장 중 오류 발생, 원본 파일 복사: {str(e)}")
        # 오류 발생 시 원본 파일 복사
        fallback_path = os.path.join(output_dir, f"audio{Path(source_path).suffix}")
        shutil.copy(source_path, fallback_path)
        return fallback_path
