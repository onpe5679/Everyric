import whisper
import torch
from typing import List, Dict
import os
import sys
import re

# 상위 디렉토리에서 모듈 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger, log_step

logger = setup_logger(__name__)

def _get_device(device_preference: str) -> str:
    """최적의 장치를 선택합니다."""
    if device_preference == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"🚀 CUDA GPU 감지됨! GPU로 처리합니다. (GPU 개수: {torch.cuda.device_count()})")
        else:
            device = "cpu"
            logger.info("💻 GPU를 사용할 수 없습니다. CPU로 처리합니다.")
    elif device_preference == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"🚀 GPU 강제 사용 모드. (GPU 개수: {torch.cuda.device_count()})")
        else:
            logger.warning("⚠️ CUDA GPU를 사용할 수 없습니다. CPU로 폴백합니다.")
            device = "cpu"
    else:
        device = "cpu"
        logger.info("💻 CPU 사용 모드")
    
    return device

def transcribe_audio(
    audio_path: str,
    device: str = "auto",
    model_name: str = "tiny",
    *,
    condition_on_previous_text: bool = False,
    temperature: float = 0.0,
    max_segment_duration: float = 6.0,
    split_on_punctuation: bool = True,
) -> List[Dict]:
    """
    openai-whisper 기반 음성 인식 및 타이밍 추출
    Returns:
      List of segments, each with 'text', 'start', 'end'
    """
    logger.info(f"🔍 오디오 파일 분석 시작: {audio_path}")
    logger.info(f"🤖 사용할 Whisper 모델: {model_name}")
    
    # 최적 장치 선택
    actual_device = _get_device(device)
    
    # 모델 로드
    with log_step("Whisper 모델 로드", logger):
        try:
            model = whisper.load_model(model_name, device=actual_device)
            logger.info(f"✅ 모델이 {actual_device}에 성공적으로 로드되었습니다.")
        except Exception as e:
            if actual_device == "cuda":
                logger.warning(f"⚠️ GPU 모델 로드 실패: {str(e)}")
                logger.info("🔄 CPU로 폴백합니다...")
                actual_device = "cpu"
                model = whisper.load_model(model_name, device=actual_device)
                logger.info("✅ CPU에서 모델 로드 완료")
            else:
                raise
    
    # 음성 인식 실행
    try:
        with log_step("음성 인식 실행", logger):
            logger.info("🎯 Whisper 모델로 음성 분석 중... (진행률은 내부적으로 표시됩니다)")
            result = model.transcribe(
                audio_path,
                verbose=True,  # Whisper 자체 진행률 표시 활성화
                condition_on_previous_text=condition_on_previous_text,
                temperature=temperature,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
            )
        
        segments = result.get("segments", [])
        # 과도하게 긴 세그먼트는 문장부호 기준으로 분할하여 시간을 비례 분배
        if split_on_punctuation and segments:
            segments = _split_long_segments(segments, max_segment_duration)
        logger.info(f"✅ 음성 인식 완료. {len(segments)}개의 세그먼트를 찾았습니다.")
        return segments
        
    except Exception as e:
        logger.error(f"❌ 음성 인식 중 오류 발생: {str(e)}")
        raise


def _split_long_segments(segments: List[Dict], max_duration: float) -> List[Dict]:
    """세그먼트가 너무 길면 문장부호(.,!?/…/。/、/！/？/・/♪/\n) 기준으로 나누고,
    각 조각의 문자 비율로 시간을 분배합니다."""
    punct_pattern = re.compile(r"([。．\.、，,！!？\?…・♪\n])")
    out: List[Dict] = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = (seg.get("text") or "").strip()
        duration = max(0.0, end - start)

        if duration <= max_duration or not text:
            out.append(seg)
            continue

        # 문장부호 보존 분할: 구분자도 토큰으로 유지 후 재결합
        parts = [t for t in punct_pattern.split(text) if t is not None and t != ""]
        # 구분자까지 합쳐 문장 단위 리스트 생성
        sentences: List[str] = []
        buf = ""
        for token in parts:
            buf += token
            if punct_pattern.fullmatch(token):
                sentences.append(buf.strip())
                buf = ""
        if buf.strip():
            sentences.append(buf.strip())

        total_chars = sum(len(s) for s in sentences) or 1
        # 최소 1문장 보장
        cur = start
        for i, s in enumerate(sentences):
            frac = len(s) / total_chars
            seg_dur = duration * frac
            # 경계 누적 오차 최소화: 마지막 문장은 end에 스냅
            seg_start = cur
            seg_end = end if i == len(sentences) - 1 else min(end, cur + seg_dur)
            cur = seg_end
            if s:
                out.append({
                    "start": float(seg_start),
                    "end": float(seg_end),
                    "text": s,
                })
    return out
