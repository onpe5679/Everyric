import whisper
import torch
from typing import List, Dict
import os
import sys

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

def transcribe_audio(audio_path: str, device: str = "auto", model_name: str = "tiny") -> List[Dict]:
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
                verbose=True  # Whisper 자체 진행률 표시 활성화
            )
        
        segments = result.get("segments", [])
        logger.info(f"✅ 음성 인식 완료. {len(segments)}개의 세그먼트를 찾았습니다.")
        return segments
        
    except Exception as e:
        logger.error(f"❌ 음성 인식 중 오류 발생: {str(e)}")
        raise
