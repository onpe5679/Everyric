import os
import sys
import torch
from typing import List, Dict, Optional, Tuple
import tempfile
import json

# 상위 디렉토리에서 모듈 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger, log_step

logger = setup_logger(__name__)

def transcribe_audio_with_whisperx(
    audio_path: str,
    device: str = "auto",
    model_name: str = "large-v3",
    *,
    batch_size: int = 16,
    compute_type: str = "float16",
    language: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    max_segment_duration: Optional[float] = None,
) -> List[Dict]:
    """
    WhisperX 기반 정밀 음성 인식 및 단어 단위 타이밍 추출
    
    Args:
        audio_path: 오디오 파일 경로
        device: 처리 장치 ("auto", "cuda", "cpu")
        model_name: Whisper 모델명
        batch_size: 배치 크기 (GPU 메모리에 따라 조정)
        compute_type: 연산 정밀도 ("float16", "int8", "float32")
        language: 언어 코드 (None이면 자동 감지)
        min_speakers: 최소 화자 수 (화자 분리용)
        max_speakers: 최대 화자 수 (화자 분리용)
    
    Returns:
        List of segments with precise word-level timing
    """
    try:
        # cuDNN 문제 해결을 위한 환경변수 설정
        import os
        
        # cuDNN DLL을 직접 찾아서 PATH에 추가
        import glob
        
        # 가능한 cuDNN 설치 경로들
        search_paths = [
            r"C:\Program Files\NVIDIA\CUDNN",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin",
            r"C:\tools\cuda",
            r"C:\cuda"
        ]
        
        cudnn_dll_found = False
        current_path = os.environ.get("PATH", "")
        
        for search_path in search_paths:
            # 와일드카드 패턴 확장
            if '*' in search_path:
                expanded_paths = glob.glob(search_path)
            else:
                expanded_paths = [search_path]
            
            for base_path in expanded_paths:
                if not os.path.exists(base_path):
                    continue
                    
                # bin 폴더와 lib 폴더 모두 확인
                for subdir in ['bin', 'lib', '']:
                    dll_path = os.path.join(base_path, subdir) if subdir else base_path
                    if os.path.exists(dll_path):
                        # cudnn_ops_infer64_8.dll 파일이 있는지 확인
                        dll_files = glob.glob(os.path.join(dll_path, "cudnn_ops_infer64_*.dll"))
                        if dll_files and dll_path not in current_path:
                            os.environ["PATH"] = f"{dll_path};{current_path}"
                            logger.info(f"✅ cuDNN DLL 경로를 PATH에 추가: {dll_path}")
                            current_path = os.environ["PATH"]
                            cudnn_dll_found = True
        
        if not cudnn_dll_found:
            logger.warning("⚠️ cuDNN DLL을 찾을 수 없습니다. GPU 성능이 제한될 수 있습니다.")
        
        # 추가 CUDA 환경변수 설정
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
        import whisperx
    except ImportError:
        logger.error("WhisperX가 설치되지 않았습니다.")
        logger.error("설치 방법:")
        logger.error("  pip install whisperx")
        logger.error("  또는")
        logger.error("  pip install git+https://github.com/m-bain/whisperx.git")
        raise ImportError("WhisperX not installed")

    logger.info(f"🔍 WhisperX로 오디오 파일 분석 시작: {audio_path}")
    logger.info(f"🤖 사용할 WhisperX 모델: {model_name}")
    
    # 장치 설정
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"⚙️ 처리 장치: {device}")
    logger.info(f"📊 배치 크기: {batch_size}, 연산 정밀도: {compute_type}")

    try:
        # 1단계: Whisper 모델 로드 및 전사
        with log_step("WhisperX 모델 로드", logger):
            try:
                model = whisperx.load_model(
                    model_name, 
                    device, 
                    compute_type=compute_type,
                    language=language
                )
                logger.info("✅ WhisperX 모델 로드 완료")
            except Exception as e:
                if "cudnn" in str(e).lower():
                    logger.warning(f"cuDNN 오류 발생, CPU로 폴백: {e}")
                    device = "cpu"
                    compute_type = "float32"
                    model = whisperx.load_model(
                        model_name, 
                        device, 
                        compute_type=compute_type,
                        language=language
                    )
                    logger.info("✅ WhisperX 모델 로드 완료 (CPU 모드)")
                else:
                    raise

        with log_step("음성 전사", logger):
            logger.info("🎯 WhisperX로 음성 전사 중...")
            audio = whisperx.load_audio(audio_path)
            result = model.transcribe(audio, batch_size=batch_size)
            logger.info(f"✅ 전사 완료. 언어: {result.get('language', 'unknown')}")

        # 2단계: 정렬 모델 로드 및 단어 단위 타이밍 추출
        with log_step("단어 단위 정렬", logger):
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=result["language"], 
                    device=device
                )
                result = whisperx.align(
                    result["segments"], 
                    model_a, 
                    metadata, 
                    audio, 
                    device, 
                    return_char_alignments=False
                )
                logger.info("✅ 단어 단위 정렬 완료")
            except Exception as e:
                if "cudnn" in str(e).lower() or "cuda" in str(e).lower():
                    logger.warning(f"정렬 단계에서 CUDA 오류 발생, CPU로 재시도: {e}")
                    model_a, metadata = whisperx.load_align_model(
                        language_code=result["language"], 
                        device="cpu"
                    )
                    result = whisperx.align(
                        result["segments"], 
                        model_a, 
                        metadata, 
                        audio, 
                        "cpu", 
                        return_char_alignments=False
                    )
                    logger.info("✅ 단어 단위 정렬 완료 (CPU 모드)")
                else:
                    logger.warning(f"정렬 실패, 기본 세그먼트 사용: {e}")
                    # 정렬 실패 시 기본 세그먼트만 사용

        # 3단계: 화자 분리 (선택적)
        if min_speakers is not None or max_speakers is not None:
            with log_step("화자 분리", logger):
                try:
                    diarize_model = whisperx.DiarizationPipeline(
                        use_auth_token=None,  # HuggingFace 토큰이 필요할 수 있음
                        device=device
                    )
                    diarize_segments = diarize_model(
                        audio, 
                        min_speakers=min_speakers, 
                        max_speakers=max_speakers
                    )
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    logger.info("✅ 화자 분리 완료")
                except Exception as e:
                    logger.warning(f"화자 분리 실패 (선택적 기능): {e}")

        # 결과 변환: WhisperX 형식을 기존 형식으로 변환
        segments = []
        for seg in result.get("segments", []):
            segment = {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": seg.get("text", "").strip(),
            }
            
            # 단어 단위 정보 추가 (있는 경우)
            if "words" in seg:
                segment["words"] = []
                for word in seg["words"]:
                    word_info = {
                        "word": word.get("word", ""),
                        "start": float(word.get("start", 0.0)),
                        "end": float(word.get("end", 0.0)),
                        "score": float(word.get("score", 0.0)),
                    }
                    if "speaker" in word:
                        word_info["speaker"] = word["speaker"]
                    segment["words"].append(word_info)
            
            if "speaker" in seg:
                segment["speaker"] = seg["speaker"]
                
            segments.append(segment)

        logger.info(f"✅ WhisperX 음성 인식 완료. {len(segments)}개의 세그먼트를 찾았습니다.")
        
        # 단어 단위 타이밍 품질 검증
        word_count = sum(len(seg.get("words", [])) for seg in segments)
        if word_count > 0:
            logger.info(f"📝 단어 단위 타이밍: {word_count}개 단어")
        
        # 세그먼트 길이 제한이 설정된 경우 분할 적용
        if max_segment_duration and max_segment_duration > 0:
            try:
                segments = _split_long_segments(segments, max_segment_duration)
                logger.info(f"✂️ 긴 세그먼트 분할 적용(max {max_segment_duration:.2f}s): 총 {len(segments)}개")
            except Exception as e:
                logger.warning(f"세그먼트 분할 중 오류: {e}")
        return segments
        
    except Exception as e:
        logger.error(f"❌ WhisperX 음성 인식 중 오류 발생: {str(e)}")
        raise

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

def _split_long_segments(segments: List[Dict], max_duration: float) -> List[Dict]:
    """
    세그먼트가 max_duration을 초과하면 단어 경계(가능하면)에 맞춰 분할합니다.
    단어 정보가 없으면 균등 시간 간격으로 분할합니다.
    """
    import math
    new_segments: List[Dict] = []
    for seg in segments:
        start = float(seg.get('start', 0.0))
        end = float(seg.get('end', 0.0))
        text = seg.get('text', '')
        words = seg.get('words', []) or []
        duration = max(0.0, end - start)
        if duration <= max_duration or end <= start:
            new_segments.append(seg)
            continue

        if words:
            # 단어 경계 기준으로 슬라이스
            bucket: List[Dict] = []
            bucket_start = None
            for w in words:
                w_s = float(w.get('start', start))
                w_e = float(w.get('end', w_s))
                if bucket_start is None:
                    bucket_start = w_s
                bucket.append(w)
                if (w_e - bucket_start) >= max_duration:
                    # flush bucket
                    chunk_text = ' '.join((wi.get('word') or '').strip() for wi in bucket).strip()
                    chunk = {
                        'start': bucket_start,
                        'end': w_e,
                        'text': chunk_text or text,
                        'words': list(bucket),
                    }
                    new_segments.append(chunk)
                    bucket = []
                    bucket_start = None
            if bucket:
                chunk_text = ' '.join((wi.get('word') or '').strip() for wi in bucket).strip()
                chunk = {
                    'start': bucket_start if bucket_start is not None else start,
                    'end': float(bucket[-1].get('end', end)),
                    'text': chunk_text or text,
                    'words': list(bucket),
                }
                new_segments.append(chunk)
        else:
            # 균등 분할
            parts = max(1, math.ceil(duration / max_duration))
            chunk_len = duration / parts
            for i in range(parts):
                c_s = start + i * chunk_len
                c_e = min(end, c_s + chunk_len)
                # 텍스트도 대략 균등 분할
                if text and parts > 1:
                    n = len(text)
                    s_idx = int(i * n / parts)
                    e_idx = int((i + 1) * n / parts)
                    c_text = text[s_idx:e_idx].strip() or text
                else:
                    c_text = text
                new_segments.append({'start': c_s, 'end': c_e, 'text': c_text})
    return new_segments

def detect_silence_segments(audio_path: str, min_silence_len: float = 0.5, silence_thresh: float = -40.0) -> List[Tuple[float, float]]:
    """
    오디오에서 무성 구간을 감지합니다.
    
    Args:
        audio_path: 오디오 파일 경로
        min_silence_len: 최소 무성 구간 길이 (초)
        silence_thresh: 무성 판정 임계값 (dB)
    
    Returns:
        List of (start, end) tuples for silence segments
    """
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_silence
    except ImportError:
        logger.warning("pydub가 설치되지 않아 무성 구간 감지를 건너뜁니다.")
        logger.info("설치: pip install pydub")
        return []
    
    try:
        logger.info("🔇 무성 구간 감지 시작...")
        
        # 오디오 로드
        audio = AudioSegment.from_file(audio_path)
        
        # 무성 구간 감지
        silence_segments = detect_silence(
            audio,
            min_silence_len=int(min_silence_len * 1000),  # ms 단위로 변환
            silence_thresh=silence_thresh
        )
        
        # 초 단위로 변환
        silence_ranges = [(start/1000.0, end/1000.0) for start, end in silence_segments]
        
        logger.info(f"🔇 {len(silence_ranges)}개의 무성 구간을 감지했습니다.")
        
        return silence_ranges
        
    except Exception as e:
        logger.warning(f"무성 구간 감지 실패: {e}")
        return []

def apply_silence_based_timing_correction(segments: List[Dict], silence_ranges: List[Tuple[float, float]]) -> List[Dict]:
    """
    무성 구간 정보를 사용하여 세그먼트 타이밍을 보정합니다.
    """
    if not silence_ranges:
        return segments
    
    logger.info("🔧 무성 구간 기반 타이밍 보정 적용...")
    
    corrected_segments = []
    
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        
        # 세그먼트가 무성 구간과 겹치는지 확인
        overlapping_silence = []
        for sil_start, sil_end in silence_ranges:
            if not (end <= sil_start or start >= sil_end):  # 겹침 있음
                overlap_start = max(start, sil_start)
                overlap_end = min(end, sil_end)
                overlapping_silence.append((overlap_start, overlap_end))
        
        # 겹치는 무성 구간이 세그먼트의 50% 이상이면 타이밍 조정
        total_overlap = sum(oe - os for os, oe in overlapping_silence)
        segment_duration = end - start
        
        if segment_duration > 0 and total_overlap / segment_duration > 0.5:
            # 무성 구간을 피해 타이밍 조정
            # 간단한 휴리스틱: 무성 구간 직전/직후로 이동
            for sil_start, sil_end in silence_ranges:
                if start < sil_start < end:
                    seg["end"] = sil_start
                    break
                elif start < sil_end < end:
                    seg["start"] = sil_end
                    break
        
        corrected_segments.append(seg)
    
    return corrected_segments
