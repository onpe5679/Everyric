import os
import json
import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class EveryricConfig:
    """Everyric 설정 클래스"""
    
    # 기본 설정
    audio: str = ""
    lyrics: str = ""
    output: str = "output"
    
    # Whisper 설정
    model: str = "tiny"
    device: str = "auto"
    
    # 자막 설정
    max_length: int = 100
    
    # 정렬 엔진 선택
    alignment_engine: str = "dtw"  # "dtw" 또는 "llm"
    llm_alignment_engine: str = "gpt"  # "gpt" 또는 "gemini" (LLM-only 정렬용)

    # 발음 표기 설정
    pronunciation: bool = False
    source_lang: str = "auto"
    
    # 번역 설정
    translate: bool = False
    target_lang: str = "ko"
    context: str = ""
    translation_engine: str = "gemini"  # 'gemini' 또는 'gpt'
    # OpenAI(GPT)
    gpt_model: str = "gpt-4"
    gpt_max_retries: int = 5
    gpt_retry_delay: int = 2
    # Google Gemini
    google_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    gemini_max_retries: int = 5
    gemini_retry_delay: int = 2
    
    # API 설정
    openai_api_key: str = ""
    
    # 로그 설정
    log_level: str = "INFO"

    # 출력 소음 제어
    suppress_progress_bars: bool = True   # tqdm/HF 진행바 숨김
    suppress_warnings: bool = True        # 불필요한 경고 억제

    # 출력/보존 설정
    save_audio: bool = False
    audio_format: str = "wav"
    dynamic_output_folder: bool = False
    output_folder_format: str = "{date}_{time}_{title}"

    # ASR 원본/디버그 출력 보존 설정
    save_raw_asr: bool = True  # 가사 파싱 전 ASR 세그먼트 원본 JSON/TXT 보존
    save_asr_debug_outputs: bool = True  # WhisperX만으로 만든 타이밍 자막(srt/json) 추가 저장
    asr_debug_prefix: str = "whisperx_only"  # 디버그 파일 접두사

    # WhisperX 정밀(정렬) 자막 디버그 출력 설정
    save_whisperx_precise_debug: bool = True  # 정렬 완료 세그먼트 기반 자막 디버그 출력
    whisperx_precise_prefix_prevad: str = "whisperx_precise_prevad"   # VAD 보정 전 파일 접두사
    whisperx_precise_prefix_postvad: str = "whisperx_precise_postvad" # VAD 보정 후 파일 접두사

    # 보컬 분리(반주 제거) 설정
    vocal_separation: bool = False
    vocal_separation_engine: str = "demucs"  # 현재 demucs 지원
    vocal_separation_model: str = "htdemucs"  # demucs 모델명
    vocal_separation_save_stems: bool = True  # 분리된 보컬/반주 저장 여부

    # Whisper 세그먼트 분할/디코딩 설정
    whisper_condition_on_previous_text: bool = False
    whisper_temperature: float = 0.0
    whisper_max_segment_duration: float = 6.0
    whisper_split_on_punctuation: bool = True
    
    # ASR 엔진 선택 및 WhisperX 설정
    asr_engine: str = "whisper"  # "whisper" 또는 "whisperx"
    whisperx_batch_size: int = 16
    whisperx_compute_type: str = "float16"
    whisperx_language: Optional[str] = None
    whisperx_max_segment_duration: float = 6.0  # WhisperX 세그먼트 최대 길이(초), 초과 시 분할
    
    # VAD 및 무성구간 감지 설정
    enable_vad_timing_correction: bool = True
    vad_min_silence_len: float = 0.5
    vad_silence_thresh: float = -40.0

    # 시각화 디버깅 이미지 설정
    enable_visual_diagnostics: bool = True  # 처리 파이프라인 결과를 이미지로 저장
    diagnostics_image_filename: str = "diagnostics.png"
    visual_fig_width: int = 1600
    visual_fig_height: int = 2400
    visual_dpi: int = 150

class ConfigManager:
    """설정 파일 관리 클래스"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.logger = logger
        
    def load_config(self) -> EveryricConfig:
        """설정 파일을 로드합니다."""
        
        # 기본 설정으로 시작
        config = EveryricConfig()
        
        # 설정 파일이 존재하면 로드
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 설정 데이터를 config 객체에 적용
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                
                self.logger.info(f"✅ 설정 파일 로드 완료: {self.config_path}")
                
            except Exception as e:
                self.logger.warning(f"⚠️ 설정 파일 로드 실패: {str(e)} - 기본 설정 사용")
        else:
            self.logger.info(f"📄 설정 파일이 없습니다. 기본 설정으로 실행합니다.")
        
        # 환경변수에서 API 키 확인 (OpenAI)
        if not config.openai_api_key:
            env_key = os.getenv('OPENAI_API_KEY')
            if env_key:
                config.openai_api_key = env_key
                self.logger.info("🔑 환경변수에서 OpenAI API 키를 가져왔습니다.")

        # 환경변수에서 Google API 키 확인 (Gemini)
        if not getattr(config, 'google_api_key', ""):
            g_key = os.getenv('GOOGLE_API_KEY')
            if g_key:
                config.google_api_key = g_key
                self.logger.info("🔑 환경변수에서 Google API 키를 가져왔습니다.")
        
        return config
    
    def save_config(self, config: EveryricConfig) -> None:
        """설정을 파일로 저장합니다."""
        try:
            config_dict = asdict(config)
            
            # API 키는 보안상 저장하지 않음
            if 'openai_api_key' in config_dict:
                config_dict['openai_api_key'] = ""
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✅ 설정 파일 저장 완료: {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"❌ 설정 파일 저장 실패: {str(e)}")
    
    def create_sample_config(self) -> None:
        """샘플 설정 파일을 생성합니다."""
        sample_config = EveryricConfig(
            audio="https://youtu.be/example",
            lyrics="lyrics.txt",
            output="output",
            model="base",
            device="auto",
            max_length=80,
            pronunciation=True,
            source_lang="ja",
            translate=True,
            target_lang="ko",
            context="J-POP 발라드",
            log_level="INFO"
        )
        
        self.save_config(sample_config)
        self.logger.info(f"📝 샘플 설정 파일이 생성되었습니다: {self.config_path}")

def load_config(config_path: str = "config.json") -> EveryricConfig:
    """편의 함수: 설정 로드"""
    manager = ConfigManager(config_path)
    return manager.load_config()

def save_config(config: EveryricConfig, config_path: str = "config.json") -> None:
    """편의 함수: 설정 저장"""
    manager = ConfigManager(config_path)
    manager.save_config(config)

def create_sample_config(config_path: str = "config.json") -> None:
    """편의 함수: 샘플 설정 생성"""
    manager = ConfigManager(config_path)
    manager.create_sample_config()
