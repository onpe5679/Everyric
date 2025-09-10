import argparse
import json
import os
import sys
from datetime import datetime
import shutil
from dataclasses import asdict
from pathlib import Path
import copy
from config.settings import EveryricConfig, load_config
from utils.logger import setup_logger, log_step
from utils.output_manager import create_dynamic_output_folder
from utils.output_manager import save_asr_output
from audio.downloader import download_audio
from audio.separator import separate_vocals
from audio.processor import transcribe_audio
from text.lyrics import parse_lyrics
from align.enhanced_aligner import SmartAligner
from output.subtitle_formatter import save_aligned_subtitles
from visualization.diagnostics import render_diagnostics_image

# 로거 설정
# 전역 로거 설정
logger = setup_logger("everyric")

def setup_argparse():
    """명령줄 인자 파서 설정 (단순화)"""
    parser = argparse.ArgumentParser(
        prog="everyric", 
        description="🎵 Everyric - 음성 인식 텍스트 추출기 🎵",
        epilog="설정 파일(config.json)을 사용하여 더 편리하게 실행할 수 있습니다."
    )
    
    # 필수 옵션만 유지
    parser.add_argument(
        "--config", 
        default="config.json",
        help="설정 파일 경로 (기본값: config.json)"
    )
    parser.add_argument(
        "--audio", 
        help="오디오 파일 경로 또는 YouTube URL (설정 파일보다 우선)"
    )
    parser.add_argument(
        "--lyrics", 
        help="가사 파일 경로 (설정 파일보다 우선)"
    )
    parser.add_argument(
        "--create-config", 
        action="store_true",
        help="샘플 설정 파일 생성 후 종료"
    )
    
    return parser.parse_args()

def main():
    # 명령줄 인자 파싱
    args = setup_argparse()
    
    # 샘플 설정 파일 생성 요청
    if args.create_config:
        create_sample_config(args.config)
        print(f"✅ 샘플 설정 파일이 생성되었습니다: {args.config}")
        print("📝 설정 파일을 수정한 후 다시 실행해주세요.")
        return
    
    # 설정 파일 로드
    config = load_config(args.config)
    
    # 명령줄 인자로 설정 덮어쓰기
    if args.audio:
        config.audio = args.audio
    if args.lyrics:
        config.lyrics = args.lyrics
    
    # 필수 설정 확인
    if not config.audio:
        logger.error("❌ 오디오 소스가 지정되지 않았습니다.")
        logger.info("💡 다음 중 하나를 사용하세요:")
        logger.info("   1. python cli.py --audio 'URL_또는_파일경로'")
        logger.info("   2. config.json 파일에서 audio 설정")
        logger.info("   3. python cli.py --create-config (샘플 설정 파일 생성)")
        sys.exit(1)
    
    # 로그 레벨 설정
    logger.setLevel(config.log_level)
    
    # 서브프로세스 출력 인코딩 강제 (Windows cp949 디코딩 오류 완화)
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

    # 출력 소음 제어 (진행바/경고 억제)
    try:
        if getattr(config, 'suppress_progress_bars', True):
            os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
            os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
            os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        if getattr(config, 'suppress_warnings', True):
            import warnings as _warnings
            # 자주 보이는 경고 억제 패턴
            _warnings.filterwarnings('ignore', message=r"Passing `gradient_checkpointing`.*", category=UserWarning)
            _warnings.filterwarnings('ignore', message=r"Lightning automatically upgraded your loaded checkpoint.*", category=UserWarning)
            _warnings.filterwarnings('ignore', category=DeprecationWarning)
            _warnings.filterwarnings('ignore', module=r"torch|torchaudio|transformers|librosa|pydub")
            # Transformers/PyTorch/Lightning 로거 레벨 하향
            import logging as _logging
            for name in ['transformers', 'pytorch_lightning', 'lightning', 'pyannote.audio', 'torch', 'torchaudio', 'librosa']:
                try:
                    _logging.getLogger(name).setLevel(_logging.ERROR)
                except Exception:
                    pass
            try:
                from transformers import logging as tf_logging
                tf_logging.set_verbosity_error()
            except Exception:
                pass
    except Exception:
        pass

    # 시작 로그
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info(f"🎵 Everyric 음성 인식 시작 - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("-" * 60)
    logger.info(f"📄 설정 파일: {args.config}")
    logger.info(f"🔊 오디오 소스: {config.audio}")
    logger.info(f"🤖 Whisper 모델: {config.model}")
    logger.info(f"⚙️ 처리 장치: {config.device}")
    logger.info(f"📝 가사 파일: {config.lyrics if config.lyrics else '없음 (음성 인식만)'}")
    logger.info(f"📏 최대 세그먼트 길이: {config.max_length}자")
    logger.info(f"🗣️ 한글 발음 표기: {'포함' if config.pronunciation else '미포함'}")
    logger.info(f"🌐 원본 언어: {config.source_lang}")
    logger.info(f"🌍 번역 기능: {'포함' if config.translate else '미포함'}")
    if config.translate:
        logger.info(f"🎯 번역 목표 언어: {config.target_lang}")
        logger.info(f"📖 번역 컨텍스트: {config.context if config.context else '없음'}")
        logger.info(f"🔑 API 키: {'설정됨' if config.openai_api_key else '없음'}")
    # 보컬 분리 설정 로그
    try:
        v_on = getattr(config, 'vocal_separation', False)
        logger.info(f"🎛️ 반주 제거(보컬 분리): {'활성' if v_on else '비활성'}")
        if v_on:
            logger.info(f"   • 엔진: {getattr(config, 'vocal_separation_engine', 'demucs')} | 모델: {getattr(config, 'vocal_separation_model', 'htdemucs')} | stems 보존: {getattr(config, 'vocal_separation_save_stems', True)}")
    except Exception:
        pass
    logger.info(f"📂 출력 디렉토리: {os.path.abspath(config.output)}")
    logger.info("-" * 60)
    
    try:
        # 동적 출력 폴더 생성
        from utils.output_manager import create_dynamic_output_folder, get_enhanced_youtube_title
        
        dynamic_enabled = getattr(config, 'dynamic_output_folder', False)
        folder_format = getattr(config, 'output_folder_format', '{date}_{time}_{title}')
        
        if dynamic_enabled and config.audio:
            # YouTube URL인 경우 실제 제목 추출 시도
            if config.audio.startswith(("http://", "https://")):
                logger.info("🎵 YouTube 제목 추출 중...")
                actual_title = get_enhanced_youtube_title(config.audio)
                # 폴더 형식에서 {title}을 실제 제목으로 교체
                folder_format = folder_format.replace('{title}', actual_title)
        
        final_output_dir = create_dynamic_output_folder(
            config.output, 
            config.audio, 
            dynamic_enabled, 
            folder_format
        )
        
        # 출력 폴더에 입력 가사 사본 저장
        try:
            if getattr(config, 'lyrics', None) and os.path.exists(config.lyrics):
                lyrics_dest = os.path.join(final_output_dir, 'lyrics.txt')
                shutil.copy2(config.lyrics, lyrics_dest)
                logger.info(f"📝 가사 사본 저장: {lyrics_dest}")
        except Exception as e:
            logger.warning(f"가사 사본 저장 실패: {e}")
        
        # 출력 폴더에 API 키 제거된 config 스냅샷 저장
        try:
            cfg = asdict(config) if hasattr(config, '__dataclass_fields__') else dict(vars(config))
            # 민감 정보 제거
            for secret_key in ['openai_api_key', 'google_api_key']:
                if secret_key in cfg:
                    cfg[secret_key] = ''
            config_snapshot_path = os.path.join(final_output_dir, 'config.json')
            with open(config_snapshot_path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            logger.info(f"🛠️ 디버깅용 설정 스냅샷 저장(키 제거): {config_snapshot_path}")
        except Exception as e:
            logger.warning(f"설정 스냅샷 저장 실패: {e}")
        
        # 처리 파이프라인 (설정 기반)
        with log_step("오디오 다운로드/로드", logger):
            # 오디오 보존 설정 적용
            save_audio = getattr(config, 'save_audio', False)
            audio_format = getattr(config, 'audio_format', 'wav')
            # 로거 핸들러 임시 분리
            handlers = list(logger.handlers)
            for handler in handlers:
                logger.removeHandler(handler)
            
            audio_path = download_audio(config.audio, final_output_dir, save_audio, audio_format)
            original_audio_path = audio_path  # 시각화용 원본 오디오 경로 보존
            
            # 로거 핸들러 다시 추가
            for handler in handlers:
                logger.addHandler(handler)
            logger.info(f"오디오 경로: {audio_path}")
            if save_audio:
                logger.info(f"💾 오디오 파일이 {audio_format} 형식으로 보존됩니다")
        
        # 보컬 분리(반주 제거): Whisper 전에 수행
        v_on_pipeline = getattr(config, 'vocal_separation', False)
        vocals_audio_path = None  # 시각화용 보컬 오디오 경로
        accompaniment_audio_path = None  # 시각화용 반주 오디오 경로
        logger.info(f"🎛️ 반주 제거 단계 진입 여부: {'예' if v_on_pipeline else '아니오'}")
        if v_on_pipeline:
            with log_step("반주 제거(보컬 분리)", logger):
                try:
                    engine = getattr(config, 'vocal_separation_engine', 'demucs')
                    model = getattr(config, 'vocal_separation_model', 'htdemucs')
                    save_stems = getattr(config, 'vocal_separation_save_stems', True)
                    # 로거 핸들러 임시 분리
                    handlers = list(logger.handlers)
                    for handler in handlers:
                        logger.removeHandler(handler)
                        
                    vocals_path, accomp_path = separate_vocals(
                        input_path=audio_path,
                        output_dir=final_output_dir,
                        engine=engine,
                        model=model,
                        save_stems=save_stems,
                    )
                    
                    # 로거 핸들러 다시 추가
                    for handler in handlers:
                        logger.addHandler(handler)
                    logger.info(f"🎤 보컬 파일: {vocals_path}")
                    if accomp_path:
                        logger.info(f"🎵 반주 파일: {accomp_path}")
                        accompaniment_audio_path = accomp_path
                    # Whisper 입력을 보컬로 교체
                    audio_path = vocals_path
                    vocals_audio_path = vocals_path
                except Exception as e:
                    logger.error(f"반주 제거 실패, 원본 오디오로 계속 진행합니다: {e}")
        else:
            logger.info("🎚️ 반주 제거 비활성: 이 단계를 건너뜁니다")
            
        # ASR 엔진 선택 및 실행
        asr_engine = getattr(config, 'asr_engine', 'whisper')
        
        if asr_engine == "whisperx":
            with log_step("WhisperX 음성 인식", logger):
                try:
                    from audio.whisperx_processor import transcribe_audio_with_whisperx, detect_silence_segments, apply_silence_based_timing_correction
                    
                    # whisperx_max_segment_duration은 문자열로 들어올 수 있으므로 float로 보정
                    _wx_maxdur_raw = getattr(config, 'whisperx_max_segment_duration', 6.0)
                    try:
                        _wx_maxdur = float(_wx_maxdur_raw) if _wx_maxdur_raw is not None else 6.0
                    except Exception:
                        _wx_maxdur = 6.0

                    segments = transcribe_audio_with_whisperx(
                        audio_path,
                        device=config.device,
                        model_name=config.model,
                        batch_size=getattr(config, 'whisperx_batch_size', 16),
                        compute_type=getattr(config, 'whisperx_compute_type', 'float16'),
                        language=getattr(config, 'whisperx_language', None),
                        max_segment_duration=_wx_maxdur,
                    )
                    # VAD 보정 전 원본 세그먼트 백업 (단어 타임라인 포함)
                    segments_prevad = copy.deepcopy(segments)
                    # 정렬(precise) 결과 디버그 저장 - VAD 보정 전 (단어 단위)
                    if getattr(config, 'save_whisperx_precise_debug', True):
                        try:
                            from utils.word_level_formatter import save_word_level_subtitles
                            prefix_pre = getattr(config, 'whisperx_precise_prefix_prevad', 'whisperx_precise_prevad')
                            files_pre = save_word_level_subtitles(
                                segments,
                                final_output_dir,
                                filename_prefix=prefix_pre
                            )
                            logger.info(f"🧪 WhisperX 정밀 자막(보정 전, 단어단위) 저장: {', '.join(files_pre)}")
                        except Exception as e:
                            logger.warning(f"WhisperX 정밀(보정 전) 저장 실패: {e}")
                    
                    # VAD 기반 타이밍 보정 (선택적)
                    # 시각화용 무성 구간 저장 변수
                    silence_ranges_for_viz = None

                    if getattr(config, 'enable_vad_timing_correction', True):
                        silence_ranges = detect_silence_segments(
                            audio_path,
                            min_silence_len=getattr(config, 'vad_min_silence_len', 0.5),
                            silence_thresh=getattr(config, 'vad_silence_thresh', -40.0)
                        )
                        silence_ranges_for_viz = silence_ranges
                        if silence_ranges:
                            segments = apply_silence_based_timing_correction(segments, silence_ranges)
                            logger.info("✅ VAD 기반 타이밍 보정 적용 완료")
                            # 정렬(precise) 결과 디버그 저장 - VAD 보정 후 (단어 단위)
                            if getattr(config, 'save_whisperx_precise_debug', True):
                                try:
                                    from utils.word_level_formatter import save_word_level_subtitles
                                    prefix_post = getattr(config, 'whisperx_precise_prefix_postvad', 'whisperx_precise_postvad')
                                    files_post = save_word_level_subtitles(
                                        segments,
                                        final_output_dir,
                                        filename_prefix=prefix_post
                                    )
                                    logger.info(f"🧪 WhisperX 정밀 자막(보정 후, 단어단위) 저장: {', '.join(files_post)}")
                                except Exception as e:
                                    logger.warning(f"WhisperX 정밀(보정 후) 저장 실패: {e}")
                    segments_postvad = segments
                
                except ImportError:
                    logger.warning("WhisperX를 사용할 수 없습니다. 기본 Whisper로 폴백합니다.")
                    logger.info("WhisperX 설치: pip install whisperx")
                    asr_engine = "whisper"
        
        if asr_engine == "whisper":
            with log_step("Whisper 음성 인식", logger):
                segments = transcribe_audio(
                    audio_path,
                    device=config.device,
                    model_name=config.model,
                    condition_on_previous_text=getattr(config, 'whisper_condition_on_previous_text', False),
                    temperature=getattr(config, 'whisper_temperature', 0.0),
                    max_segment_duration=getattr(config, 'whisper_max_segment_duration', 6.0),
                    split_on_punctuation=getattr(config, 'whisper_split_on_punctuation', True),
                )

        logger.info(f"인식된 음성 세그먼트 수: {len(segments)}")

        # 가사 파싱 전에 원본 ASR 결과 보존 (설정 기반)
        if getattr(config, 'save_raw_asr', True):
            try:
                saved = save_asr_output(segments, final_output_dir, asr_engine)
                logger.info(f"🗄️ ASR 원본 보존 완료: {saved}")
            except Exception as e:
                logger.warning(f"ASR 원본 보존 실패: {e}")

        # WhisperX만으로 만든 타이밍 자막(SRT/JSON) 디버그 출력 (원본 가사 미사용)
        if asr_engine == "whisperx" and getattr(config, 'save_asr_debug_outputs', True):
            try:
                debug_prefix = getattr(config, 'asr_debug_prefix', 'whisperx_only')
                debug_files = save_aligned_subtitles(
                    segments,  # 이미 start/end/text 구조
                    final_output_dir,
                    formats=['srt', 'json'],
                    include_pronunciation=False,
                    source_lang=config.source_lang,
                    translated_lyrics=None,
                    filename_prefix=debug_prefix
                )
                logger.info(f"🧪 WhisperX 디버그 자막 저장: {', '.join(debug_files)}")
            except Exception as e:
                logger.warning(f"WhisperX 디버그 자막 저장 실패: {e}")
            
        if config.lyrics:
            # 가사 파일이 제공된 경우 - 스마트 정렬 수행
            with log_step("가사 파일 로드", logger):
                lyrics_lines = parse_lyrics(config.lyrics)
                logger.info(f"로드된 가사 라인 수: {len(lyrics_lines)}")
                
            with log_step("가사-타이밍 정렬", logger):
                alignment_config = getattr(config, 'alignment', {})
                # LLM 조각 최대 길이 전달
                try:
                    _cfg_copy = dict(alignment_config)
                except Exception:
                    _cfg_copy = {}
                _cfg_copy['max_length'] = getattr(config, 'max_length', 100)
                alignment_config = _cfg_copy
                alignment_engine = getattr(config, 'alignment_engine', 'dtw')
                # 무성구간: WhisperX 단계에서 계산된 값이 없으면 필요 시 재계산
                _silences_for_align = locals().get('silence_ranges_for_viz', None)
                if _silences_for_align is None and getattr(config, 'enable_vad_timing_correction', True):
                    try:
                        from audio.whisperx_processor import detect_silence_segments as _detect_sil
                        _silences_for_align = _detect_sil(
                            audio_path,
                            min_silence_len=getattr(config, 'vad_min_silence_len', 0.5),
                            silence_thresh=getattr(config, 'vad_silence_thresh', -40.0)
                        )
                    except Exception:
                        _silences_for_align = None

                if alignment_engine == 'llm':
                    logger.info(f"🤖 LLM 전용 정렬 모드 시작 (엔진: {getattr(config, 'llm_alignment_engine', 'gpt')})")
                    try:
                        from align.llm_only_aligner import LLMOnlyAligner
                        llm_engine = getattr(config, 'llm_alignment_engine', 'gpt')
                        aligner = LLMOnlyAligner(
                            engine=llm_engine,
                            openai_api_key=getattr(config, 'openai_api_key', None),
                            google_api_key=getattr(config, 'google_api_key', None),
                            alignment_config=alignment_config,
                        )
                        aligned_subtitles = aligner.align_lyrics_with_timing(lyrics_lines, segments, _silences_for_align)
                        
                        # LLM 정렬기의 유사도 정보 저장 (diagnostics용)
                        if hasattr(aligner, '_last_similarity_info'):
                            _llm_similarity_info = aligner._last_similarity_info
                            logger.info(f"📊 LLM 유사도 정보 저장됨: {_llm_similarity_info}")
                    except Exception as e:
                        logger.error(f"LLM 전용 정렬 중 오류: {e}")
                        aligned_subtitles = []
                else:
                    logger.info("🧠 DTW 기반 정렬 모드 시작")
                    aligner = SmartAligner(max_segment_length=config.max_length, alignment_config=alignment_config)
                    aligned_subtitles = aligner.align_lyrics_with_timing(lyrics_lines, segments, _silences_for_align)
                logger.info(f"정렬된 자막 수: {len(aligned_subtitles)}")
            
            # 발음 표기 추가 (LLM 기반)
            if config.pronunciation and aligned_subtitles:
                with log_step("발음 표기 추가", logger):
                    try:
                        from text.llm_pronunciation import add_pronunciation_to_subtitles
                        # LLM 엔진은 정렬에 사용한 것과 동일하게
                        pronunciation_engine = getattr(config, 'llm_alignment_engine', 'gemini')
                        logger.info(f"🗣️ LLM 발음 변환 시작 - 엔진: {pronunciation_engine}")
                        
                        # API 키 확인 및 전달
                        openai_key = getattr(config, 'openai_api_key', None)
                        google_key = getattr(config, 'google_api_key', None)
                        
                        # LLMPronunciationConverter 직접 사용
                        from text.llm_pronunciation import LLMPronunciationConverter
                        converter = LLMPronunciationConverter(
                            engine=pronunciation_engine,
                            openai_api_key=openai_key,
                            google_api_key=google_key
                        )
                        aligned_subtitles = converter.add_pronunciation_to_subtitles(
                            aligned_subtitles, 
                            config.source_lang
                        )
                        logger.info("✅ LLM 발음 표기 추가 완료")
                    except Exception as e:
                        logger.warning(f"LLM 발음 표기 추가 실패: {e}")
                        # 기존 모듈로 폴백
                        try:
                            from text.pronunciation import add_pronunciation_to_subtitles as fallback_pronunciation
                            aligned_subtitles = fallback_pronunciation(aligned_subtitles, config.source_lang)
                            logger.info("✅ 기존 발음 표기로 폴백 완료")
                        except Exception as e2:
                            logger.warning(f"발음 표기 폴백도 실패: {e2}")

            # 번역 처리
            translated_lyrics = None
            if config.translate and aligned_subtitles:
                with log_step("가사 번역", logger):
                    try:
                        # 가사 텍스트만 추출
                        lyrics_text = [item['text'] for item in aligned_subtitles if item['text'].strip()]
                        
                        # 번역 엔진 선택
                        translation_engine = getattr(config, 'translation_engine', 'gemini')
                        
                        if translation_engine == 'gemini':
                            from translation.translator import translate_lyrics_with_gemini
                            from translation.translator import translate_lyrics_with_gpt
                            
                            # Google API 키 설정 (환경변수 또는 config에서)
                            google_api_key = getattr(config, 'google_api_key', None) or os.getenv('GOOGLE_API_KEY')
                            
                            if not google_api_key:
                                logger.warning("Gemini API 키가 없어 GPT로 폴백합니다.")
                                translated_lyrics = translate_lyrics_with_gpt(
                                    lyrics_text,
                                    target_lang=config.target_lang,
                                    source_lang=config.source_lang,
                                    context=config.context,
                                    api_key=config.openai_api_key,
                                    model=config.gpt_model,
                                    max_retries=config.gpt_max_retries,
                                    retry_delay=config.gpt_retry_delay
                                )
                                translation_engine = 'gpt'
                            else:
                                try:
                                    translated_lyrics = translate_lyrics_with_gemini(
                                        lyrics_text,
                                        target_lang=config.target_lang,
                                        source_lang=config.source_lang,
                                        context=config.context,
                                        api_key=google_api_key,
                                        model=getattr(config, 'gemini_model', 'gemini-1.5-flash'),
                                        max_retries=getattr(config, 'gemini_max_retries', 5),
                                        retry_delay=getattr(config, 'gemini_retry_delay', 2)
                                    )
                                except Exception as gemini_error:
                                    logger.warning(f"Gemini 번역 실패로 GPT로 폴백합니다: {gemini_error}")
                                    translated_lyrics = translate_lyrics_with_gpt(
                                        lyrics_text,
                                        target_lang=config.target_lang,
                                        source_lang=config.source_lang,
                                        context=config.context,
                                        api_key=config.openai_api_key,
                                        model=config.gpt_model,
                                        max_retries=config.gpt_max_retries,
                                        retry_delay=config.gpt_retry_delay
                                    )
                                    translation_engine = 'gpt'
                        else:  # GPT 번역
                            from translation.translator import translate_lyrics_with_gpt
                            
                            translated_lyrics = translate_lyrics_with_gpt(
                                lyrics_text,
                                target_lang=config.target_lang,
                                source_lang=config.source_lang,
                                context=config.context,
                                api_key=config.openai_api_key,
                                model=config.gpt_model,
                                max_retries=config.gpt_max_retries,
                                retry_delay=config.gpt_retry_delay
                            )
                        
                        logger.info(f"번역 완료: {len(translated_lyrics)}줄 (엔진: {translation_engine})")
                    except Exception as e:
                        logger.error(f"번역 실패: {str(e)}")
                        logger.warning("번역 없이 계속 진행합니다.")
                
            with log_step("자막 파일 생성", logger):
                output_files = save_aligned_subtitles(
                    aligned_subtitles, 
                    final_output_dir, 
                    include_pronunciation=config.pronunciation,
                    translated_lyrics=translated_lyrics
                )
                logger.info(f"생성된 파일: {', '.join(output_files)}")
        else:
            # 가사 파일이 없는 경우 - 음성 인식 결과만 저장
            with log_step("음성 인식 결과 저장", logger):
                from output.formatter import save_transcription_text
                output_file = save_transcription_text(segments, final_output_dir)
            
        # 시각화 디버그 이미지 생성
        if getattr(config, 'enable_visual_diagnostics', True):
            try:
                _aligned = aligned_subtitles if config.lyrics else None
                _lyrics = lyrics_lines if config.lyrics else None
                _silences = locals().get('silence_ranges_for_viz', None)
                render_diagnostics_image(
                    output_dir=final_output_dir,
                    original_audio_path=original_audio_path,
                    vocals_audio_path=vocals_audio_path,
                    total_duration_hint=None,
                    aligned_subtitles=_aligned,
                    whisperx_segments_postvad=locals().get('segments_postvad'),
                    whisperx_segments_prevad=locals().get('segments_prevad'),
                    lyrics_lines=_lyrics,
                    silence_ranges=_silences,
                    similarity_info=locals().get('_llm_similarity_info'),
                    image_filename=getattr(config, 'diagnostics_image_filename', 'diagnostics.png'),
                    fig_width=getattr(config, 'visual_fig_width', 1600),
                    fig_height=getattr(config, 'visual_fig_height', 2400),
                    dpi=getattr(config, 'visual_dpi', 150),
                )
            except Exception as e:
                logger.warning(f"진단 이미지 생성 실패: {e}")

        # 완료 메시지
        duration = (datetime.now() - start_time).total_seconds()
        logger.info("-" * 60)
        logger.info(f"✅ 모든 처리가 완료되었습니다! (소요시간: {duration:.2f}초)")
        if config.lyrics:
            logger.info(f"💾 생성된 자막 파일들: {', '.join(output_files)}")
        else:
            logger.info(f"💾 생성된 텍스트 파일: {output_file}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.critical(f"❌ 처리 중 치명적 오류 발생: {str(e)}", exc_info=True)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("🚫 사용자에 의해 작업이 중단되었습니다.")
        sys.exit(130)  # SIGINT (Ctrl+C) 종료 코드

if __name__ == "__main__":
    main()
