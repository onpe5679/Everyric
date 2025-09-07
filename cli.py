import argparse
import os
import sys
import logging
from datetime import datetime

# 로거 설정
from utils.logger import setup_logger, log_step

# 모듈 임포트
from audio.downloader import download_audio
from audio.processor import transcribe_audio
from text.lyrics import parse_lyrics
from align.smart_aligner import align_lyrics_with_whisper
from output.subtitle_formatter import save_aligned_subtitles

# 전역 로거 설정
logger = setup_logger("everyric")

def setup_argparse():
    """명령줄 인자 파서 설정"""
    parser = argparse.ArgumentParser(prog="everyric", description="🎵 Everyric - 음성 인식 텍스트 추출기 🎵")
    parser.add_argument(
        "--audio", 
        required=True, 
        help="오디오 파일 경로 또는 YouTube URL"
    )
    parser.add_argument(
        "--lyrics", 
        help="가사 파일 경로 (선택사항 - 제공시 정확한 타이밍 자막 생성)"
    )
    parser.add_argument(
        "--output", 
        default="output", 
        help="출력 디렉토리 (기본값: 'output')"
    )
    parser.add_argument(
        "--model", 
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper 모델 크기 (기본값: tiny)"
    )
    parser.add_argument(
        "--device", 
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="처리 장치 선택 (기본값: auto - GPU 사용 가능하면 GPU, 아니면 CPU)"
    )
    parser.add_argument(
        "--log-level", 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="로그 레벨 설정 (기본값: INFO)"
    )
    return parser.parse_args()

def main():
    # 명령줄 인자 파싱
    args = setup_argparse()
    
    # 로그 레벨 설정
    logger.setLevel(args.log_level)
    
    # 시작 로그
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info(f"🎵 Everyric 음성 인식 시작 - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("-" * 60)
    logger.info(f"🔊 오디오 소스: {args.audio}")
    logger.info(f"🤖 Whisper 모델: {args.model}")
    logger.info(f"⚙️ 처리 장치: {args.device}")
    logger.info(f"📝 가사 파일: {args.lyrics if args.lyrics else '없음 (음성 인식만)'}")
    logger.info(f"📂 출력 디렉토리: {os.path.abspath(args.output)}")
    logger.info("-" * 60)
    
    try:
        # 처리 파이프라인 (단순화)
        with log_step("오디오 다운로드/로드", logger):
            audio_path = download_audio(args.audio, args.output)
            logger.info(f"오디오 경로: {audio_path}")
            
        with log_step("음성 인식", logger):
            segments = transcribe_audio(audio_path, device=args.device, model_name=args.model)
            logger.info(f"인식된 음성 세그먼트 수: {len(segments)}")
            
        if args.lyrics:
            # 가사 파일이 제공된 경우 - 스마트 정렬 수행
            with log_step("가사 파일 로드", logger):
                lyrics_lines = parse_lyrics(args.lyrics)
                logger.info(f"로드된 가사 라인 수: {len(lyrics_lines)}")
                
            with log_step("가사-타이밍 스마트 정렬", logger):
                aligned_subtitles = align_lyrics_with_whisper(lyrics_lines, segments)
                logger.info(f"정렬된 자막 수: {len(aligned_subtitles)}")
                
            with log_step("자막 파일 생성", logger):
                output_files = save_aligned_subtitles(aligned_subtitles, args.output, ['srt', 'json'])
                logger.info(f"생성된 파일: {', '.join(output_files)}")
        else:
            # 가사 파일이 없는 경우 - 음성 인식 결과만 저장
            with log_step("음성 인식 결과 저장", logger):
                from output.formatter import save_transcription_text
                output_file = save_transcription_text(segments, args.output)
            
        # 완료 메시지
        duration = (datetime.now() - start_time).total_seconds()
        logger.info("-" * 60)
        logger.info(f"✅ 모든 처리가 완료되었습니다! (소요시간: {duration:.2f}초)")
        if args.lyrics:
            logger.info(f"💾 생성된 자막 파일들: {', '.join([os.path.abspath(f) for f in output_files])}")
        else:
            logger.info(f"💾 생성된 텍스트 파일: {os.path.abspath(output_file)}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.critical(f"❌ 처리 중 치명적 오류 발생: {str(e)}", exc_info=True)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("🚫 사용자에 의해 작업이 중단되었습니다.")
        sys.exit(130)  # SIGINT (Ctrl+C) 종료 코드

if __name__ == "__main__":
    main()
