import os
import sys
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def separate_vocals(
    input_path: str,
    output_dir: str,
    engine: str = "demucs",
    model: str = "htdemucs",
    save_stems: bool = True,
) -> Tuple[str, Optional[str]]:
    """
    반주 제거(보컬 분리)를 수행합니다. 기본 엔진은 Demucs입니다.

    Args:
        input_path: 원본 오디오 파일 경로
        output_dir: 결과를 저장할 기본 출력 디렉토리
        engine: 보컬 분리 엔진 (현재 'demucs'만 지원)
        model: demucs 모델명 (예: 'htdemucs')
        save_stems: 분리된 보컬/반주 파일을 함께 저장할지 여부

    Returns:
        (vocal_path, accompaniment_path)
        accompaniment_path는 없을 수 있습니다.
    """
    import tempfile
    import shutil
    
    os.makedirs(output_dir, exist_ok=True)

    if engine.lower() != "demucs":
        logger.warning(f"현재 지원하지 않는 보컬 분리 엔진입니다: {engine}. demucs로 대체합니다.")

    # 임시 디렉토리 사용 (경로 길이/한글 문제 회피)
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info("🎼 임시 작업 디렉토리에서 보컬 분리를 수행합니다")
        
        # 입력 파일을 임시 디렉토리에 복사 (간단한 이름으로)
        temp_input = os.path.join(temp_dir, "input.wav")
        
        # 입력이 wav가 아니면 ffmpeg로 변환
        if not str(input_path).lower().endswith(".wav"):
            cmd_conv = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-ac", "2",
                "-ar", "44100",
                temp_input,
            ]
            logger.info("🎼 입력을 WAV로 변환 중...")
            try:
                conv = subprocess.run(
                    cmd_conv,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                )
                if conv.returncode != 0:
                    logger.error(f"ffmpeg 변환 실패: {conv.stderr}")
                    raise RuntimeError("Audio conversion failed")
            except FileNotFoundError:
                logger.error("ffmpeg를 찾을 수 없습니다.")
                raise RuntimeError("ffmpeg not found")
        else:
            # WAV 파일이면 그대로 복사
            shutil.copy2(input_path, temp_input)

        # Demucs 실행 (임시 디렉토리에서)
        temp_stems = os.path.join(temp_dir, "stems")
        cmd = [
            sys.executable, "-m", "demucs",
            "-n", model,
            "--two-stems=vocals",
            "-o", temp_stems,
            temp_input,
        ]

        logger.info("🎤 보컬 분리(Demucs) 시작")
        logger.debug(f"Demucs 명령어: {' '.join(cmd)}")
        try:
            # torchaudio 백엔드 이슈(#570) 우회: 우선 soundfile 백엔드 사용 시도
            env = os.environ.copy()
            env["TORCHAUDIO_USE_SOUNDFILE"] = "1"
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                env=env,
            )
            logger.debug(f"Demucs 반환 코드: {result.returncode}")
            logger.debug(f"Demucs stdout: {result.stdout}")
            logger.debug(f"Demucs stderr: {result.stderr}")
            
            if result.returncode != 0:
                # 백엔드 저장 오류 시 sox_io로 재시도
                if "Couldn't find appropriate backend" in (result.stderr or ""):
                    logger.warning("torchaudio soundfile 백엔드 저장 실패. sox_io로 재시도합니다.")
                    env_retry = os.environ.copy()
                    env_retry["TORCHAUDIO_USE_SOUNDFILE"] = "0"
                    result_retry = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="ignore",
                        env=env_retry,
                    )
                    logger.debug(f"Demucs 재시도 반환 코드: {result_retry.returncode}")
                    logger.debug(f"Demucs 재시도 stdout: {result_retry.stdout}")
                    logger.debug(f"Demucs 재시도 stderr: {result_retry.stderr}")
                    if result_retry.returncode != 0:
                        logger.error("Demucs 재시도 실패")
                        logger.error(f"stderr: {result_retry.stderr}")
                        logger.error(f"stdout: {result_retry.stdout}")
                        logger.error("pip install soundfile 를 실행해 사운드파일 백엔드를 사용할 수 있도록 해주세요.")
                        raise RuntimeError("Demucs separation failed (both backends)")
                    else:
                        logger.info("✅ 보컬 분리 완료 (sox_io 백엔드)")
                else:
                    logger.error(f"Demucs 실행 실패 (코드 {result.returncode})")
                    logger.error(f"stderr: {result.stderr}")
                    logger.error(f"stdout: {result.stdout}")
                    raise RuntimeError(f"Demucs separation failed with code {result.returncode}")
            else:
                logger.info("✅ 보컬 분리 완료 (soundfile 백엔드)")
        except FileNotFoundError:
            logger.error("demucs 모듈을 찾을 수 없습니다. 'pip install demucs'로 설치해주세요.")
            raise

        # 결과 파일 찾기 및 복사
        model_dir = os.path.join(temp_stems, model)
        vocal_path = None
        accomp_path = None
        
        if os.path.isdir(model_dir):
            for subdir in os.listdir(model_dir):
                subdir_path = os.path.join(model_dir, subdir)
                if os.path.isdir(subdir_path):
                    for file in os.listdir(subdir_path):
                        if file.lower().endswith('.wav'):
                            if 'vocals' in file.lower() and 'no_vocals' not in file.lower():
                                vocal_path = os.path.join(subdir_path, file)
                            elif 'no_vocals' in file.lower() or 'accompaniment' in file.lower():
                                accomp_path = os.path.join(subdir_path, file)
                    break

        if not vocal_path:
            raise FileNotFoundError("보컬 스템 파일을 찾을 수 없습니다.")

        # 최종 출력 디렉토리에 복사
        final_vocals = os.path.join(output_dir, "audio_vocals.wav")
        shutil.copy2(vocal_path, final_vocals)
        logger.info(f"🎤 보컬 파일 저장: {final_vocals}")

        final_accomp = None
        if accomp_path:
            final_accomp = os.path.join(output_dir, "audio_no_vocals.wav")
            shutil.copy2(accomp_path, final_accomp)
            logger.info(f"🎵 반주 파일 저장: {final_accomp}")

        # stems 보존 옵션 처리
        if save_stems:
            stems_backup = os.path.join(output_dir, "stems")
            if os.path.exists(stems_backup):
                shutil.rmtree(stems_backup)
            shutil.copytree(temp_stems, stems_backup)
            logger.info(f"📁 stems 디렉토리 보존: {stems_backup}")

        return final_vocals, final_accomp
