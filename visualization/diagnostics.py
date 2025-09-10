import os
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경에서 PNG 저장
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager, rcParams

try:
    import librosa
    import librosa.display  # noqa: F401
except Exception:
    librosa = None

logger = logging.getLogger(__name__)

_CJK_FONT_CANDIDATES = [
    # Prefer broadly covering Noto CJK if installed
    'Noto Sans CJK KR', 'Noto Sans CJK JP', 'Noto Sans CJK SC', 'Noto Sans CJK TC',
    'Noto Sans KR', 'Noto Sans JP', 'Noto Sans SC', 'Noto Sans TC',
    # Windows defaults
    'Malgun Gothic',         # Korean
    'Yu Gothic UI', 'Yu Gothic', 'Meiryo', 'MS Gothic',  # Japanese
    # Common fallback
    'Arial Unicode MS', 'DejaVu Sans'
]

_SELECTED_FONT = None

def _configure_matplotlib_fonts():
    global _SELECTED_FONT
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
        selected = None
        # Build a prioritized font family list composed of available candidates
        family_list = []
        for cand in _CJK_FONT_CANDIDATES:
            if cand in available and cand not in family_list:
                family_list.append(cand)
                if selected is None:
                    selected = cand

        # Always append a couple of generic fallbacks at the end
        family_list += ['DejaVu Sans', 'Arial', 'sans-serif']

        if family_list:
            rcParams['font.family'] = family_list
            rcParams['font.sans-serif'] = family_list
            _SELECTED_FONT = selected
        # 유니코드 마이너스 기호 깨짐 방지
        rcParams['axes.unicode_minus'] = False
        if _SELECTED_FONT:
            logger.info(f"📚 Matplotlib 폰트 설정: {_SELECTED_FONT}")
        else:
            logger.warning("CJK 폰트를 찾지 못했습니다. 시스템 기본 폰트를 사용합니다.")
    except Exception as e:
        logger.warning(f"Matplotlib 폰트 설정 실패: {e}")

_configure_matplotlib_fonts()


def _load_rms_envelope(audio_path: str, frame_length: int = 2048, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    오디오에서 RMS(진폭) 엔벨로프와 시간축을 계산합니다.
    Returns: times(s), rms(0~1), duration(s)
    """
    if librosa is None:
        raise RuntimeError("librosa가 설치되어 있지 않습니다. requirements.txt를 설치해주세요.")

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    # 정규화
    if np.max(rms) > 0:
        rms = rms / np.max(rms)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return times, rms, duration


def _draw_time_ticks(ax: plt.Axes, total_duration: float, tick_sec: float) -> None:
    ax.set_ylim(total_duration, 0)  # 위->아래로 시간 증가
    ticks = np.arange(0, total_duration + tick_sec, tick_sec)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{int(t//60):02d}:{int(t%60):02d}" for t in ticks])
    ax.grid(axis='y', linestyle=':', alpha=0.3)


def _draw_bar_from_rms(ax: plt.Axes, times: np.ndarray, rms: np.ndarray, total_duration: float,
                       title: str, silence_ranges: Optional[List[Tuple[float, float]]] = None,
                       cmap: str = 'Greys') -> None:
    # RMS를 세로 바 이미지로 변환 (time x 1 이미지)
    h = len(times)
    img = rms.reshape(h, 1)
    ax.imshow(img, aspect='auto', origin='upper', cmap=cmap,
              extent=[0, 1, total_duration, 0], vmin=0.0, vmax=1.0)
    _draw_time_ticks(ax, total_duration, tick_sec=10)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title(title)

    # 무성 구간 표시 (빨간 반투명 박스)
    if silence_ranges:
        for s, e in silence_ranges:
            if e <= 0 or e <= s:
                continue
            rect = patches.Rectangle((0, s), 1, e - s, linewidth=0, edgecolor=None,
                                     facecolor=(1, 0, 0, 0.25))
            ax.add_patch(rect)


def _draw_segments_column(ax: plt.Axes, segments: List[Dict], total_duration: float,
                          title: str, color: str) -> None:
    ax.set_ylim(total_duration, 0)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    _draw_time_ticks(ax, total_duration, tick_sec=10)
    ax.set_title(title)

    for seg in segments or []:
        start = float(seg.get('start', 0.0))
        end = float(seg.get('end', 0.0))
        text = (seg.get('text') or '').strip()
        if end <= start:
            continue
        rect = patches.Rectangle((0.05, start), 0.90, end - start, linewidth=0,
                                 edgecolor=None, facecolor=color, alpha=0.35)
        ax.add_patch(rect)
        mid = (start + end) / 2
        if text:
            ax.text(0.5, mid, text, ha='center', va='center', fontsize=8,
                    color='black', alpha=0.9, wrap=True, fontfamily=_SELECTED_FONT or rcParams.get('font.family'))


def _draw_lyrics_list(ax: plt.Axes, lyrics_lines: List[str], title: str) -> None:
    ax.axis('off')
    ax.set_title(title)
    if not lyrics_lines:
        return
    # 위에서 아래로 나열
    y = 0.98
    for i, line in enumerate(lyrics_lines):
        if not line.strip():
            y -= 0.02
            continue
        ax.text(0.0, y, line.strip(), fontsize=10, va='top', ha='left', transform=ax.transAxes,
                fontfamily=_SELECTED_FONT or rcParams.get('font.family'))
        y -= 0.05
        if y < 0.02:
            break


def render_diagnostics_image(
    output_dir: str,
    original_audio_path: str,
    vocals_audio_path: Optional[str] = None,
    total_duration_hint: Optional[float] = None,
    aligned_subtitles: Optional[List[Dict]] = None,
    whisperx_segments_postvad: Optional[List[Dict]] = None,
    whisperx_segments_prevad: Optional[List[Dict]] = None,
    lyrics_lines: Optional[List[str]] = None,
    silence_ranges: Optional[List[Tuple[float, float]]] = None,
    similarity_info: Optional[Dict] = None,
    image_filename: str = 'diagnostics.png',
    fig_width: int = 1600,
    fig_height: int = 2400,
    dpi: int = 150,
) -> str:
    """
    단계별 결과를 한 장의 이미지로 시각화하여 저장합니다.
    좌측(수직 타임라인): 원본/보컬 RMS 바 + 무성구간(빨간 박스)
    중앙/우측: [최종가사, WhisperX(보정후), WhisperX(보정전), WhisperX(단어단위), 원본가사]
    """
    os.makedirs(output_dir, exist_ok=True)

    # 오디오 로딩 및 RMS 계산
    try:
        times_o, rms_o, dur_o = _load_rms_envelope(original_audio_path)
    except Exception as e:
        logger.warning(f"원본 오디오 로딩 실패: {e}")
        # output_dir/audio.wav 대체 시도
        try:
            fallback = os.path.join(output_dir, 'audio.wav')
            if os.path.exists(fallback):
                times_o, rms_o, dur_o = _load_rms_envelope(fallback)
                logger.info("원본 오디오 대체: audio.wav 로딩 성공")
            else:
                times_o, rms_o, dur_o = np.linspace(0, total_duration_hint or 0, 2), np.zeros(2), (total_duration_hint or 0)
        except Exception as e2:
            logger.warning(f"fallback audio.wav 로딩 실패: {e2}")
            times_o, rms_o, dur_o = np.linspace(0, total_duration_hint or 0, 2), np.zeros(2), (total_duration_hint or 0)

    # 보조 바는 '보컬' 고정 표기 (요청사항)
    times_voc, rms_voc, dur_voc = None, None, None
    if vocals_audio_path:
        try:
            times_voc, rms_voc, dur_voc = _load_rms_envelope(vocals_audio_path)
        except Exception as e:
            logger.warning(f"보컬 오디오 로딩 실패: {e}")

    total_duration = max(filter(lambda x: x is not None, [dur_o, dur_voc, total_duration_hint or 0.0]))
    if total_duration <= 0:
        # 세그먼트 기반 추정
        max_end = 0.0
        for coll in [aligned_subtitles or [], whisperx_segments_postvad or [], whisperx_segments_prevad or []]:
            for seg in coll:
                max_end = max(max_end, float(seg.get('end', 0.0)))
        total_duration = max_end

    # Figure 생성
    fig_w_in = fig_width / dpi
    fig_h_in = fig_height / dpi
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)

    # 그리드: 좌측 2개 바(원본/보컬) + 5개 컬럼 = 총 7열
    widths = [0.6, 0.6, 3.5, 3.5, 3.5, 3.5, 3.0]
    gs = fig.add_gridspec(ncols=len(widths), nrows=1, width_ratios=widths, wspace=0.2)

    ax_o = fig.add_subplot(gs[0, 0])  # 원본 바
    _draw_bar_from_rms(ax_o, times_o, rms_o, total_duration, title='원본', silence_ranges=silence_ranges, cmap='Greys')

    ax_v = fig.add_subplot(gs[0, 1])  # 보컬 바
    if times_voc is not None and rms_voc is not None and dur_voc:
        _draw_bar_from_rms(ax_v, times_voc, rms_voc, total_duration, title='보컬', silence_ranges=None, cmap='Greys')
    else:
        ax_v.axis('off')
        ax_v.set_title('보컬(없음)')

    ax_final = fig.add_subplot(gs[0, 2])
    _draw_segments_column(ax_final, aligned_subtitles or [], total_duration, title='최종가사', color='tab:green')

    ax_wx_post = fig.add_subplot(gs[0, 3])
    _draw_segments_column(ax_wx_post, whisperx_segments_postvad or [], total_duration, title='WhisperX(보정후)', color='tab:blue')

    ax_wx_pre = fig.add_subplot(gs[0, 4])
    _draw_segments_column(ax_wx_pre, whisperx_segments_prevad or [], total_duration, title='WhisperX(보정전)', color='tab:orange')

    # 단어 단위: prevad에서 words를 추출하여 단어별 구간으로 표시
    def _extract_word_segments(segs: Optional[List[Dict]]) -> List[Dict]:
        words_as_segments: List[Dict] = []
        if not segs:
            return words_as_segments
        for seg in segs:
            for w in seg.get('words', []) or []:
                ws = float(w.get('start', seg.get('start', 0.0)))
                we = float(w.get('end', ws))
                if we <= ws:
                    continue
                words_as_segments.append({'start': ws, 'end': we, 'text': (w.get('word') or '').strip()})
        return words_as_segments

    ax_wx_words = fig.add_subplot(gs[0, 5])
    _draw_segments_column(ax_wx_words, _extract_word_segments(whisperx_segments_prevad), total_duration, title='WhisperX(단어)', color='tab:purple')

    ax_lyrics = fig.add_subplot(gs[0, 6])
    _draw_lyrics_list(ax_lyrics, lyrics_lines or [], title='원본가사')

    # 유사도 정보 박스 렌더링 (우측 상단 여백에 표기)
    if similarity_info:
        try:
            overall = similarity_info.get('overall', None)
            avg = similarity_info.get('average', None)
            indiv = similarity_info.get('individual', []) or []
            if indiv:
                min_sim = min(indiv)
                max_sim = max(indiv)
            else:
                min_sim = max_sim = None

            lines = ["Similarity Stats"]
            if overall is not None:
                lines.append(f"Overall: {overall:.1f}%")
            if avg is not None:
                lines.append(f"Average: {avg:.1f}%")
            if min_sim is not None and max_sim is not None:
                lines.append(f"Range: {min_sim:.1f}% - {max_sim:.1f}%")

            text = "\n".join(lines)
            fig.text(0.98, 0.98, text, ha='right', va='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#888'))
        except Exception as e:
            logger.warning(f"유사도 정보 렌더링 실패: {e}")

    out_path = os.path.join(output_dir, image_filename)
    try:
        fig.suptitle('Everyric Diagnostics', fontsize=14)
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"🖼️ 진단 이미지 저장: {out_path}")
        return out_path
    except Exception as e:
        plt.close(fig)
        logger.error(f"진단 이미지 저장 실패: {e}")
        raise
