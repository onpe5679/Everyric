from typing import List, Dict
import logging

# 로거 설정
logger = logging.getLogger(__name__)

def _format_time(time: float) -> str:
    minutes, seconds = divmod(time, 60)
    return f"{int(minutes):02d}:{seconds:05.2f}"

def align_segments(lyrics: List[str], segments: List[Dict]) -> List[Dict]:
    """
    가사와 음성 인식 세그먼트를 정렬합니다.
    간단하게 각 가사 라인을 순서대로 세그먼트에 매핑합니다.
    
    Args:
        lyrics: 가사 라인 리스트
        segments: 음성 인식 세그먼트 리스트 (start, end, text 키 포함)
        
    Returns:
        List[Dict]: 정렬된 세그먼트 리스트 (start, end, original 키 포함)
    """
    try:
        logger.info(" 가사와 음성 세그먼트 정렬 시작")
        
        if not lyrics:
            logger.warning(" 정렬할 가사가 없습니다.")
            return []
            
        if not segments:
            logger.warning(" 정렬할 음성 세그먼트가 없습니다.")
            return []
            
        # 가사와 세그먼트 수 로깅
        logger.info(f"📊 가사 라인 수: {len(lyrics)}, 음성 세그먼트 수: {len(segments)}")
        
        if len(lyrics) != len(segments):
            logger.warning(f"⚠️ 가사와 세그먼트 수가 일치하지 않습니다! 가사: {len(lyrics)}, 세그먼트: {len(segments)}")
            logger.info("🔄 세그먼트 수에 맞춰 가사를 조정합니다...")
        
        aligned = []
        
        # 세그먼트 수만큼 정렬 (세그먼트가 더 많으면 가사를 반복/확장)
        for i in range(len(segments)):
            segment = segments[i]
            
            # 가사가 부족하면 마지막 가사를 반복하거나 빈 문자열 사용
            if i < len(lyrics):
                lyric_text = lyrics[i]
            elif lyrics:  # 가사가 있지만 부족한 경우 마지막 가사 반복
                lyric_text = lyrics[-1] + f" (반복 {i - len(lyrics) + 2})"
            else:  # 가사가 아예 없는 경우
                lyric_text = f"[세그먼트 {i+1}]"
            
            aligned_segment = {
                'start': segment['start'],
                'end': segment['end'],
                'original': lyric_text,
                'recognized': segment.get('text', '').strip()
            }
            aligned.append(aligned_segment)
            
            # 디버그 정보: 첫 5개와 마지막 5개 세그먼트만 로깅
            if i < 5 or i >= len(segments) - 5:
                try:
                    logger.debug(f"정렬됨 - 세그먼트 {i+1}/{len(segments)}: "
                               f"{_format_time(segment['start'])} → {_format_time(segment['end'])} | "
                               f"인식: '{aligned_segment['recognized']}' | "
                               f"가사: '{lyric_text}'")
                except Exception as debug_error:
                    logger.debug(f"정렬됨 - 세그먼트 {i+1}/{len(segments)} (시간 포맷 오류: {debug_error})")
        
        logger.info(f"✅ {len(aligned)}개의 세그먼트가 정렬되었습니다.")
        return aligned
    except Exception as e:
        logger.error(f"정렬 실패: {e}")
        return []
