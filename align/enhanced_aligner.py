"""
향상된 가사 정렬 시스템 - DTW + 이중 VAD 스냅 + 가드레일 + LLM 백오프
"""
import logging
from typing import List, Dict, Optional, Tuple
from .dtw_aligner import DTWAligner
from .post_rules import PostProcessingRules, snap_to_silence_ranges
from .llm_fallback import LLMFallbackAligner

logger = logging.getLogger(__name__)

class EnhancedAligner:
    """DTW + 후처리 규칙을 결합한 향상된 정렬기"""
    
    def __init__(self, 
                 max_segment_length: int = 15,
                 alignment_config: Optional[Dict] = None):
        """
        Args:
            max_segment_length: 최대 세그먼트 길이 (문자 수, 기존 호환성)
            alignment_config: 정렬 설정 딕셔너리
        """
        self.max_segment_length = max_segment_length
        
        # 기본 설정
        config = alignment_config or {}
        
        # DTW 정렬기 설정
        self.dtw_aligner = DTWAligner(
            similarity_weight=config.get('similarity_weight', 0.7),
            time_weight=config.get('time_weight', 0.3),
            char_per_second=config.get('char_per_second', 8.0)
        )
        
        # 후처리 규칙 설정
        self.post_processor = PostProcessingRules(
            min_duration=config.get('min_duration', 0.3),
            max_duration=config.get('max_duration', 15.0),
            min_gap=config.get('min_gap', 0.1),
            track_start=config.get('track_start', 0.0),
            track_end=config.get('track_end', None)
        )
        
        # VAD 스냅 설정
        self.enable_vad_snap = config.get('enable_vad_snap', True)
        self.vad_snap_window = config.get('vad_snap_window', 0.5)
        
        # LLM 백오프 설정
        self.llm_fallback = LLMFallbackAligner(
            similarity_threshold=config.get('llm_similarity_threshold', 0.3),
            unmatched_ratio_threshold=config.get('llm_unmatched_threshold', 0.4),
            enable_llm_fallback=config.get('enable_llm_fallback', False)
        )
        
        logger.info(f"🔧 향상된 정렬기 초기화 완료 (DTW + 가드레일 + LLM백오프)")
    
    def align_lyrics_with_timing(self, 
                               lyrics_lines: List[str], 
                               segments: List[Dict],
                               silence_ranges: Optional[List[Tuple[float, float]]] = None) -> List[Dict]:
        """
        가사와 세그먼트를 정렬하고 후처리 적용
        
        Args:
            lyrics_lines: 가사 라인 리스트
            segments: ASR 세그먼트 리스트
            silence_ranges: 무성구간 리스트 (이중 VAD 스냅용)
            
        Returns:
            정렬된 자막 리스트
        """
        if not lyrics_lines or not segments:
            logger.warning("가사 또는 세그먼트가 비어있습니다")
            return []
        
        logger.info(f"🎯 향상된 정렬 시작: 가사 {len(lyrics_lines)}줄, 세그먼트 {len(segments)}개")
        
        # 1단계: DTW 기반 전역 정렬
        logger.info("📊 DTW 전역 정렬 수행 중...")
        aligned_subtitles = self.dtw_aligner.align_lyrics_with_timing(lyrics_lines, segments)
        
        if not aligned_subtitles:
            logger.warning("DTW 정렬 결과가 비어있습니다")
            return []
        
        logger.info(f"✅ DTW 정렬 완료: {len(aligned_subtitles)}개 자막 생성")
        
        # 2단계: 이중 VAD 스냅 (선택적)
        if self.enable_vad_snap and silence_ranges:
            logger.info("🔇 이중 VAD 스냅 적용 중...")
            aligned_subtitles = snap_to_silence_ranges(
                aligned_subtitles, 
                silence_ranges, 
                self.vad_snap_window
            )
            logger.info("✅ VAD 스냅 완료")
        
        # 3단계: 가드레일 후처리 적용
        logger.info("🛡️ 가드레일 후처리 적용 중...")
        
        # 트랙 종료 시간을 세그먼트에서 추정
        if segments:
            track_end = max(float(seg.get('end', 0.0)) for seg in segments) + 1.0
            self.post_processor.track_end = track_end
        
        aligned_subtitles = self.post_processor.apply_all_rules(aligned_subtitles)
        
        # 4단계: LLM 백오프 (선택적)
        if self.llm_fallback.should_use_llm_fallback(lyrics_lines, aligned_subtitles, segments):
            try:
                # OpenAI API 키 가져오기 (환경변수 또는 config에서)
                import os
                api_key = os.getenv('OPENAI_API_KEY')
                
                llm_result = self.llm_fallback.realign_with_llm(lyrics_lines, segments, api_key)
                if llm_result:
                    logger.info("🤖 LLM 백오프 결과로 교체")
                    aligned_subtitles = llm_result
                    # LLM 결과에도 가드레일 적용
                    aligned_subtitles = self.post_processor.apply_all_rules(aligned_subtitles)
            except Exception as e:
                logger.warning(f"LLM 백오프 실패: {e}")
        
        logger.info(f"🎉 향상된 정렬 완료: 최종 {len(aligned_subtitles)}개 자막")
        
        # 품질 통계 로깅
        self._log_alignment_quality(lyrics_lines, aligned_subtitles, segments)
        
        return aligned_subtitles
    
    def _log_alignment_quality(self, 
                             lyrics_lines: List[str], 
                             aligned_subtitles: List[Dict],
                             segments: List[Dict]) -> None:
        """정렬 품질 통계 로깅"""
        try:
            total_lyrics = len(lyrics_lines)
            total_aligned = len(aligned_subtitles)
            total_segments = len(segments)
            
            # 매칭률 계산
            match_rate = (total_aligned / total_lyrics * 100) if total_lyrics > 0 else 0
            
            # 평균 자막 길이
            if aligned_subtitles:
                durations = [float(sub['end']) - float(sub['start']) for sub in aligned_subtitles]
                avg_duration = sum(durations) / len(durations)
                min_duration = min(durations)
                max_duration = max(durations)
            else:
                avg_duration = min_duration = max_duration = 0.0
            
            logger.info("📈 정렬 품질 통계:")
            logger.info(f"   • 매칭률: {match_rate:.1f}% ({total_aligned}/{total_lyrics})")
            logger.info(f"   • 세그먼트 활용: {total_segments}개 → {total_aligned}개 자막")
            logger.info(f"   • 평균 자막 길이: {avg_duration:.2f}초")
            logger.info(f"   • 길이 범위: {min_duration:.2f}~{max_duration:.2f}초")
            
        except Exception as e:
            logger.warning(f"품질 통계 계산 실패: {e}")

# 기존 SmartAligner와의 호환성을 위한 래퍼
class SmartAligner(EnhancedAligner):
    """기존 SmartAligner 인터페이스 호환성 유지"""
    
    def __init__(self, max_segment_length: int = 15, alignment_config: Optional[Dict] = None):
        super().__init__(max_segment_length, alignment_config)
        logger.info("⚠️ SmartAligner는 EnhancedAligner로 업그레이드되었습니다")
