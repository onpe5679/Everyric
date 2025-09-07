import logging
from typing import List, Dict, Tuple
import re
from .phonetic_matcher import PhoneticMatcher

logger = logging.getLogger(__name__)

class SmartAligner:
    """Whisper 음성 인식 결과와 원본 가사를 지능적으로 정렬하는 클래스"""
    
    def __init__(self):
        self.logger = logger
        self.phonetic_matcher = PhoneticMatcher()
        
    def align_lyrics_with_timing(self, lyrics: List[str], whisper_segments: List[Dict]) -> List[Dict]:
        """
        원본 가사와 Whisper 세그먼트를 정렬하여 정확한 타이밍 정보를 생성합니다.
        
        Args:
            lyrics: 원본 가사 라인 리스트
            whisper_segments: Whisper 음성 인식 결과 세그먼트
            
        Returns:
            정렬된 가사-타이밍 데이터 리스트
        """
        self.logger.info(f"🔄 스마트 정렬 시작 - 가사: {len(lyrics)}줄, 세그먼트: {len(whisper_segments)}개")
        
        # 1. 가사와 Whisper 텍스트 전처리
        processed_lyrics = self._preprocess_lyrics(lyrics)
        processed_segments = self._preprocess_segments(whisper_segments)
        
        # 2. 전체 텍스트 기반 매칭
        alignment_map = self._create_alignment_map(processed_lyrics, processed_segments)
        
        # 3. 타이밍 정보 할당
        aligned_result = self._assign_timing(lyrics, whisper_segments, alignment_map)
        
        self.logger.info(f"✅ 정렬 완료 - {len(aligned_result)}개의 타이밍 가사 생성")
        return aligned_result
    
    def _preprocess_lyrics(self, lyrics: List[str]) -> List[str]:
        """가사 전처리 - 특수문자 제거, 공백 정리"""
        processed = []
        for lyric in lyrics:
            # 특수문자 제거, 소문자 변환, 공백 정리
            clean = re.sub(r'[^\w\s가-힣ぁ-んァ-ヶー一-龯]', '', lyric)
            clean = re.sub(r'\s+', ' ', clean).strip().lower()
            if clean:
                processed.append(clean)
        return processed
    
    def _preprocess_segments(self, segments: List[Dict]) -> List[str]:
        """Whisper 세그먼트 전처리"""
        processed = []
        for segment in segments:
            text = segment.get('text', '').strip()
            # 특수문자 제거, 소문자 변환, 공백 정리
            clean = re.sub(r'[^\w\s가-힣ぁ-んァ-ヶー一-龯]', '', text)
            clean = re.sub(r'\s+', ' ', clean).strip().lower()
            if clean:
                processed.append(clean)
        return processed
    
    def _create_alignment_map(self, lyrics: List[str], segments: List[str]) -> List[Tuple[int, List[int]]]:
        """
        가사 라인과 세그먼트 간의 매핑 생성
        
        Returns:
            [(lyric_index, [segment_indices]), ...] 형태의 매핑
        """
        alignment_map = []
        
        # 전체 텍스트 결합
        full_lyrics = ' '.join(lyrics)
        full_segments = ' '.join(segments)
        
        # 발음 기반 매칭
        lyric_pos = 0
        segment_pos = 0
        
        for lyric_idx, lyric in enumerate(lyrics):
            # 현재 가사 라인에 해당하는 세그먼트들 찾기
            best_match_ratio = 0
            best_segments = []
            
            # 여러 세그먼트 조합으로 매칭 시도
            for seg_count in range(1, min(5, len(segments) - segment_pos + 1)):
                if segment_pos + seg_count > len(segments):
                    break
                    
                combined_segments = ' '.join(segments[segment_pos:segment_pos + seg_count])
                # 발음 기반 유사도 계산
                ratio = self.phonetic_matcher.calculate_phonetic_similarity(lyric, combined_segments)
                
                if ratio > best_match_ratio:
                    best_match_ratio = ratio
                    best_segments = list(range(segment_pos, segment_pos + seg_count))
            
            # 매칭 결과 저장
            if best_segments and best_match_ratio > 0.3:  # 30% 이상 유사도
                alignment_map.append((lyric_idx, best_segments))
                segment_pos = max(best_segments) + 1
            else:
                # 매칭 실패 시 다음 세그먼트로
                if segment_pos < len(segments):
                    alignment_map.append((lyric_idx, [segment_pos]))
                    segment_pos += 1
                else:
                    alignment_map.append((lyric_idx, []))
        
        return alignment_map
    
    def _assign_timing(self, original_lyrics: List[str], segments: List[Dict], 
                      alignment_map: List[Tuple[int, List[int]]]) -> List[Dict]:
        """매핑 정보를 기반으로 타이밍 할당"""
        result = []
        
        for lyric_idx, segment_indices in alignment_map:
            if not segment_indices:
                # 매칭된 세그먼트가 없는 경우
                result.append({
                    'start': 0.0,
                    'end': 0.0,
                    'text': original_lyrics[lyric_idx],
                    'confidence': 0.0
                })
                continue
            
            # 매칭된 세그먼트들의 타이밍 정보 결합
            start_time = segments[segment_indices[0]]['start']
            end_time = segments[segment_indices[-1]]['end']
            
            # 신뢰도 계산 (매칭된 세그먼트 수와 유사도 기반)
            confidence = min(1.0, len(segment_indices) / 3.0)
            
            result.append({
                'start': start_time,
                'end': end_time,
                'text': original_lyrics[lyric_idx],
                'confidence': confidence,
                'matched_segments': segment_indices
            })
            
            # 디버그 정보
            if lyric_idx < 5 or lyric_idx >= len(original_lyrics) - 5:
                matched_texts = [segments[i].get('text', '') for i in segment_indices]
                self.logger.debug(f"정렬 {lyric_idx+1}: '{original_lyrics[lyric_idx]}' → "
                                f"세그먼트 {segment_indices} (신뢰도: {confidence:.2f})")
        
        return result

def align_lyrics_with_whisper(lyrics: List[str], whisper_segments: List[Dict]) -> List[Dict]:
    """편의 함수: 스마트 정렬 실행"""
    aligner = SmartAligner()
    return aligner.align_lyrics_with_timing(lyrics, whisper_segments)
