import logging
from typing import List, Dict, Tuple
import re
from .phonetic_matcher import PhoneticMatcher

logger = logging.getLogger(__name__)

class SmartAligner:
    """Whisper 음성 인식 결과와 원본 가사를 지능적으로 정렬하는 클래스"""
    
    def __init__(self, max_segment_length: int = 100, alignment_config: dict = None):
        """
        Args:
            max_segment_length: 각 자막 세그먼트의 최대 문자 길이 (기본값: 100)
            alignment_config: 정렬 설정 딕셔너리
        """
        self.logger = logger
        self.phonetic_matcher = PhoneticMatcher()
        self.max_segment_length = max_segment_length
        
        # 설정 기본값
        default_config = {
            "similarity_threshold": 0.15,
            "max_segment_combinations": 5,
            "enable_sequential_fallback": True,
            "split_long_segments": True,
            "time_distribution_method": "character_based"
        }
        
        self.config = {**default_config, **(alignment_config or {})}
        
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
            for seg_count in range(1, min(self.config["max_segment_combinations"], len(segments) - segment_pos + 1)):
                if segment_pos + seg_count > len(segments):
                    break
                    
                combined_segments = ' '.join(segments[segment_pos:segment_pos + seg_count])
                # 발음 기반 유사도 계산
                ratio = self.phonetic_matcher.calculate_phonetic_similarity(lyric, combined_segments)
                
                if ratio > best_match_ratio:
                    best_match_ratio = ratio
                    best_segments = list(range(segment_pos, segment_pos + seg_count))
            
            # 매칭 결과 저장 (설정 기반 임계값 적용)
            if best_segments and best_match_ratio > self.config["similarity_threshold"]:
                alignment_map.append((lyric_idx, best_segments))
                segment_pos = max(best_segments) + 1
            else:
                # 순차적 폴백 설정이 활성화된 경우
                if self.config["enable_sequential_fallback"]:
                    if segment_pos < len(segments):
                        alignment_map.append((lyric_idx, [segment_pos]))
                        segment_pos += 1
                    else:
                        # 세그먼트가 부족한 경우 마지막 세그먼트 재사용
                        if len(segments) > 0:
                            alignment_map.append((lyric_idx, [len(segments) - 1]))
                        else:
                            alignment_map.append((lyric_idx, []))
                else:
                    alignment_map.append((lyric_idx, []))
        
        return alignment_map
    
    def _assign_timing(self, original_lyrics: List[str], segments: List[Dict], 
                      alignment_map: List[Tuple[int, List[int]]]) -> List[Dict]:
        """매핑 정보를 기반으로 타이밍 할당 (최대 길이 제한 적용)"""
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
            
            lyric_text = original_lyrics[lyric_idx]
            
            # 최대 길이 제한 적용 (설정 기반)
            if not self.config["split_long_segments"] or len(lyric_text) <= self.max_segment_length:
                # 길이가 제한 내이거나 분할 비활성화인 경우 그대로 추가
                result.append({
                    'start': start_time,
                    'end': end_time,
                    'text': lyric_text,
                    'confidence': confidence,
                    'matched_segments': segment_indices
                })
            else:
                # 길이가 제한을 초과하는 경우 분할
                split_segments = self._split_long_text(
                    lyric_text, start_time, end_time, confidence, segment_indices
                )
                result.extend(split_segments)
                self.logger.debug(f"긴 텍스트 분할: '{lyric_text[:30]}...' → {len(split_segments)}개 세그먼트")
            
            # 디버그 정보
            if lyric_idx < 5 or lyric_idx >= len(original_lyrics) - 5:
                matched_texts = [segments[i].get('text', '') for i in segment_indices]
                self.logger.debug(f"정렬 {lyric_idx+1}: '{original_lyrics[lyric_idx][:30]}...' → "
                                f"세그먼트 {segment_indices} (신뢰도: {confidence:.2f})")
        
        return result
    
    def _split_long_text(self, text: str, start_time: float, end_time: float, 
                        confidence: float, segment_indices: List[int]) -> List[Dict]:
        """
        긴 텍스트를 최대 길이 제한에 맞게 분할합니다.
        
        Args:
            text: 분할할 텍스트
            start_time: 시작 시간
            end_time: 종료 시간
            confidence: 신뢰도
            segment_indices: 매칭된 세그먼트 인덱스들
            
        Returns:
            분할된 세그먼트 리스트
        """
        segments = []
        total_duration = end_time - start_time
        
        # 자연스러운 분할점 찾기 (공백, 쉼표, 마침표 등)
        split_points = self._find_natural_split_points(text)
        
        # 분할점을 기준으로 청크 생성
        chunks = self._create_chunks_from_split_points(text, split_points)
        
        # 각 청크에 시간 할당 (설정 기반 시간 분배)
        if self.config["time_distribution_method"] == "character_based":
            # 문자 수 기반 분배
            total_chars = sum(len(chunk.strip()) for chunk in chunks if chunk.strip())
            current_time = start_time
            
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                    
                chunk_chars = len(chunk.strip())
                chunk_duration = (total_duration * chunk_chars / total_chars) if total_chars > 0 else (total_duration / len(chunks))
                
                chunk_start = current_time
                chunk_end = current_time + chunk_duration
                current_time = chunk_end
                
                segments.append({
                    'start': chunk_start,
                    'end': chunk_end,
                    'text': chunk.strip(),
                    'confidence': confidence * 0.8,
                    'matched_segments': segment_indices,
                    'is_split': True
                })
        else:
            # 균등 분배 (기본값)
            current_time = start_time
            chunk_duration = total_duration / len(chunks)
            
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                    
                chunk_start = current_time
                chunk_end = current_time + chunk_duration
                current_time = chunk_end
                
                segments.append({
                    'start': chunk_start,
                    'end': chunk_end,
                    'text': chunk.strip(),
                    'confidence': confidence * 0.8,  # 분할된 세그먼트는 신뢰도 약간 감소
                    'matched_segments': segment_indices,
                    'is_split': True
                })
        
        return segments
    
    def _find_natural_split_points(self, text: str) -> List[int]:
        """텍스트에서 자연스러운 분할점을 찾습니다."""
        split_points = [0]  # 시작점
        
        # 우선순위: 마침표 > 쉼표 > 공백
        separators = ['.', '!', '?', ',', ';', ' ']
        
        current_pos = 0
        while current_pos < len(text):
            # 최대 길이 내에서 가장 좋은 분할점 찾기
            search_end = min(current_pos + self.max_segment_length, len(text))
            best_split = search_end
            
            # 뒤에서부터 분할점 찾기 (더 자연스러운 분할을 위해)
            for pos in range(search_end - 1, current_pos, -1):
                if text[pos] in separators:
                    best_split = pos + 1
                    break
            
            if best_split > current_pos:
                split_points.append(best_split)
                current_pos = best_split
            else:
                # 분할점을 찾지 못한 경우 강제 분할
                current_pos += self.max_segment_length
                if current_pos < len(text):
                    split_points.append(current_pos)
        
        if split_points[-1] != len(text):
            split_points.append(len(text))
        
        return split_points
    
    def _create_chunks_from_split_points(self, text: str, split_points: List[int]) -> List[str]:
        """분할점을 기준으로 텍스트 청크를 생성합니다."""
        chunks = []
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

def align_lyrics_with_whisper(lyrics: List[str], whisper_segments: List[Dict]) -> List[Dict]:
    """편의 함수: 스마트 정렬 실행"""
    aligner = SmartAligner()
    return aligner.align_lyrics_with_timing(lyrics, whisper_segments)
