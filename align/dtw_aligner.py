"""
DTW 기반 전역 가사-세그먼트 정렬 시스템
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import re
from difflib import SequenceMatcher

@dataclass
class AlignmentSegment:
    """정렬용 세그먼트 구조체"""
    start: float
    end: float
    text: str
    confidence: float = 1.0

def normalize_text(text: str) -> str:
    """텍스트 정규화 (공백, 특수문자 제거)"""
    if not text:
        return ""
    # 한글, 일본어, 영문, 숫자만 유지
    text = re.sub(r'[^\w\sあ-んア-ンー一-龯가-힣]', '', text)
    # 연속 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def fuzzy_similarity(text1: str, text2: str) -> float:
    """두 텍스트 간 유사도 계산 (0~1)"""
    if not text1 or not text2:
        return 0.0
    
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    if not norm1 or not norm2:
        return 0.0
    
    # SequenceMatcher 기반 유사도
    matcher = SequenceMatcher(None, norm1, norm2)
    return matcher.ratio()

def time_length_prior(lyrics_line: str, segment: AlignmentSegment, 
                     char_per_second: float = 8.0) -> float:
    """시간-길이 사전 확률 (짧을수록 좋음)"""
    if not lyrics_line:
        return 1.0
    
    expected_duration = len(lyrics_line) / char_per_second
    actual_duration = max(0.1, segment.end - segment.start)
    
    # 예상 길이와 실제 길이 차이를 패널티로
    ratio = min(expected_duration, actual_duration) / max(expected_duration, actual_duration)
    return 1.0 - ratio

def dtw_align(lyrics_lines: List[str], 
              segments: List[AlignmentSegment],
              similarity_weight: float = 0.7,
              time_weight: float = 0.3) -> List[Tuple[int, int]]:
    """
    DTW를 사용한 전역 정렬
    
    Returns:
        List of (lyrics_idx, segment_idx) pairs representing the alignment path
    """
    N = len(lyrics_lines)  # 가사 라인 수
    M = len(segments)      # 세그먼트 수
    
    if N == 0 or M == 0:
        return []
    
    # DTW 거리 매트릭스 초기화
    # dp[i][j] = lyrics[0:i]와 segments[0:j]를 정렬하는 최소 비용
    dp = np.full((N + 1, M + 1), float('inf'))
    dp[0][0] = 0.0
    
    # 백트래킹을 위한 경로 저장
    path = {}
    
    # DTW 동적 프로그래밍
    for i in range(N + 1):
        for j in range(M + 1):
            if i == 0 and j == 0:
                continue
                
            candidates = []
            
            # Match: lyrics[i-1] <-> segments[j-1]
            if i > 0 and j > 0:
                similarity = fuzzy_similarity(lyrics_lines[i-1], segments[j-1].text)
                time_prior = time_length_prior(lyrics_lines[i-1], segments[j-1])
                
                cost = 1.0 - (similarity_weight * similarity + time_weight * (1.0 - time_prior))
                candidates.append((dp[i-1][j-1] + cost, 'match'))
            
            # Insert: skip segment (세그먼트 건너뛰기)
            if j > 0:
                candidates.append((dp[i][j-1] + 0.5, 'insert'))
            
            # Delete: skip lyrics line (가사 라인 건너뛰기)  
            if i > 0:
                candidates.append((dp[i-1][j] + 0.8, 'delete'))
            
            if candidates:
                min_cost, operation = min(candidates)
                dp[i][j] = min_cost
                path[(i, j)] = operation
    
    # 백트래킹으로 최적 경로 복원
    alignment_path = []
    i, j = N, M
    
    while i > 0 or j > 0:
        if (i, j) not in path:
            break
            
        operation = path[(i, j)]
        
        if operation == 'match':
            alignment_path.append((i-1, j-1))
            i -= 1
            j -= 1
        elif operation == 'insert':
            j -= 1  # 세그먼트만 건너뛰기
        elif operation == 'delete':
            i -= 1  # 가사만 건너뛰기
    
    alignment_path.reverse()
    return alignment_path

def assign_timings_from_path(lyrics_lines: List[str],
                           segments: List[AlignmentSegment], 
                           alignment_path: List[Tuple[int, int]]) -> List[Optional[Tuple[float, float]]]:
    """
    DTW 경로로부터 가사 라인별 타이밍 할당
    """
    N = len(lyrics_lines)
    timings = [None] * N
    
    if not alignment_path:
        return timings
    
    # 경로를 가사 인덱스별로 그룹화
    lyrics_to_segments = {}
    for lyrics_idx, segment_idx in alignment_path:
        if lyrics_idx not in lyrics_to_segments:
            lyrics_to_segments[lyrics_idx] = []
        lyrics_to_segments[lyrics_idx].append(segment_idx)
    
    # 각 가사 라인에 타이밍 할당
    for lyrics_idx in range(N):
        if lyrics_idx in lyrics_to_segments:
            segment_indices = lyrics_to_segments[lyrics_idx]
            
            if len(segment_indices) == 1:
                # 1:1 매칭
                seg_idx = segment_indices[0]
                seg = segments[seg_idx]
                timings[lyrics_idx] = (seg.start, seg.end)
                
            elif len(segment_indices) > 1:
                # 1:N 매칭 (여러 세그먼트를 하나의 가사 라인에 합병)
                start_time = segments[segment_indices[0]].start
                end_time = segments[segment_indices[-1]].end
                timings[lyrics_idx] = (start_time, end_time)
    
    return timings

def split_merge_by_character_length(lyrics_lines: List[str],
                                  timings: List[Optional[Tuple[float, float]]]) -> List[Optional[Tuple[float, float]]]:
    """
    문자 길이 비례 분할/합병
    """
    N = len(lyrics_lines)
    result_timings = list(timings)
    
    # 매칭되지 않은 라인들을 보간으로 처리
    for i in range(N):
        if result_timings[i] is None:
            result_timings[i] = interpolate_timing(i, result_timings, lyrics_lines)
    
    return result_timings

def interpolate_timing(target_idx: int, 
                      timings: List[Optional[Tuple[float, float]]],
                      lyrics_lines: List[str]) -> Optional[Tuple[float, float]]:
    """
    매칭되지 않은 가사 라인의 타이밍을 보간
    """
    N = len(timings)
    
    # 이전/이후 유효한 타이밍 찾기
    prev_timing = None
    next_timing = None
    
    for i in range(target_idx - 1, -1, -1):
        if timings[i] is not None:
            prev_timing = timings[i]
            break
    
    for i in range(target_idx + 1, N):
        if timings[i] is not None:
            next_timing = timings[i]
            break
    
    # 보간 계산
    if prev_timing and next_timing:
        # 문자 길이 비례 보간
        char_count_before = sum(len(lyrics_lines[j]) for j in range(target_idx))
        char_count_target = len(lyrics_lines[target_idx])
        char_count_after = sum(len(lyrics_lines[j]) for j in range(target_idx + 1, N))
        
        total_chars = char_count_before + char_count_target + char_count_after
        if total_chars > 0:
            time_span = next_timing[1] - prev_timing[0]
            target_duration = (char_count_target / total_chars) * time_span
            
            start_time = prev_timing[1]
            end_time = start_time + target_duration
            return (start_time, end_time)
    
    elif prev_timing:
        # 이전 타이밍만 있는 경우
        duration = max(1.0, len(lyrics_lines[target_idx]) / 8.0)  # 8자/초 가정
        return (prev_timing[1], prev_timing[1] + duration)
        
    elif next_timing:
        # 이후 타이밍만 있는 경우  
        duration = max(1.0, len(lyrics_lines[target_idx]) / 8.0)
        return (next_timing[0] - duration, next_timing[0])
    
    # 둘 다 없으면 기본값
    return (0.0, 3.0)

class DTWAligner:
    """DTW 기반 가사 정렬기"""
    
    def __init__(self, 
                 similarity_weight: float = 0.7,
                 time_weight: float = 0.3,
                 char_per_second: float = 8.0):
        self.similarity_weight = similarity_weight
        self.time_weight = time_weight  
        self.char_per_second = char_per_second
    
    def align_lyrics_with_timing(self, 
                               lyrics_lines: List[str],
                               segments: List[Dict]) -> List[Dict]:
        """
        가사 라인과 ASR 세그먼트를 DTW로 정렬
        
        Args:
            lyrics_lines: 가사 라인 리스트
            segments: ASR 세그먼트 리스트 (start, end, text 포함)
            
        Returns:
            정렬된 자막 리스트 (start, end, text, confidence 포함)
        """
        if not lyrics_lines or not segments:
            return []
        
        # 세그먼트를 AlignmentSegment로 변환
        alignment_segments = []
        for seg in segments:
            alignment_segments.append(AlignmentSegment(
                start=float(seg.get('start', 0.0)),
                end=float(seg.get('end', 0.0)), 
                text=seg.get('text', ''),
                confidence=float(seg.get('confidence', 1.0))
            ))
        
        # DTW 정렬 수행
        alignment_path = dtw_align(
            lyrics_lines, 
            alignment_segments,
            self.similarity_weight,
            self.time_weight
        )
        
        # 타이밍 할당
        timings = assign_timings_from_path(lyrics_lines, alignment_segments, alignment_path)
        
        # 문자 길이 기반 분할/합병
        timings = split_merge_by_character_length(lyrics_lines, timings)
        
        # 결과 생성
        aligned_subtitles = []
        for i, (line, timing) in enumerate(zip(lyrics_lines, timings)):
            if timing and line.strip():
                aligned_subtitles.append({
                    'start': timing[0],
                    'end': timing[1], 
                    'text': line.strip(),
                    'confidence': 0.8  # DTW 기본 신뢰도
                })
        
        return aligned_subtitles
