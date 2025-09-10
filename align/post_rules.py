"""
가사 정렬 후처리 가드레일 모듈
타이밍 보정, 겹침 제거, 최소/최대 길이 보정 등
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class PostProcessingRules:
    """정렬 후처리 규칙 적용기"""
    
    def __init__(self,
                 min_duration: float = 0.3,
                 max_duration: float = 15.0,
                 min_gap: float = 0.1,
                 track_start: float = 0.0,
                 track_end: Optional[float] = None):
        """
        Args:
            min_duration: 최소 자막 길이 (초)
            max_duration: 최대 자막 길이 (초)  
            min_gap: 자막 간 최소 간격 (초)
            track_start: 트랙 시작 시간
            track_end: 트랙 종료 시간 (None이면 자동 계산)
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_gap = min_gap
        self.track_start = track_start
        self.track_end = track_end
    
    def apply_all_rules(self, subtitles: List[Dict]) -> List[Dict]:
        """모든 후처리 규칙을 순차 적용"""
        if not subtitles:
            return subtitles
        
        logger.info(f"🔧 후처리 규칙 적용 시작: {len(subtitles)}개 자막")
        
        # 1. 시간순 정렬
        result = self.sort_by_time(subtitles)
        
        # 2. 트랙 범위 클램프
        result = self.clamp_to_track_bounds(result)
        
        # 3. 단조성 보정 (겹침 제거)
        result = self.enforce_monotonic_timing(result)
        
        # 4. 최소/최대 길이 보정
        result = self.fix_duration_bounds(result)
        
        # 5. 최소 간격 보정
        result = self.enforce_minimum_gaps(result)
        
        # 6. 아웃라이어 분할
        result = self.split_outliers(result)
        
        # 7. 빈 자막 제거
        result = self.remove_empty_subtitles(result)
        
        logger.info(f"✅ 후처리 완료: {len(result)}개 자막")
        return result
    
    def sort_by_time(self, subtitles: List[Dict]) -> List[Dict]:
        """시작 시간 기준으로 정렬"""
        return sorted(subtitles, key=lambda x: float(x.get('start', 0.0)))
    
    def clamp_to_track_bounds(self, subtitles: List[Dict]) -> List[Dict]:
        """트랙 범위 내로 타이밍 제한"""
        if not subtitles:
            return subtitles
        
        # track_end 자동 계산
        track_end = self.track_end
        if track_end is None:
            track_end = max(float(sub.get('end', 0.0)) for sub in subtitles) + 1.0
        
        result = []
        for sub in subtitles:
            start = max(self.track_start, float(sub.get('start', 0.0)))
            end = min(track_end, float(sub.get('end', start + 1.0)))
            
            if end > start:  # 유효한 구간만 유지
                new_sub = dict(sub)
                new_sub['start'] = start
                new_sub['end'] = end
                result.append(new_sub)
        
        return result
    
    def enforce_monotonic_timing(self, subtitles: List[Dict]) -> List[Dict]:
        """단조성 보정: end[i] <= start[i+1] 보장"""
        if len(subtitles) <= 1:
            return subtitles
        
        result = []
        for i, sub in enumerate(subtitles):
            start = float(sub.get('start', 0.0))
            end = float(sub.get('end', start + 1.0))
            
            # 이전 자막과 겹치지 않도록 조정
            if result:
                prev_end = float(result[-1]['end'])
                if start < prev_end:
                    start = prev_end + self.min_gap
                    end = max(end, start + self.min_duration)
            
            # 다음 자막과 겹치지 않도록 조정
            if i + 1 < len(subtitles):
                next_start = float(subtitles[i + 1].get('start', end + 1.0))
                if end > next_start - self.min_gap:
                    end = next_start - self.min_gap
                    end = max(start + self.min_duration, end)
            
            if end > start:
                new_sub = dict(sub)
                new_sub['start'] = start
                new_sub['end'] = end
                result.append(new_sub)
        
        return result
    
    def fix_duration_bounds(self, subtitles: List[Dict]) -> List[Dict]:
        """최소/최대 길이 제한 적용"""
        result = []
        
        for sub in subtitles:
            start = float(sub.get('start', 0.0))
            end = float(sub.get('end', start + 1.0))
            duration = end - start
            
            # 너무 짧은 경우
            if duration < self.min_duration:
                end = start + self.min_duration
            
            # 너무 긴 경우 (분할은 split_outliers에서 처리)
            elif duration > self.max_duration:
                end = start + self.max_duration
            
            new_sub = dict(sub)
            new_sub['start'] = start
            new_sub['end'] = end
            result.append(new_sub)
        
        return result
    
    def enforce_minimum_gaps(self, subtitles: List[Dict]) -> List[Dict]:
        """자막 간 최소 간격 보장"""
        if len(subtitles) <= 1:
            return subtitles
        
        result = []
        for i, sub in enumerate(subtitles):
            start = float(sub.get('start', 0.0))
            end = float(sub.get('end', start + 1.0))
            
            # 다음 자막과의 간격 확보
            if i + 1 < len(subtitles):
                next_start = float(subtitles[i + 1].get('start', end + 1.0))
                required_end = next_start - self.min_gap
                
                if end > required_end:
                    end = required_end
                    # 최소 길이 보장
                    if end - start < self.min_duration:
                        start = end - self.min_duration
            
            if end > start:
                new_sub = dict(sub)
                new_sub['start'] = start
                new_sub['end'] = end
                result.append(new_sub)
        
        return result
    
    def split_outliers(self, subtitles: List[Dict]) -> List[Dict]:
        """비정상적으로 긴 자막을 분할"""
        result = []
        
        for sub in subtitles:
            start = float(sub.get('start', 0.0))
            end = float(sub.get('end', start + 1.0))
            duration = end - start
            text = sub.get('text', '')
            
            # 최대 길이를 초과하는 경우 분할
            if duration > self.max_duration and len(text) > 10:
                # 문자 수 기준으로 분할점 계산
                num_parts = int(np.ceil(duration / self.max_duration))
                chars_per_part = len(text) // num_parts
                
                for i in range(num_parts):
                    part_start = start + (i * duration / num_parts)
                    part_end = start + ((i + 1) * duration / num_parts)
                    
                    # 텍스트 분할
                    text_start = i * chars_per_part
                    text_end = (i + 1) * chars_per_part if i < num_parts - 1 else len(text)
                    part_text = text[text_start:text_end].strip()
                    
                    if part_text:
                        part_sub = dict(sub)
                        part_sub['start'] = part_start
                        part_sub['end'] = part_end
                        part_sub['text'] = part_text
                        result.append(part_sub)
            else:
                result.append(sub)
        
        return result
    
    def remove_empty_subtitles(self, subtitles: List[Dict]) -> List[Dict]:
        """빈 자막 제거"""
        result = []
        for sub in subtitles:
            text = sub.get('text', '').strip()
            start = float(sub.get('start', 0.0))
            end = float(sub.get('end', start))
            
            if text and end > start:
                result.append(sub)
        
        return result

def snap_to_silence_ranges(subtitles: List[Dict], 
                          silence_ranges: List[Tuple[float, float]],
                          snap_window: float = 0.5) -> List[Dict]:
    """
    자막 경계를 무성구간에 스냅
    
    Args:
        subtitles: 자막 리스트
        silence_ranges: 무성구간 리스트 [(start, end), ...]
        snap_window: 스냅 윈도우 크기 (초)
    """
    if not silence_ranges:
        return subtitles
    
    logger.info(f"🔇 무성구간 스냅 적용: {len(silence_ranges)}개 구간")
    
    result = []
    for sub in subtitles:
        start = float(sub.get('start', 0.0))
        end = float(sub.get('end', start + 1.0))
        
        # 시작점 스냅
        best_start = start
        min_start_dist = float('inf')
        
        for sil_start, sil_end in silence_ranges:
            # 무성구간 시작점과의 거리
            dist_to_start = abs(start - sil_start)
            if dist_to_start <= snap_window and dist_to_start < min_start_dist:
                min_start_dist = dist_to_start
                best_start = sil_start
            
            # 무성구간 끝점과의 거리  
            dist_to_end = abs(start - sil_end)
            if dist_to_end <= snap_window and dist_to_end < min_start_dist:
                min_start_dist = dist_to_end
                best_start = sil_end
        
        # 끝점 스냅
        best_end = end
        min_end_dist = float('inf')
        
        for sil_start, sil_end in silence_ranges:
            # 무성구간 시작점과의 거리
            dist_to_start = abs(end - sil_start)
            if dist_to_start <= snap_window and dist_to_start < min_end_dist:
                min_end_dist = dist_to_start
                best_end = sil_start
            
            # 무성구간 끝점과의 거리
            dist_to_end = abs(end - sil_end)
            if dist_to_end <= snap_window and dist_to_end < min_end_dist:
                min_end_dist = dist_to_end
                best_end = sil_end
        
        # 스냅된 타이밍이 유효한지 확인
        if best_end > best_start:
            new_sub = dict(sub)
            new_sub['start'] = best_start
            new_sub['end'] = best_end
            result.append(new_sub)
        else:
            result.append(sub)  # 원본 유지
    
    return result
