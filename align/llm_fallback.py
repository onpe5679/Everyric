"""
LLM 백오프 시스템 - 정렬 실패 시 LLM을 활용한 재정렬
"""
import logging
from typing import List, Dict, Optional, Tuple
import json
import re

logger = logging.getLogger(__name__)

class LLMFallbackAligner:
    """LLM 기반 백오프 정렬기"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.3,
                 unmatched_ratio_threshold: float = 0.4,
                 enable_llm_fallback: bool = False):
        """
        Args:
            similarity_threshold: 평균 유사도 임계값 (이하일 때 LLM 호출)
            unmatched_ratio_threshold: 미매칭 비율 임계값 (이상일 때 LLM 호출)
            enable_llm_fallback: LLM 백오프 활성화 여부
        """
        self.similarity_threshold = similarity_threshold
        self.unmatched_ratio_threshold = unmatched_ratio_threshold
        self.enable_llm_fallback = enable_llm_fallback
    
    def should_use_llm_fallback(self, 
                               lyrics_lines: List[str],
                               aligned_subtitles: List[Dict],
                               segments: List[Dict]) -> bool:
        """LLM 백오프 사용 여부 판단"""
        if not self.enable_llm_fallback:
            return False
        
        try:
            # 매칭률 계산
            total_lyrics = len(lyrics_lines)
            total_aligned = len(aligned_subtitles)
            
            if total_lyrics == 0:
                return False
            
            unmatched_ratio = 1.0 - (total_aligned / total_lyrics)
            
            # 평균 유사도 계산 (간단한 추정)
            from .dtw_aligner import fuzzy_similarity
            
            similarities = []
            for sub in aligned_subtitles:
                sub_text = sub.get('text', '')
                # 가장 유사한 세그먼트 찾기
                best_sim = 0.0
                for seg in segments:
                    seg_text = seg.get('text', '')
                    sim = fuzzy_similarity(sub_text, seg_text)
                    best_sim = max(best_sim, sim)
                similarities.append(best_sim)
            
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            
            logger.info(f"📊 정렬 품질 평가: 평균유사도={avg_similarity:.3f}, 미매칭률={unmatched_ratio:.3f}")
            
            # 임계값 검사
            use_llm = (avg_similarity < self.similarity_threshold or 
                      unmatched_ratio > self.unmatched_ratio_threshold)
            
            if use_llm:
                logger.info("🤖 LLM 백오프 조건 충족 - LLM 재정렬 수행")
            
            return use_llm
            
        except Exception as e:
            logger.warning(f"LLM 백오프 판단 실패: {e}")
            return False
    
    def realign_with_llm(self,
                        lyrics_lines: List[str],
                        segments: List[Dict],
                        api_key: Optional[str] = None) -> Optional[List[Dict]]:
        """LLM을 사용한 재정렬"""
        if not api_key:
            logger.warning("LLM API 키가 없어 백오프를 건너뜁니다")
            return None
        
        try:
            logger.info("🤖 LLM 기반 재정렬 시작...")
            
            # 프롬프트 생성
            prompt = self._create_alignment_prompt(lyrics_lines, segments)
            
            # LLM 호출 (OpenAI GPT 사용)
            response = self._call_llm(prompt, api_key)
            
            if response:
                # 응답 파싱
                aligned_subtitles = self._parse_llm_response(response, lyrics_lines, segments)
                
                if aligned_subtitles:
                    logger.info(f"✅ LLM 재정렬 완료: {len(aligned_subtitles)}개 자막")
                    return aligned_subtitles
            
            logger.warning("LLM 재정렬 실패")
            return None
            
        except Exception as e:
            logger.error(f"LLM 백오프 오류: {e}")
            return None
    
    def _create_alignment_prompt(self, 
                               lyrics_lines: List[str], 
                               segments: List[Dict]) -> str:
        """정렬용 프롬프트 생성"""
        
        # 가사 라인들을 번호와 함께 포맷
        lyrics_text = "\n".join([f"{i+1}. {line}" for i, line in enumerate(lyrics_lines)])
        
        # 세그먼트들을 시간과 함께 포맷
        segments_text = "\n".join([
            f"[{seg.get('start', 0):.2f}-{seg.get('end', 0):.2f}s] {seg.get('text', '')}"
            for seg in segments
        ])
        
        prompt = f"""다음은 음성 인식 결과와 가사입니다. 각 가사 라인을 적절한 시간대의 음성 인식 결과와 매칭해주세요.

가사 라인들:
{lyrics_text}

음성 인식 결과 (시간대별):
{segments_text}

다음 JSON 형식으로 매칭 결과를 출력해주세요:
[
  {{"lyrics_line": 1, "start_time": 0.0, "end_time": 3.5, "text": "첫 번째 가사"}},
  {{"lyrics_line": 2, "start_time": 3.5, "end_time": 7.2, "text": "두 번째 가사"}},
  ...
]

규칙:
1. 가사의 의미와 음성 인식 결과의 유사성을 고려하세요
2. 시간 순서를 지켜주세요 (start_time이 증가하는 순서)
3. 겹치지 않는 시간대를 할당하세요
4. 모든 가사 라인을 포함해주세요"""

        return prompt
    
    def _call_llm(self, prompt: str, api_key: str) -> Optional[str]:
        """LLM API 호출"""
        try:
            import openai
            
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 가사와 음성 인식 결과를 정확하게 매칭하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM API 호출 실패: {e}")
            return None
    
    def _parse_llm_response(self, 
                          response: str, 
                          lyrics_lines: List[str],
                          segments: List[Dict]) -> Optional[List[Dict]]:
        """LLM 응답 파싱"""
        try:
            # JSON 부분 추출
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                logger.warning("LLM 응답에서 JSON을 찾을 수 없습니다")
                return None
            
            json_str = json_match.group(0)
            alignment_data = json.loads(json_str)
            
            # 결과 변환
            aligned_subtitles = []
            for item in alignment_data:
                lyrics_idx = item.get('lyrics_line', 1) - 1  # 1-based to 0-based
                start_time = float(item.get('start_time', 0.0))
                end_time = float(item.get('end_time', start_time + 1.0))
                text = item.get('text', '')
                
                # 유효성 검사
                if (0 <= lyrics_idx < len(lyrics_lines) and 
                    end_time > start_time and 
                    text.strip()):
                    
                    aligned_subtitles.append({
                        'start': start_time,
                        'end': end_time,
                        'text': text.strip(),
                        'confidence': 0.6  # LLM 기본 신뢰도
                    })
            
            return aligned_subtitles
            
        except Exception as e:
            logger.error(f"LLM 응답 파싱 실패: {e}")
            return None
