"""
LLM 전용 가사 정렬기
- 입력: 가사 라인, ASR 세그먼트(start/end/text)
- 처리: LLM에게 전체 매칭을 요청하여 타임코드 부여
- 후처리: 무성구간 스냅(옵션) + 가드레일 규칙 적용
"""
import os
import re
import json
import logging
from typing import List, Dict, Optional, Tuple

from .post_rules import PostProcessingRules, snap_to_silence_ranges

logger = logging.getLogger(__name__)


def _build_prompt(lyrics_lines: List[str], segments: List[Dict], silence_ranges: Optional[List[Tuple[float, float]]] = None, max_len: int = 100) -> str:
    # 기존 가사 줄바꿈은 무시하고 하나의 문단으로 결합
    flat_lyrics = " ".join([line.strip() for line in lyrics_lines if line and line.strip()])

    # 세그먼트 타임코드 나열
    seg_lines = []
    total_duration = 0.0
    for seg in segments:
        try:
            s = float(seg.get('start', 0.0))
            e = float(seg.get('end', 0.0))
            total_duration = max(total_duration, e)
        except Exception:
            s, e = 0.0, 0.0
        t = (seg.get('text') or '').strip()
        seg_lines.append(f"[{s:.2f}-{e:.2f}s] {t}")
    segments_text = "\n".join(seg_lines)
    
    # 시간 평균 계산 (균등 분배 기준점)
    avg_time_per_line = total_duration / len(lyrics_lines) if lyrics_lines else 1.0
    
    # 무성구간 정보 추가
    silence_info = ""
    if silence_ranges:
        silence_list = [f"[{s:.2f}-{e:.2f}s] 무성구간" for s, e in silence_ranges]
        silence_info = f"\n무성구간 (가사 배치 금지):\n" + "\n".join(silence_list) + "\n"

    prompt = f"""
다음은 음성 인식 결과와 가사입니다. 가사는 원래 줄바꿈을 모두 무시하고 하나의 문단으로 취급하세요. 이 문단을 자연스러운 문장/구로 나누되, 각 조각의 최대 길이는 {max_len} 글자를 넘지 않도록 하세요(가능하면 더 짧게, 의미 단위 유지). 그 후 각 조각을 음성 인식 결과의 시간대에 매칭하여 JSON으로 출력하세요.

가사(단일 문단):
{flat_lyrics}

음성 인식 결과 (시간대별):
{segments_text}
{silence_info}
기준 정보:
- 전체 길이: {total_duration:.2f}초
- 가사 라인당 평균 시간: {avg_time_per_line:.2f}초

출력 형식:
{{
  "alignments": [
    {{"start_time": 0.00, "end_time": 3.20, "text": "조각1", "similarity": 85}},
    {{"start_time": 3.20, "end_time": 6.00, "text": "조각2", "similarity": 92}}
  ],
  "overall_similarity": 88.5
}}

중요 규칙:
1) 시간 순서 유지: start_time이 증가하는 순서, end_time <= next.start_time
2) 무성구간 회피: 무성구간에는 가사를 배치하지 마세요
3) 균등 분배 우선: 평균 {avg_time_per_line:.2f}초를 기준으로 하되, 유사도가 높은 구간을 우선 선택
4) 파트 끝 몰림 방지: 마지막 가사들이 한 구간에 몰리지 않도록 시간을 고르게 분배
5) 유사도 평가: 각 가사와 해당 구간 음성 인식 결과의 유사도를 0-100점으로 평가
6) 전체 유사도: 모든 매칭의 평균 유사도 계산
"""
    return prompt


def _parse_json_from_text(text: str) -> Optional[Tuple[List[Dict], float]]:
    """JSON 파싱 및 유사도 정보 추출"""
    try:
        # 새로운 형식 시도 (alignments + overall_similarity)
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if isinstance(data, dict) and 'alignments' in data:
                alignments = data.get('alignments', [])
                overall_similarity = data.get('overall_similarity', 0.0)
                return alignments, overall_similarity
        
        # 기존 형식 시도 (배열만)
        array_match = re.search(r"\[.*\]", text, re.DOTALL)
        if array_match:
            data = json.loads(array_match.group(0))
            if isinstance(data, list):
                return data, 0.0
        
        return None
    except Exception:
        return None


class LLMOnlyAligner:
    """LLM만 사용하는 가사 정렬기"""

    def __init__(self,
                 engine: str = "gpt",  # "gpt" | "gemini"
                 openai_api_key: Optional[str] = None,
                 google_api_key: Optional[str] = None,
                 alignment_config: Optional[Dict] = None):
        self.engine = engine
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.google_api_key = google_api_key or os.getenv('GOOGLE_API_KEY')

        cfg = alignment_config or {}
        # 텍스트 조각 최대 길이 설정 (config.max_length 전달 기대)
        self.max_len = int(cfg.get('max_length', 100))
        # 후처리 규칙
        self.post = PostProcessingRules(
            min_duration=cfg.get('min_duration', 0.3),
            max_duration=cfg.get('max_duration', 15.0),
            min_gap=cfg.get('min_gap', 0.1),
            track_start=cfg.get('track_start', 0.0),
            track_end=cfg.get('track_end', None)
        )
        # VAD 스냅
        self.enable_vad_snap = cfg.get('enable_vad_snap', True)
        self.vad_snap_window = cfg.get('vad_snap_window', 0.5)

    def align_lyrics_with_timing(self,
                                 lyrics_lines: List[str],
                                 segments: List[Dict],
                                 silence_ranges: Optional[List[Tuple[float, float]]] = None) -> List[Dict]:
        if not lyrics_lines or not segments:
            return []

        prompt = _build_prompt(lyrics_lines, segments, silence_ranges, max_len=self.max_len)
        raw = None
        try:
            if self.engine == 'gemini':
                raw = self._call_gemini(prompt)
            else:
                raw = self._call_gpt(prompt)
        except Exception as e:
            logger.error(f"LLM 정렬 호출 실패: {e}")
            return []

        parsed_result = _parse_json_from_text(raw or '') if isinstance(raw, str) else None
        if not parsed_result:
            logger.warning("LLM 응답 파싱 실패")
            return []

        parsed, overall_similarity = parsed_result

        # 유사도 정보 로깅
        if overall_similarity > 0:
            logger.info(f"🎯 LLM 정렬 전체 유사도: {overall_similarity:.1f}%")
        
        # JSON → subtitles
        subs: List[Dict] = []
        individual_similarities = []
        
        for item in parsed:
            try:
                # lyrics_line이 없을 수 있으므로 안전 처리
                s = float(item.get('start_time', 0.0))
                e = float(item.get('end_time', s + 1.0))
                txt_raw = (item.get('text') or '').strip()
                if not txt_raw and item.get('lyrics_line') is not None:
                    try:
                        idx = int(item.get('lyrics_line')) - 1
                        if 0 <= idx < len(lyrics_lines):
                            txt_raw = lyrics_lines[idx].strip()
                    except Exception:
                        pass
                txt = txt_raw
                similarity = float(item.get('similarity', 0))
                
                if e <= s or not txt:
                    continue
                    
                subs.append({
                    'start': s, 
                    'end': e, 
                    'text': txt, 
                    'confidence': similarity / 100.0,  # 유사도를 confidence로 사용
                    'similarity': similarity
                })
                individual_similarities.append(similarity)
            except Exception:
                continue

        if not subs:
            return []

        # 개별 유사도 통계 로깅
        if individual_similarities:
            avg_sim = sum(individual_similarities) / len(individual_similarities)
            min_sim = min(individual_similarities)
            max_sim = max(individual_similarities)
            logger.info(f"📊 개별 유사도 - 평균: {avg_sim:.1f}%, 범위: {min_sim:.1f}%-{max_sim:.1f}%")

        # VAD 스냅 비활성화 (LLM이 이미 무성구간을 고려했으므로)
        # if self.enable_vad_snap and silence_ranges:
        #     subs = snap_to_silence_ranges(subs, silence_ranges, self.vad_snap_window)

        # 트랙 종료 추정 후 가드레일 적용
        try:
            track_end = max(float(seg.get('end', 0.0)) for seg in segments) + 1.0
            self.post.track_end = track_end
        except Exception:
            pass
        subs = self.post.apply_all_rules(subs)
        
        # 최종 결과에 유사도 정보 저장 (diagnostics용)
        if hasattr(self, '_last_similarity_info'):
            delattr(self, '_last_similarity_info')
        self._last_similarity_info = {
            'overall': overall_similarity,
            'individual': individual_similarities,
            'average': sum(individual_similarities) / len(individual_similarities) if individual_similarities else 0
        }
        
        return subs

    def _call_gpt(self, prompt: str) -> Optional[str]:
        api_key = self.openai_api_key
        if not api_key:
            raise RuntimeError("OpenAI API 키가 없습니다.")
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 가사와 음성 인식 결과를 정밀하게 정렬하는 전문가입니다."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI 호출 실패: {e}")
            raise

    def _call_gemini(self, prompt: str) -> Optional[str]:
        api_key = self.google_api_key
        if not api_key:
            raise RuntimeError("Google(Gemini) API 키가 없습니다.")
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            resp = model.generate_content(prompt)
            # 문자열로 반환
            return resp.text
        except Exception as e:
            logger.error(f"Gemini 호출 실패: {e}")
            raise
