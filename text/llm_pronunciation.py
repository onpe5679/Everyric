"""
LLM 기반 발음 변환 모듈
기존 pronunciation.py를 대체하여 LLM으로 발음 표기 생성
"""
import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class LLMPronunciationConverter:
    """LLM 기반 발음 변환기"""
    
    def __init__(self, 
                 engine: str = "gemini",  # "gpt" | "gemini"
                 openai_api_key: Optional[str] = None,
                 google_api_key: Optional[str] = None):
        self.engine = engine
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.google_api_key = google_api_key or os.getenv('GOOGLE_API_KEY')
        
        logger.info(f"🗣️ LLM 발음 변환기 초기화 - 엔진: {engine}")
        logger.debug(f"API 키 상태 - OpenAI: {'있음' if self.openai_api_key else '없음'}, Google: {'있음' if self.google_api_key else '없음'}")
    
    def add_pronunciation_to_subtitles(self, 
                                     subtitles: List[Dict], 
                                     source_lang: str = "auto") -> List[Dict]:
        """자막에 발음 표기 추가"""
        if not subtitles:
            return subtitles
        
        logger.info(f"🗣️ LLM 발음 변환 시작: {len(subtitles)}개 자막")
        
        # 텍스트만 추출
        texts = [sub.get('text', '') for sub in subtitles]
        
        try:
            # LLM으로 발음 변환
            pronunciations = self._convert_pronunciations(texts, source_lang)
            
            # 결과 적용
            result = []
            for i, sub in enumerate(subtitles):
                new_sub = dict(sub)
                if i < len(pronunciations) and pronunciations[i]:
                    new_sub['pronunciation'] = pronunciations[i]
                else:
                    new_sub['pronunciation'] = sub.get('text', '')
                result.append(new_sub)
            
            logger.info(f"✅ LLM 발음 변환 완료")
            return result
            
        except Exception as e:
            logger.error(f"LLM 발음 변환 실패: {e}")
            # 실패 시 원본 텍스트를 발음으로 사용
            result = []
            for sub in subtitles:
                new_sub = dict(sub)
                new_sub['pronunciation'] = sub.get('text', '')
                result.append(new_sub)
            return result
    
    def _convert_pronunciations(self, texts: List[str], source_lang: str) -> List[str]:
        """텍스트 리스트를 발음 표기로 변환"""
        if not texts:
            return []
        
        # 언어 감지 및 프롬프트 생성
        lang_info = self._detect_language_info(source_lang, texts[0] if texts else "")
        prompt = self._build_pronunciation_prompt(texts, lang_info)
        
        # LLM 호출
        try:
            if self.engine == 'gemini':
                response = self._call_gemini(prompt)
            else:
                response = self._call_gpt(prompt)
            
            if response:
                return self._parse_pronunciation_response(response, len(texts))
            
        except Exception as e:
            logger.error(f"LLM 발음 변환 호출 실패: {e}")
        
        return texts  # 실패 시 원본 반환
    
    def _detect_language_info(self, source_lang: str, sample_text: str) -> Dict[str, str]:
        """언어 정보 감지"""
        # 간단한 언어 감지
        if source_lang != "auto":
            lang = source_lang
        else:
            # 한글, 일본어, 중국어 등 감지
            if any('\u3040' <= c <= '\u309F' or '\u30A0' <= c <= '\u30FF' for c in sample_text):
                lang = "ja"  # 일본어
            elif any('\u4E00' <= c <= '\u9FFF' for c in sample_text):
                lang = "zh"  # 중국어
            elif any('\uAC00' <= c <= '\uD7AF' for c in sample_text):
                lang = "ko"  # 한국어
            else:
                lang = "en"  # 기본 영어
        
        lang_map = {
            "ja": {"name": "일본어", "target": "한글 발음"},
            "zh": {"name": "중국어", "target": "한글 발음"},
            "ko": {"name": "한국어", "target": "한글 발음"},
            "en": {"name": "영어", "target": "한글 발음"}
        }
        
        return lang_map.get(lang, {"name": "알 수 없음", "target": "한글 발음"})
    
    def _build_pronunciation_prompt(self, texts: List[str], lang_info: Dict[str, str]) -> str:
        """발음 변환 프롬프트 생성"""
        texts_list = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
        
        prompt = f"""
다음 {lang_info['name']} 텍스트들을 {lang_info['target']}로 변환해주세요.

텍스트 목록:
{texts_list}

변환 규칙:
1) 각 텍스트를 한국어 발음으로 정확하게 표기
2) 자연스러운 한국어 발음 표기 사용
3) 원문의 의미와 발음을 모두 고려
4) 번호 순서대로 한 줄씩 출력

출력 형식 (번호 없이 발음만):
첫 번째 텍스트의 한글 발음
두 번째 텍스트의 한글 발음
...
"""
        return prompt
    
    def _parse_pronunciation_response(self, response: str, expected_count: int) -> List[str]:
        """LLM 응답에서 발음 추출"""
        lines = response.strip().split('\n')
        pronunciations = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 번호 제거 (1. 2. 등)
            if line and line[0].isdigit():
                parts = line.split('.', 1)
                if len(parts) > 1:
                    line = parts[1].strip()
            
            pronunciations.append(line)
        
        # 개수 맞추기
        while len(pronunciations) < expected_count:
            pronunciations.append("")
        
        return pronunciations[:expected_count]
    
    def _call_gpt(self, prompt: str) -> Optional[str]:
        """GPT API 호출"""
        if not self.openai_api_key:
            raise RuntimeError("OpenAI API 키가 없습니다.")
        
        try:
            import openai
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 다국어 발음 변환 전문가입니다. 정확하고 자연스러운 한국어 발음 표기를 제공합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"GPT 호출 실패: {e}")
            raise
    
    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Gemini API 호출"""
        if not self.google_api_key:
            logger.error("Google(Gemini) API 키가 없습니다.")
            raise RuntimeError("Google(Gemini) API 키가 없습니다.")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.google_api_key)
            
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini 호출 실패: {e}")
            raise


# 기존 인터페이스 호환성을 위한 함수
def add_pronunciation_to_subtitles(subtitles: List[Dict], 
                                 source_lang: str = "auto",
                                 engine: str = "gemini") -> List[Dict]:
    """기존 pronunciation.py와 호환되는 함수"""
    converter = LLMPronunciationConverter(engine=engine)
    return converter.add_pronunciation_to_subtitles(subtitles, source_lang)
