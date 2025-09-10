import os
import time
import logging
from typing import List, Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.getLogger(__name__).warning("Google Generative AI 라이브러리가 설치되지 않았습니다. Gemini 번역을 사용하려면 'pip install google-generativeai'를 실행하세요.")

logger = logging.getLogger(__name__)

class GPTTranslator:
    """OpenAI GPT API를 이용한 가사 번역 클래스"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo", 
                 max_retries: int = 5, retry_delay: int = 2):
        """
        Args:
            api_key: OpenAI API 키 (없으면 환경변수에서 가져옴)
            model: 사용할 GPT 모델
            max_retries: 최대 재시도 횟수
            retry_delay: 재시도 기본 대기 시간
        """
        self.logger = logger
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if not self.api_key:
            raise ValueError("OpenAI API 키가 필요합니다. 환경변수 OPENAI_API_KEY를 설정하거나 직접 전달해주세요.")
        
        self.logger.info(f"🤖 GPT 번역기 초기화 - 모델: {self.model}, 최대 재시도: {self.max_retries}회")
        
        # OpenAI 클라이언트 초기화 (v1.0+ 호환)
        try:
            # OpenAI v1.0+ 방식
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.use_v1_api = True
            self.logger.debug("✅ OpenAI v1.0+ 클라이언트 초기화 완료")
        except ImportError:
            # 구버전 방식 폴백
            openai.api_key = self.api_key
            self.use_v1_api = False
            self.logger.debug("✅ OpenAI 구버전 클라이언트 초기화 완료")
        
    def translate_lyrics(self, lyrics: List[str], target_lang: str = 'ko', 
                        source_lang: str = 'auto', context: str = '') -> List[str]:
        """
        가사를 번역합니다.
        
        Args:
            lyrics: 번역할 가사 리스트
            target_lang: 목표 언어 ('ko', 'en', 'ja' 등)
            source_lang: 원본 언어 ('auto', 'en', 'ja' 등)
            context: 번역 컨텍스트 (곡명, 아티스트 등)
            
        Returns:
            번역된 가사 리스트
        """
        if not lyrics:
            return []
            
        self.logger.info(f"🌐 GPT 번역 시작 - {len(lyrics)}줄 ({source_lang} → {target_lang})")
        self.logger.debug(f"📝 번역 컨텍스트: {context if context else '없음'}")
        
        # 번역 프롬프트 생성
        prompt = self._create_translation_prompt(lyrics, target_lang, source_lang, context)
        self.logger.debug(f"📄 생성된 프롬프트 길이: {len(prompt)}자")
        
        try:
            # GPT API 호출
            self.logger.debug("🔄 GPT API 호출 시작...")
            response = self._call_gpt_api(prompt)
            self.logger.debug(f"📥 GPT 응답 길이: {len(response)}자")
            
            # 응답 파싱
            translated_lyrics = self._parse_translation_response(response, len(lyrics))
            
            self.logger.info(f"✅ 번역 완료 - {len(translated_lyrics)}줄")
            return translated_lyrics
            
        except Exception as e:
            self.logger.error(f"❌ 번역 실패: {str(e)}")
            self.logger.error(f"🔍 오류 타입: {type(e).__name__}")
            if hasattr(e, 'response'):
                self.logger.error(f"📡 API 응답 상태: {getattr(e.response, 'status_code', 'N/A')}")
            # 실패 시 원본 반환
            return lyrics
    
    def _create_translation_prompt(self, lyrics: List[str], target_lang: str, 
                                 source_lang: str, context: str) -> str:
        """번역 프롬프트를 생성합니다."""
        
        lang_names = {
            'ko': '한국어',
            'en': '영어', 
            'ja': '일본어',
            'auto': '자동감지'
        }
        
        target_lang_name = lang_names.get(target_lang, target_lang)
        source_lang_name = lang_names.get(source_lang, source_lang)
        
        # 가사를 번호와 함께 포맷팅
        formatted_lyrics = '\n'.join([f"{i+1}. {line}" for i, line in enumerate(lyrics)])
        
        prompt = f"""다음은 음악 가사입니다. 각 줄을 {target_lang_name}로 자연스럽게 번역해주세요.

번역 지침:
1. 가사의 의미와 감정을 보존하면서 자연스러운 {target_lang_name}로 번역
2. 음악적 리듬감을 고려하여 번역
3. 각 줄의 번호를 유지하여 응답
4. 원본과 동일한 줄 수로 번역
5. 빈 줄이나 반복 구간도 그대로 유지

{f"컨텍스트: {context}" if context else ""}

원본 가사 ({source_lang_name}):
{formatted_lyrics}

번역된 가사 ({target_lang_name}):"""

        return prompt
    
    def _call_gpt_api(self, prompt: str) -> str:
        """GPT API를 호출합니다."""
        
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"🔄 GPT API 호출 시도 {attempt + 1}/{self.max_retries} (모델: {self.model})")
                
                if self.use_v1_api:
                    # OpenAI v1.0+ 방식
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "당신은 전문 번역가입니다. 음악 가사를 자연스럽고 감정적으로 번역하는 것이 특기입니다."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=2000,
                        temperature=0.3,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0
                    )
                    result = response.choices[0].message.content.strip()
                    self.logger.info(f"✅ API 호출 성공 (시도 {attempt + 1}회)")
                    return result
                else:
                    # 구버전 방식
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "당신은 전문 번역가입니다. 음악 가사를 자연스럽고 감정적으로 번역하는 것이 특기입니다."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=2000,
                        temperature=0.3,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0
                    )
                    result = response.choices[0].message.content.strip()
                    self.logger.info(f"✅ API 호출 성공 (시도 {attempt + 1}회)")
                    return result
                
            except Exception as api_error:
                # 상세한 오류 정보 로깅
                error_str = str(api_error)
                error_type = type(api_error).__name__
                
                self.logger.error(f"❌ API 호출 실패 (시도 {attempt + 1}/{self.max_retries})")
                self.logger.error(f"🔍 오류 타입: {error_type}")
                self.logger.error(f"📝 오류 메시지: {error_str}")
                
                # 특정 오류 타입별 처리
                if "rate_limit" in error_str.lower() or "429" in error_str or "RateLimitError" in error_type:
                    wait_time = self.retry_delay * (2 ** attempt)  # 지수 백오프
                    self.logger.warning(f"⏳ API 속도 제한 - {wait_time}초 대기 후 재시도")
                    time.sleep(wait_time)
                elif "quota" in error_str.lower() or "billing" in error_str.lower():
                    self.logger.error("💳 API 할당량 초과 또는 결제 문제")
                    raise api_error
                elif "authentication" in error_str.lower() or "401" in error_str:
                    self.logger.error("🔐 API 키 인증 실패")
                    raise api_error
                elif "model" in error_str.lower() and "not found" in error_str.lower():
                    self.logger.error(f"🤖 모델 '{self.model}'을 찾을 수 없음")
                    raise api_error
                elif "timeout" in error_str.lower():
                    wait_time = self.retry_delay * (attempt + 1)
                    self.logger.warning(f"⏰ 타임아웃 - {wait_time}초 대기 후 재시도")
                    time.sleep(wait_time)
                else:
                    # 기타 오류
                    self.logger.error(f"🚨 알 수 없는 API 오류: {error_str}")
                    if attempt == self.max_retries - 1:
                        raise api_error
                    time.sleep(self.retry_delay)
                
        
        raise Exception(f"GPT API 호출 최대 재시도 횟수 ({self.max_retries}회) 초과")
    
    def _parse_translation_response(self, response: str, expected_lines: int) -> List[str]:
        """GPT 응답을 파싱하여 번역된 가사 리스트를 반환합니다."""
        
        lines = response.strip().split('\n')
        translated_lyrics = []
        
        for line in lines:
            line = line.strip()
            if not line:
                translated_lyrics.append('')
                continue
                
            # 번호 제거 (1., 2., 등)
            if line and line[0].isdigit():
                # "1. 번역된 텍스트" 형태에서 번호 부분 제거
                parts = line.split('.', 1)
                if len(parts) > 1:
                    translated_lyrics.append(parts[1].strip())
                else:
                    translated_lyrics.append(line)
            else:
                translated_lyrics.append(line)
        
        # 예상 줄 수와 맞지 않으면 조정
        if len(translated_lyrics) != expected_lines:
            self.logger.warning(f"번역 결과 줄 수 불일치: 예상 {expected_lines}, 실제 {len(translated_lyrics)}")
            
            # 부족한 경우 빈 줄 추가
            while len(translated_lyrics) < expected_lines:
                translated_lyrics.append('')
            
            # 초과한 경우 자르기
            translated_lyrics = translated_lyrics[:expected_lines]
        
        return translated_lyrics

def translate_lyrics_with_gpt(lyrics: List[str], target_lang: str = 'ko', 
                             source_lang: str = 'auto', context: str = '',
                             api_key: Optional[str] = None, model: str = "gpt-3.5-turbo",
                             max_retries: int = 5, retry_delay: int = 2) -> List[str]:
    """
    편의 함수: GPT를 이용한 가사 번역
    
    Args:
        lyrics: 번역할 가사 리스트
        target_lang: 목표 언어
        source_lang: 원본 언어
        context: 번역 컨텍스트
        api_key: OpenAI API 키
        model: 사용할 GPT 모델
        max_retries: 최대 재시도 횟수
        retry_delay: 재시도 기본 대기 시간
        
    Returns:
        번역된 가사 리스트
    """
    translator = GPTTranslator(api_key, model, max_retries, retry_delay)
    return translator.translate_lyrics(lyrics, target_lang, source_lang, context)


class GeminiTranslator:
    """Google Gemini API를 이용한 가사 번역 클래스"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash", 
                 max_retries: int = 5, retry_delay: int = 2):
        """
        Args:
            api_key: Google API 키 (환경변수 GOOGLE_API_KEY에서 자동 로드)
            model: 사용할 Gemini 모델 (기본값: gemini-1.5-flash)
            max_retries: 최대 재시도 횟수
            retry_delay: 재시도 간격 (초)
        """
        self.logger = logger
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI 라이브러리가 필요합니다. 'pip install google-generativeai'를 실행하세요.")
        
        if not self.api_key:
            self.logger.error("❌ Google API 키가 설정되지 않았습니다.")
            self.logger.info("💡 환경변수 GOOGLE_API_KEY를 설정하거나 config.json에서 설정하세요.")
            raise ValueError("Google API 키가 필요합니다.")
        
        try:
            # Gemini API 설정
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
            self.logger.info(f"✅ Gemini API 클라이언트 초기화 완료 (모델: {self.model})")
        except Exception as e:
            self.logger.error(f"❌ Gemini API 클라이언트 초기화 실패: {str(e)}")
            raise
    
    def _create_translation_prompt(self, lyrics: List[str], target_lang: str, 
                                 source_lang: str, context: str) -> str:
        """번역 프롬프트 생성"""
        lang_names = {
            'ko': '한국어',
            'en': '영어', 
            'ja': '일본어',
            'zh': '중국어',
            'es': '스페인어',
            'fr': '프랑스어',
            'de': '독일어',
            'auto': '자동 감지'
        }
        
        target_lang_name = lang_names.get(target_lang, target_lang)
        source_lang_name = lang_names.get(source_lang, source_lang)
        
        context_info = f"\n\n추가 컨텍스트: {context}" if context else ""
        
        lyrics_text = '\n'.join([f"{i+1}. {line}" for i, line in enumerate(lyrics)])
        
        prompt = f"""다음 가사를 {target_lang_name}로 번역해주세요.

원본 언어: {source_lang_name}
번역 대상 언어: {target_lang_name}

번역 지침:
1. 가사의 의미와 감정을 자연스럽게 전달하세요
2. 운율이나 리듬보다는 의미 전달을 우선하세요
3. 각 줄을 정확히 번역하되, 자연스러운 {target_lang_name} 표현을 사용하세요
4. 번역된 가사만 출력하고, 번호나 추가 설명은 포함하지 마세요
5. 원본과 같은 줄 수를 유지하세요{context_info}

원본 가사:
{lyrics_text}

번역된 가사:"""
        
        return prompt
    
    def _call_gemini_api(self, prompt: str) -> str:
        """Gemini API 호출 (재시도 로직 포함)"""
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"🔄 Gemini API 호출 시도 {attempt + 1}/{self.max_retries} (모델: {self.model})")
                self.logger.debug(f"📝 프롬프트 길이: {len(prompt)} 문자")
                
                # Gemini API 호출
                response = self.client.generate_content(prompt)
                
                if response.text:
                    self.logger.info(f"✅ API 호출 성공 (응답 길이: {len(response.text)} 문자)")
                    self.logger.debug(f"📄 응답 내용 미리보기: {response.text[:100]}...")
                    return response.text.strip()
                else:
                    raise ValueError("빈 응답을 받았습니다.")
                    
            except Exception as api_error:
                error_type = type(api_error).__name__
                error_str = str(api_error)
                
                self.logger.error(f"❌ API 호출 실패 (시도 {attempt + 1}/{self.max_retries})")
                self.logger.error(f"🔍 오류 타입: {error_type}")
                self.logger.error(f"📝 오류 메시지: {error_str}")
                
                # 특정 오류에 대한 처리
                if "quota" in error_str.lower() or "billing" in error_str.lower():
                    self.logger.error("💳 API 할당량 초과 또는 결제 문제입니다.")
                    break
                elif "api_key" in error_str.lower() or "authentication" in error_str.lower():
                    self.logger.error("🔑 API 키 인증 문제입니다.")
                    break
                elif "model" in error_str.lower() and "not found" in error_str.lower():
                    self.logger.error(f"🤖 모델 '{self.model}'을 찾을 수 없습니다.")
                    break
                
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # 지수 백오프
                    self.logger.info(f"⏳ {wait_time}초 후 재시도...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"❌ 모든 재시도 실패. 번역을 포기합니다.")
                    raise api_error
        
        raise Exception("최대 재시도 횟수 초과")
    
    def translate_lyrics(self, lyrics: List[str], target_lang: str = 'ko', 
                        source_lang: str = 'auto', context: str = '') -> List[str]:
        """
        가사를 번역합니다.
        
        Args:
            lyrics: 번역할 가사 리스트
            target_lang: 번역 대상 언어 (기본값: 'ko')
            source_lang: 원본 언어 (기본값: 'auto')
            context: 추가 컨텍스트 정보
            
        Returns:
            번역된 가사 리스트
        """
        if not lyrics:
            self.logger.warning("번역할 가사가 없습니다.")
            return []
        
        self.logger.info(f"🌐 Gemini를 이용한 가사 번역 시작 ({len(lyrics)}줄)")
        self.logger.info(f"📋 번역 설정: {source_lang} → {target_lang}")
        
        try:
            # 번역 프롬프트 생성
            prompt = self._create_translation_prompt(lyrics, target_lang, source_lang, context)
            
            # API 호출
            translated_text = self._call_gemini_api(prompt)
            
            # 응답 파싱
            translated_lines = [line.strip() for line in translated_text.split('\n') if line.strip()]
            
            # 줄 수 검증
            if len(translated_lines) != len(lyrics):
                self.logger.warning(f"⚠️ 번역된 줄 수({len(translated_lines)})가 원본({len(lyrics)})과 다릅니다.")
                # 줄 수를 맞추기 위한 보정
                if len(translated_lines) < len(lyrics):
                    # 부족한 경우 빈 줄 추가
                    translated_lines.extend([''] * (len(lyrics) - len(translated_lines)))
                else:
                    # 초과한 경우 자르기
                    translated_lines = translated_lines[:len(lyrics)]
            
            self.logger.info(f"✅ 번역 완료: {len(translated_lines)}줄")
            
            # 번역 결과 로깅 (처음 3줄만)
            for i, (original, translated) in enumerate(zip(lyrics[:3], translated_lines[:3])):
                self.logger.debug(f"번역 {i+1}: '{original}' → '{translated}'")
            
            return translated_lines
            
        except Exception as e:
            self.logger.error(f"❌ 번역 중 오류 발생: {str(e)}")
            raise


def translate_lyrics_with_gemini(lyrics: List[str], target_lang: str = 'ko', 
                               source_lang: str = 'auto', context: str = '',
                               api_key: Optional[str] = None, model: str = "gemini-1.5-flash",
                               max_retries: int = 5, retry_delay: int = 2) -> List[str]:
    """
    Google Gemini를 이용한 가사 번역 편의 함수
    
    Args:
        lyrics: 번역할 가사 리스트
        target_lang: 번역 대상 언어 (기본값: 'ko')
        source_lang: 원본 언어 (기본값: 'auto')
        context: 추가 컨텍스트 정보
        api_key: Google API 키
        model: 사용할 Gemini 모델
        max_retries: 최대 재시도 횟수
        retry_delay: 재시도 간격
        
    Returns:
        번역된 가사 리스트
    """
    translator = GeminiTranslator(api_key, model, max_retries, retry_delay)
    return translator.translate_lyrics(lyrics, target_lang, source_lang, context)
