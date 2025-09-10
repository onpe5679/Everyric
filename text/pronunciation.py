import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class KoreanPronunciationConverter:
    """원어 텍스트를 한글 발음으로 변환하는 간단한 클래스"""
    
    def __init__(self):
        self.logger = logger
        
        # 간단한 일본어 → 한글 매핑 (자주 사용되는 것만)
        self.jp_to_kr = {
            # 기본 히라가나
            'あ': '아', 'い': '이', 'う': '우', 'え': '에', 'お': '오',
            'か': '카', 'き': '키', 'く': '쿠', 'け': '케', 'こ': '코',
            'が': '가', 'ぎ': '기', 'ぐ': '구', 'げ': '게', 'ご': '고',
            'さ': '사', 'し': '시', 'す': '스', 'せ': '세', 'そ': '소',
            'ざ': '자', 'じ': '지', 'ず': '즈', 'ぜ': '제', 'ぞ': '조',
            'た': '타', 'ち': '치', 'つ': '츠', 'て': '테', 'と': '토',
            'だ': '다', 'で': '데', 'ど': '도',
            'な': '나', 'に': '니', 'ぬ': '누', 'ね': '네', 'の': '노',
            'は': '하', 'ひ': '히', 'ふ': '후', 'へ': '헤', 'ほ': '호',
            'ば': '바', 'び': '비', 'ぶ': '부', 'べ': '베', 'ぼ': '보',
            'ぱ': '파', 'ぴ': '피', 'ぷ': '푸', 'ぺ': '페', 'ぽ': '포',
            'ま': '마', 'み': '미', 'む': '무', 'め': '메', 'も': '모',
            'や': '야', 'ゆ': '유', 'よ': '요',
            'ら': '라', 'り': '리', 'る': '루', 'れ': '레', 'ろ': '로',
            'わ': '와', 'を': '오', 'ん': '응',
            
            # 가타카나 (동일한 발음)
            'ア': '아', 'イ': '이', 'ウ': '우', 'エ': '에', 'オ': '오',
            'カ': '카', 'キ': '키', 'ク': '쿠', 'ケ': '케', 'コ': '코',
            'ガ': '가', 'ギ': '기', 'グ': '구', 'ゲ': '게', 'ゴ': '고',
            'サ': '사', 'シ': '시', 'ス': '스', 'セ': '세', 'ソ': '소',
            'ザ': '자', 'ジ': '지', 'ズ': '즈', 'ゼ': '제', 'ゾ': '조',
            'タ': '타', 'チ': '치', 'ツ': '츠', 'テ': '테', 'ト': '토',
            'ダ': '다', 'デ': '데', 'ド': '도',
            'ナ': '나', 'ニ': '니', 'ヌ': '누', 'ネ': '네', 'ノ': '노',
            'ハ': '하', 'ヒ': '히', 'フ': '후', 'ヘ': '헤', 'ホ': '호',
            'バ': '바', 'ビ': '비', 'ブ': '부', 'ベ': '베', 'ボ': '보',
            'パ': '파', 'ピ': '피', 'プ': '푸', 'ペ': '페', 'ポ': '포',
            'マ': '마', 'ミ': '미', 'ム': '무', 'メ': '메', 'モ': '모',
            'ヤ': '야', 'ユ': '유', 'ヨ': '요',
            'ラ': '라', 'リ': '리', 'ル': '루', 'レ': '레', 'ロ': '로',
            'ワ': '와', 'ヲ': '오', 'ン': '응'
        }
        
        # 간단한 영어 → 한글 매핑
        self.en_to_kr = {
            'a': '아', 'b': '비', 'c': '시', 'd': '디', 'e': '이',
            'f': '에프', 'g': '지', 'h': '에이치', 'i': '아이', 'j': '제이',
            'k': '케이', 'l': '엘', 'm': '엠', 'n': '엔', 'o': '오',
            'p': '피', 'q': '큐', 'r': '알', 's': '에스', 't': '티',
            'u': '유', 'v': '브이', 'w': '더블유', 'x': '엑스', 'y': '와이', 'z': '지'
        }
    
    def convert_to_korean_pronunciation(self, text: str, source_lang: str = 'auto') -> str:
        """
        원어 텍스트를 한글 발음으로 변환합니다.
        
        Args:
            text: 변환할 텍스트
            source_lang: 원본 언어 ('en', 'ja', 'auto')
            
        Returns:
            한글 발음으로 변환된 텍스트
        """
        if not text.strip():
            return text
            
        # 언어 자동 감지
        if source_lang == 'auto':
            source_lang = self._detect_language(text)
        
        self.logger.debug(f"발음 변환: '{text}' ({source_lang} → 한글)")
        
        if source_lang == 'ja':
            return self._convert_japanese_to_korean(text)
        elif source_lang == 'en':
            return self._convert_english_to_korean(text)
        else:
            # 지원하지 않는 언어는 원문 반환
            return text
    
    def _detect_language(self, text: str) -> str:
        """텍스트의 언어를 자동 감지합니다."""
        # 일본어 문자 (히라가나, 가타카나) 확인
        japanese_chars = re.findall(r'[ひらがなカタカナぁ-んァ-ヶー]', text)
        if japanese_chars:
            return 'ja'
        
        # 영어 문자 확인
        english_chars = re.findall(r'[a-zA-Z]', text)
        if english_chars:
            return 'en'
        
        # 기본값은 일본어
        return 'ja'
    
    def _convert_japanese_to_korean(self, text: str) -> str:
        """일본어 텍스트를 한글 발음으로 변환합니다."""
        result = ""
        
        for char in text:
            if char in self.jp_to_kr:
                result += self.jp_to_kr[char]
            elif char.isspace() or char in '.,!?':
                result += char
            # 변환할 수 없는 문자는 제거 (한글만 출력)
        
        return result
    
    def _convert_english_to_korean(self, text: str) -> str:
        """영어 텍스트를 한글 발음으로 변환합니다."""
        result = ""
        
        for char in text.lower():
            if char in self.en_to_kr:
                result += self.en_to_kr[char]
            elif char.isspace() or char in '.,!?':
                result += char
            # 변환할 수 없는 문자는 제거
        
        return result

def convert_pronunciation(text: str, lang: str = 'auto') -> str:
    """
    텍스트를 한글 발음으로 변환합니다.
    
    Args:
        text: 변환할 텍스트
        lang: 원본 언어 ('en', 'ja', 'auto')
        
    Returns:
        한글 발음으로 변환된 텍스트
    """
    converter = KoreanPronunciationConverter()
    return converter.convert_to_korean_pronunciation(text, lang)
