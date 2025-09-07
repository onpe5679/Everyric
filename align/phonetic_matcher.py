import re
import logging
from typing import List, Dict, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class PhoneticMatcher:
    """발음 기반 텍스트 유사도 매칭 클래스"""
    
    def __init__(self):
        # 일본어 히라가나-카타카나 매핑
        self.hiragana_to_katakana = str.maketrans(
            'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん'
            'がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽゃゅょっ',
            'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン'
            'ガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポャュョッ'
        )
        
        # 한국어 자음/모음 유사도 매핑
        self.korean_consonant_groups = {
            'ㄱ': ['ㄱ', 'ㅋ', 'ㄲ'],
            'ㄴ': ['ㄴ'],
            'ㄷ': ['ㄷ', 'ㅌ', 'ㄸ'],
            'ㄹ': ['ㄹ'],
            'ㅁ': ['ㅁ'],
            'ㅂ': ['ㅂ', 'ㅍ', 'ㅃ'],
            'ㅅ': ['ㅅ', 'ㅆ'],
            'ㅇ': ['ㅇ'],
            'ㅈ': ['ㅈ', 'ㅊ', 'ㅉ'],
            'ㅎ': ['ㅎ']
        }
        
        self.korean_vowel_groups = {
            'ㅏ': ['ㅏ', 'ㅑ'],
            'ㅓ': ['ㅓ', 'ㅕ'],
            'ㅗ': ['ㅗ', 'ㅛ', 'ㅜ', 'ㅠ'],
            'ㅡ': ['ㅡ', 'ㅣ'],
            'ㅐ': ['ㅐ', 'ㅒ', 'ㅔ', 'ㅖ']
        }
    
    def normalize_japanese(self, text: str) -> str:
        """일본어 텍스트 정규화 - 히라가나를 카타카나로 통일"""
        # 히라가나를 카타카나로 변환
        normalized = text.translate(self.hiragana_to_katakana)
        # 장음 기호 정규화
        normalized = re.sub(r'ー+', 'ー', normalized)
        # 작은 문자 정규화
        normalized = re.sub(r'[ャュョッ]', lambda m: m.group().replace('ャ', 'ヤ').replace('ュ', 'ユ').replace('ョ', 'ヨ').replace('ッ', ''), normalized)
        return normalized
    
    def normalize_korean(self, text: str) -> str:
        """한국어 텍스트 정규화 - 자모 분리 및 유사음 그룹화"""
        try:
            # 한글을 자모로 분리
            decomposed = []
            for char in text:
                if '가' <= char <= '힣':  # 한글 음절
                    # 초성, 중성, 종성 분리
                    code = ord(char) - ord('가')
                    initial = code // (21 * 28)
                    medial = (code % (21 * 28)) // 28
                    final = code % 28
                    
                    # 초성 추가
                    initial_chars = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 
                                   'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
                    decomposed.append(initial_chars[initial])
                    
                    # 중성 추가
                    medial_chars = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 
                                  'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
                    decomposed.append(medial_chars[medial])
                    
                    # 종성 추가 (있는 경우)
                    if final > 0:
                        final_chars = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 
                                     'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 
                                     'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
                        decomposed.append(final_chars[final])
                else:
                    decomposed.append(char)
            
            return ''.join(decomposed)
        except:
            return text
    
    def calculate_phonetic_similarity(self, text1: str, text2: str) -> float:
        """발음 기반 유사도 계산"""
        if not text1 or not text2:
            return 0.0
        
        # 언어 감지 및 정규화
        normalized1 = self._normalize_text(text1)
        normalized2 = self._normalize_text(text2)
        
        # 기본 문자열 유사도
        basic_similarity = SequenceMatcher(None, normalized1, normalized2).ratio()
        
        # 음성학적 유사도 보정
        phonetic_bonus = self._calculate_phonetic_bonus(text1, text2)
        
        # 최종 유사도 (기본 유사도 + 음성학적 보너스)
        final_similarity = min(1.0, basic_similarity + phonetic_bonus)
        
        return final_similarity
    
    def _normalize_text(self, text: str) -> str:
        """텍스트 정규화 (언어별)"""
        # 공백 및 특수문자 제거
        text = re.sub(r'[^\w가-힣ぁ-んァ-ヶー一-龯]', '', text).lower()
        
        # 일본어 정규화
        if re.search(r'[ぁ-んァ-ヶー]', text):
            text = self.normalize_japanese(text)
        
        # 한국어 정규화
        if re.search(r'[가-힣]', text):
            text = self.normalize_korean(text)
        
        return text
    
    def _calculate_phonetic_bonus(self, text1: str, text2: str) -> float:
        """음성학적 유사도 보너스 계산"""
        bonus = 0.0
        
        # 일본어 음성학적 유사도
        if re.search(r'[ぁ-んァ-ヶー]', text1) or re.search(r'[ぁ-んァ-ヶー]', text2):
            bonus += self._japanese_phonetic_similarity(text1, text2)
        
        # 한국어 음성학적 유사도
        if re.search(r'[가-힣]', text1) or re.search(r'[가-힣]', text2):
            bonus += self._korean_phonetic_similarity(text1, text2)
        
        return min(0.3, bonus)  # 최대 30% 보너스
    
    def _japanese_phonetic_similarity(self, text1: str, text2: str) -> float:
        """일본어 음성학적 유사도"""
        # 장음, 촉음 등의 유사성 검사
        similarity = 0.0
        
        # 장음 기호 유사성
        if 'ー' in text1 and 'ー' in text2:
            similarity += 0.1
        
        # 탁음/반탁음 유사성 (가/카, 바/파 등)
        voiced_pairs = [
            ('が', 'か'), ('ぎ', 'き'), ('ぐ', 'く'), ('げ', 'け'), ('ご', 'こ'),
            ('ざ', 'さ'), ('じ', 'し'), ('ず', 'す'), ('ぜ', 'せ'), ('ぞ', 'そ'),
            ('だ', 'た'), ('ぢ', 'ち'), ('づ', 'つ'), ('で', 'て'), ('ど', 'と'),
            ('ば', 'は'), ('び', 'ひ'), ('ぶ', 'ふ'), ('べ', 'へ'), ('ぼ', 'ほ'),
            ('ぱ', 'は'), ('ぴ', 'ひ'), ('ぷ', 'ふ'), ('ぺ', 'へ'), ('ぽ', 'ほ')
        ]
        
        for voiced, unvoiced in voiced_pairs:
            if (voiced in text1 and unvoiced in text2) or (unvoiced in text1 and voiced in text2):
                similarity += 0.05
        
        return similarity
    
    def _korean_phonetic_similarity(self, text1: str, text2: str) -> float:
        """한국어 음성학적 유사도"""
        similarity = 0.0
        
        # 자음 유사성 검사
        for group in self.korean_consonant_groups.values():
            count1 = sum(text1.count(char) for char in group)
            count2 = sum(text2.count(char) for char in group)
            if count1 > 0 and count2 > 0:
                similarity += 0.05
        
        # 모음 유사성 검사
        for group in self.korean_vowel_groups.values():
            count1 = sum(text1.count(char) for char in group)
            count2 = sum(text2.count(char) for char in group)
            if count1 > 0 and count2 > 0:
                similarity += 0.05
        
        return similarity
