import os
import logging
from typing import List, Union

# 로거 설정
logger = logging.getLogger(__name__)

def parse_lyrics(source: str) -> List[str]:
    """
    가사 입력 처리: 파일 경로면 읽어서 라인 리스트 반환, 아니면 줄바꿈으로 분리
    
    Args:
        source: 파일 경로 또는 가사 텍스트
        
    Returns:
        List[str]: 전처리된 가사 라인 리스트
    """
    try:
        if os.path.isfile(source):
            logger.info(f"📄 가사 파일 로드 중: {source}")
            with open(source, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            logger.debug(f"파일에서 {len(lines)}줄의 가사를 로드했습니다.")
        else:
            logger.info("📝 인라인 가사 파싱")
            lines = [line.strip() for line in source.splitlines() if line.strip()]
            logger.debug(f"텍스트에서 {len(lines)}줄의 가사를 추출했습니다.")
        
        if not lines:
            logger.warning("⚠️ 비어있는 가사가 감지되었습니다.")
            
        return lines
        
    except UnicodeDecodeError as e:
        logger.error(f"❌ 파일 인코딩 오류: {source} (UTF-8 인코딩이 필요합니다)")
        raise
    except Exception as e:
        logger.error(f"❌ 가사 파싱 중 오류 발생: {str(e)}")
        raise
