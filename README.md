# Everyric

유튜브 영상의 노래 및 사용자 입력 가사를 기반으로 문장 단위 타이밍 자막과 다국어 번역/발음 병기 자막을 자동 생성하는 도구입니다.

## 디렉토리 구조
```
everyric/
├─ audio/           # 오디오 처리 모듈
├─ text/            # 가사 파싱·번역·발음 모듈
├─ align/           # 타이밍 정렬 알고리즘
├─ output/          # 자막 생성(.json/.srt/.ass)
├─ cli.py           # 엔트리 포인트 CLI
├─ requirements.txt # 의존성 목록
└─ README.md        # 프로젝트 설명

tests/              # 단위 테스트
```
