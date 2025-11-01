import streamlit as st
import eng_to_ipa as ipa
import numpy as np
from scipy.spatial.distance import cosine
import json # JSON 출력을 보기 좋게 표시하기 위해 필요합니다.

# =========================================================
# 1. 음소 임베딩 및 데이터 정의 (rhyme_api.py에서 이식)
# =========================================================

# (예시) 근사 라임 판단에 중요한 음소들의 임베딩 (5차원)
# *Streamlit Cloud 환경에서도 동일하게 작동합니다.*
PHONEME_EMBEDDINGS = {
    'æ': np.array([1.0, 0.0, 0.5, 0.0, 0.0]),
    'ʌ': np.array([0.9, 0.1, 0.6, 0.0, 0.0]),
    'aɪ': np.array([0.5, 0.8, 0.1, 0.0, 0.0]),
    'ɛ': np.array([0.4, 0.7, 0.2, 0.0, 0.0]),
    'i': np.array([0.2, 0.9, 0.1, 0.0, 0.0]),
    't': np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
    'd': np.array([0.0, 0.9, 0.0, 1.0, 0.1]),
    'n': np.array([0.0, 0.8, 0.1, 1.0, 0.0]),
    'nd': np.array([0.0, 0.7, 0.1, 1.0, 0.1]),
    'r': np.array([0.1, 0.0, 0.8, 0.5, 0.5]),
}

# (데모를 위한 근사 라임 후보 목록)
NEAR_RHYME_CANDIDATES = {
    'mind': ['kind', 'spend', 'night', 'lend', 'signed'],
    'tough': ['stuff', 'rough', 'glove', 'love', 'cuff'],
    'ocean': ['motion', 'lotion', 'open', 'frozen'],
    'heart': ['start', 'dark', 'spark', 'art', 'part'], # 에미넴 스타일 추가
}

# =========================================================
# 2. 핵심 계산 함수 (rhyme_api.py의 로직과 동일)
# =========================================================

def text_to_ipa_clean(word):
    """주어진 단어를 IPA로 변환하고 스트레스 마크를 제거합니다."""
    try:
        pronunciation = ipa.convert(word.lower()).split(' ')[0]
        clean_ipa = pronunciation.replace('ˈ', '').replace('ˌ', '').strip()
        return clean_ipa
    except Exception:
        return None

def calculate_rhyme_score(ipa1, ipa2):
    """두 IPA 문자열의 벡터 유사도 점수 (코사인 유사도)를 계산합니다."""
    phons1 = list(ipa1)[-3:]
    phons2 = list(ipa2)[-3:]
    
    if not phons1 or not phons2 or len(phons1) != len(phons2):
        return 0.0
    
    vec1_list = [PHONEME_EMBEDDINGS.get(p, np.zeros(5)) for p in phons1]
    vec2_list = [PHONEME_EMBEDDINGS.get(p, np.zeros(5)) for p in phons2]
    
    vec1 = np.concatenate(vec1_list)
    vec2 = np.concatenate(vec2_list)
    
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0

    similarity = 1 - cosine(vec1, vec2)
    return max(0, similarity)

# @st.cache_data(show_spinner=False) <-- 오류 발생으로 인해 캐싱 데코레이터를 제거했습니다.
def get_rhyme_candidates_with_score(target_word: str):
    """대상 단어의 IPA와 근사 라임 후보 및 점수를 반환합니다."""
    target_ipa = text_to_ipa_clean(target_word)
    
    if not target_ipa:
        return {"target_word": target_word, "target_ipa": "N/A", "candidates": []}

    candidates_list = []
    candidates = NEAR_RHYME_CANDIDATES.get(target_word.lower(), [])
    
    for candidate_word in candidates:
        candidate_ipa = text_to_ipa_clean(candidate_word)
        if candidate_ipa:
            score = calculate_rhyme_score(target_ipa, candidate_ipa)
            candidates_list.append({
                "word": candidate_word,
                "score": round(score, 2),
                "ipa": candidate_ipa
            })
            
    candidates_list.sort(key=lambda x: x['score'], reverse=True)

    return {
        "target_word": target_word,
        "target_ipa": target_ipa,
        "candidates": candidates_list
    }


# =========================================================
# 3. Streamlit UI (사용자가 결과를 볼 수 있는 화면)
# =========================================================

st.set_page_config(page_title="Phonetics Analyzer (Gemini Tool Demo)", layout="centered")

st.title("🎤 Gemini Near Rhyme Tool (Streamlit Version)")
st.caption("음소 임베딩 기반 근사 라임 분석 데모")

st.markdown("""
이 앱은 Gemini가 외부 도구로 활용할 API의 **계산 로직**을 포함하고 있습니다.
Streamlit Cloud 배포 후 얻게 될 주소가 Gemini가 호출할 **Public API URL**이 됩니다.
""")

# 사용자 입력
input_word = st.text_input("분석할 단어를 입력하세요 (예: mind, tough, heart)", "mind")

if input_word:
    st.subheader(f"🔍 '{input_word}'에 대한 음소 분석 결과")
    
    # 계산 로직 실행
    analysis_result = get_rhyme_candidates_with_score(input_word)
    
    # 1. IPA 표시
    st.markdown(f"**대상 단어 IPA:** `{analysis_result['target_ipa']}`")
    
    # 2. 유사도 테이블 표시
    st.markdown("---")
    st.markdown("#### 근사 라임 후보 및 임베딩 유사도 점수")
    
    if analysis_result['candidates']:
        # 테이블 데이터 준비
        data = []
        for c in analysis_result['candidates']:
            rhyme_type = "Perfect Rhyme" if c['score'] >= 0.99 else ("Near Rhyme" if c['score'] >= 0.70 else "Poor Match")
            data.append({
                "Word": c['word'],
                "IPA": c['ipa'],
                "Phonetic Score": f"{c['score']:.2f}",
                "Rhyme Type": rhyme_type
            })
        
        st.dataframe(data, use_container_width=True, hide_index=True)
    else:
        st.warning("후보 단어를 찾지 못했거나 계산에 실패했습니다.")

    # 3. Gemini가 받을 API 응답 (발표 강조점)
    st.markdown("---")
    st.markdown("#### 🤖 Gemini가 받게 될 최종 API 응답 (JSON)")
    st.code(json.dumps(analysis_result, indent=2), language='json')