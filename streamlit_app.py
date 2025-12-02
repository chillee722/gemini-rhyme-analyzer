import streamlit as st
import eng_to_ipa as ipa
import numpy as np
from scipy.spatial.distance import cosine
import json
import nltk
from nltk.corpus import cmudict
import sys
import os

# --- CMUDict 다운로드 및 로드 ---
# Python 3.13 호환성을 위해 NLTK 예외 처리를 변경합니다.
@st.cache_resource
def load_cmudict():
    try:
        # CMUDict 데이터가 로컬에 있는지 확인합니다.
        nltk.data.find('corpora/cmudict')
    except LookupError: # <--- NLTK 데이터가 없을 때 발생하는 일반적인 예외
        # 데이터가 없으면 다운로드합니다. (Streamlit Cloud에서 자동 실행)
        nltk.download('cmudict')
    except AttributeError:
        # 매우 오래된 버전에서 발생하는 예외를 대비합니다.
        nltk.download('cmudict')
        
    return cmudict.dict()

p_dict = load_cmudict()

# =========================================================
# 1. 음소 임베딩 정의 (유사도 계산의 핵심 데이터)
# =========================================================

# (예시) 근사 라임 판단에 중요한 음소들의 임베딩 (5차원)
PHONEME_EMBEDDINGS = {
    'æ': np.array([1.0, 0.0, 0.5, 0.0, 0.0]),  # as in 'cat', 'hat'
    'ʌ': np.array([0.9, 0.1, 0.6, 0.0, 0.0]),  # as in 'cut'
    'aɪ': np.array([0.5, 0.8, 0.1, 0.0, 0.0]), # as in 'mind'
    'ɛ': np.array([0.4, 0.7, 0.2, 0.0, 0.0]),  # as in 'spend'
    'i': np.array([0.2, 0.9, 0.1, 0.0, 0.0]),  # as in 'feel'
    't': np.array([0.0, 1.0, 0.0, 1.0, 0.0]),  # Consonant 't'
    'd': np.array([0.0, 0.9, 0.0, 1.0, 0.1]),  # Consonant 'd' (t와 유사)
    'n': np.array([0.0, 0.8, 0.1, 1.0, 0.0]),  # Consonant 'n'
    'nd': np.array([0.0, 0.7, 0.1, 1.0, 0.1]), # Consonant cluster 'nd'
    'r': np.array([0.1, 0.0, 0.8, 0.5, 0.5]),
    'ʊ': np.array([0.3, 0.6, 0.4, 0.0, 0.0]),  # as in 'good'
    'k': np.array([0.0, 0.5, 0.0, 0.5, 0.0]),  # Consonant 'k'
}

# =========================================================
# 2. 핵심 계산 함수 (CMUDict 활용 로직)
# =========================================================

@st.cache_data(show_spinner=False)
def get_arpabet_and_rhyme_unit(word):
    """CMUDict에서 단어의 ARPAbet 발음과 라임 유닛을 추출합니다."""
    word = word.lower()
    if word not in p_dict:
        return None, None 

    pron = p_dict[word][0] 
    
    rhyme_start_index = -1
    for i, phon in enumerate(pron):
        if phon[-1] in ('1', '2'): 
            rhyme_start_index = i
            break
            
    if rhyme_start_index == -1:
        return None, None 

    clean_arpabet = [phon.rstrip('0123') for phon in pron]
    rhyme_unit = clean_arpabet[rhyme_start_index:]
    
    return clean_arpabet, rhyme_unit

def arpabet_to_ipa(arpabet_phons):
    """ARPAbet 음소열을 eng-to-ipa를 사용하여 IPA 문자열로 변환합니다."""
    arpabet_str = ', '.join(arpabet_phons)
    try:
        ipa_str = ipa.convert(arpabet_str, mode='arpabet').strip().replace(' ', '').replace('ˈ', '').replace('ˌ', '')
        return ipa_str
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


@st.cache_data(show_spinner=False)
def get_rhyme_candidates_with_score(target_word: str, top_n=100):
    """CMUDict 전체를 검색하여 라임 유닛이 일치하는 후보를 찾고 점수를 매깁니다."""
    
    target_arpabet, target_rhyme_unit = get_arpabet_and_rhyme_unit(target_word)
    
    if not target_rhyme_unit:
        return {"target_word": target_word, "target_ipa": "N/A", "candidates": []}

    target_ipa = arpabet_to_ipa(target_rhyme_unit)
    
    candidates_list = []
    
    for word, pron_list in p_dict.items():
        pron_arpabet = pron_list[0]
        pron_clean = [p.rstrip('0123') for p in pron_arpabet]
        
        if word == target_word.lower() or len(word) <= 2 or len(pron_clean) < len(target_rhyme_unit): 
            continue
            
        if pron_clean[-len(target_rhyme_unit):] == target_rhyme_unit:
            
            candidate_ipa = arpabet_to_ipa(pron_clean)
            if candidate_ipa:
                score = calculate_rhyme_score(target_ipa, candidate_ipa) 
                
                candidates_list.append({
                    "word": word,
                    "score": round(score, 2),
                    "ipa": candidate_ipa
                })

    candidates_list.sort(key=lambda x: x['score'], reverse=True)

    return {
        "target_word": target_word,
        "target_ipa": target_ipa,
        "candidates": candidates_list[:top_n]
    }


# =========================================================
# 3. Streamlit UI (CMUDict가 활성화된 UI)
# =========================================================

st.set_page_config(page_title="Phonetics Analyzer (CMUDict Integrated)", layout="centered")

st.title("🎤 CMUDict 통합: 음소 임베딩 근사 라임 분석")
st.caption("✅ CMUDict 기반으로 전체 영어 단어 검색 가능 (기말 프로젝트 개선 사항 반영)")

st.markdown("""
이 툴은 **수동 딕셔너리** 대신 **CMUDict (13만 단어)**를 활용하여, 
입력 단어와 **음소 유사성**이 높은 모든 단어를 검색하고 점수를 매깁니다. 
이 JSON 결과가 Gemini에게 제공할 **API 응답**입니다.
""")

# 사용자 입력
input_word = st.text_input("분석할 단어를 입력하세요 (예: tough, mind, heart)", "mind")

if input_word:
    st.subheader(f"🔍 '{input_word}'에 대한 CMUDict 기반 분석 결과")
    
    # 계산 로직 실행
    with st.spinner('CMUDict를 스캔하고 음소 임베딩을 계산 중...'):
        analysis_result = get_rhyme_candidates_with_score(input_word)
    
    # 1. IPA 표시
    st.markdown(f"**대상 단어 IPA (라임 유닛):** `{analysis_result['target_ipa']}`")
    
    # 2. 유사도 테이블 표시
    st.markdown("---")
    st.markdown("#### CMUDict 기반 검색 결과 (상위 100개 중 점수 순 정렬)")
    
    if analysis_result['candidates']:
        # 테이블 데이터 준비
        data = []
        for c in analysis_result['candidates']:
            rhyme_type = "Perfect Rhyme" if c['score'] >= 0.99 else ("Near Rhyme" if c['score'] >= 0.70 else "Slant/Poor Match")
            data.append({
                "Word": c['word'],
                "IPA (Phonetics)": c['ipa'],
                "Phonetic Score": f"{c['score']:.2f}",
                "Rhyme Type": rhyme_type
            })
        
        st.dataframe(data, use_container_width=True, hide_index=True)
    else:
        st.warning(f"CMUDict에서 '{input_word}'에 대한 라임 유닛을 찾지 못했습니다. (단어가 사전에 없거나 너무 짧을 수 있습니다.)")

    # 3. Gemini가 받을 API 응답 (발표 강조점)
    st.markdown("---")
    st.markdown("#### 🤖 Gemini에게 제공할 최종 API 응답 (JSON)")
    st.code(json.dumps(analysis_result, indent=2), language='json')
